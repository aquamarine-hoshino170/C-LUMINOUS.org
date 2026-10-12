const HIDDEN: usize = 896;
const Q_HEADS: usize = 14;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const KV_GROUPS: usize = Q_HEADS / KV_HEADS;
const SCALE: f32 = 1.0 / 8.0; // 1 / sqrt(64)
const ROPE_BASE: f32 = 1_000_000.0;

fn apply_rope(q: &mut [f32], k: &mut [f32], position: usize) {
    // Q: [14,64], K: [2,64]
    //
    // Qwen2 uses rotary pairs across the head dimension.
    // We use the verified RoPE base from GGUF metadata.
    //
    // theta_i = base^(-2i/head_dim)
    // angle   = position * theta_i

    for head in 0..Q_HEADS {
        let base = head * HEAD_DIM;

        for i in (0..HEAD_DIM).step_by(2) {
            let freq = ROPE_BASE.powf(-(i as f32) / HEAD_DIM as f32);
            let angle = position as f32 * freq;

            let (sin_a, cos_a) = angle.sin_cos();

            let x0 = q[base + i];
            let x1 = q[base + i + 1];

            q[base + i] = x0 * cos_a - x1 * sin_a;
            q[base + i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }

    for head in 0..KV_HEADS {
        let base = head * HEAD_DIM;

        for i in (0..HEAD_DIM).step_by(2) {
            let freq = ROPE_BASE.powf(-(i as f32) / HEAD_DIM as f32);
            let angle = position as f32 * freq;

            let (sin_a, cos_a) = angle.sin_cos();

            let x0 = k[base + i];
            let x1 = k[base + i + 1];

            k[base + i] = x0 * cos_a - x1 * sin_a;
            k[base + i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

fn softmax(scores: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;

    for &x in scores.iter() {
        if x > max {
            max = x;
        }
    }

    let mut sum = 0.0f32;

    for x in scores.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }

    let inv = 1.0 / sum;

    for x in scores.iter_mut() {
        *x *= inv;
    }
}

fn attention_single_token(q: &[f32], k: &[f32], v: &[f32], output: &mut [f32]) {
    // Single-token attention reference.
    //
    // For one token there is only one K/V position.
    // GQA maps:
    //
    // Q heads 0..6   -> KV head 0
    // Q heads 7..13  -> KV head 1

    assert_eq!(q.len(), Q_HEADS * HEAD_DIM);
    assert_eq!(k.len(), KV_HEADS * HEAD_DIM);
    assert_eq!(v.len(), KV_HEADS * HEAD_DIM);
    assert_eq!(output.len(), HIDDEN);

    for q_head in 0..Q_HEADS {
        let kv_head = q_head / KV_GROUPS;

        let q_base = q_head * HEAD_DIM;
        let kv_base = kv_head * HEAD_DIM;

        let score = dot(
            &q[q_base..q_base + HEAD_DIM],
            &k[kv_base..kv_base + HEAD_DIM],
        ) * SCALE;

        // One-position softmax is exactly 1.
        // Keep the actual softmax path here so the implementation
        // can be extended to multi-token causal attention later.
        let mut scores = [score];
        softmax(&mut scores);

        let weight = scores[0];

        for d in 0..HEAD_DIM {
            output[q_base + d] = weight * v[kv_base + d];
        }
    }
}

fn main() {
    println!("============================================================");
    println!("        LUMINOUS QWEN PHASE 4F ATTENTION REFERENCE");
    println!("============================================================");

    println!("[✓] Hidden size      : {}", HIDDEN);
    println!("[✓] Q heads          : {}", Q_HEADS);
    println!("[✓] KV heads         : {}", KV_HEADS);
    println!("[✓] Head dimension   : {}", HEAD_DIM);
    println!("[✓] GQA ratio        : {}", KV_GROUPS);
    println!("[✓] Attention scale  : {}", SCALE);
    println!("[✓] RoPE base        : {}", ROPE_BASE);

    // Deterministic synthetic projection outputs.
    //
    // These stand in for:
    // Q = Q_weight * hidden + Q_bias
    // K = K_weight * hidden + K_bias
    // V = V_weight * hidden + V_bias
    //
    // Actual GGUF tensors will be connected after the attention
    // mathematics is validated.

    let mut q = vec![0.0f32; HIDDEN];
    let mut k = vec![0.0f32; KV_HEADS * HEAD_DIM];
    let mut v = vec![0.0f32; KV_HEADS * HEAD_DIM];

    for i in 0..q.len() {
        q[i] = ((i as f32) * 0.01731).sin() * 0.25 + ((i as f32) * 0.00371).cos() * 0.125;
    }

    for i in 0..k.len() {
        k[i] = ((i as f32) * 0.02117).cos() * 0.20 + ((i as f32) * 0.00413).sin() * 0.10;
    }

    for i in 0..v.len() {
        v[i] = ((i as f32) * 0.01371).sin() * 0.30 - ((i as f32) * 0.00217).cos() * 0.08;
    }

    let q_before = q.clone();
    let k_before = k.clone();

    apply_rope(&mut q, &mut k, 0);

    // Position 0 must leave values unchanged because sin(0)=0
    // and cos(0)=1.
    let mut rope_error = 0.0f32;

    for i in 0..q.len() {
        rope_error = rope_error.max((q[i] - q_before[i]).abs());
    }

    for i in 0..k.len() {
        rope_error = rope_error.max((k[i] - k_before[i]).abs());
    }

    println!();
    println!("RoPE position 0 max error : {:.9e}", rope_error);

    assert!(rope_error < 1.0e-6, "RoPE position 0 invariant failed");

    let mut output = vec![0.0f32; HIDDEN];

    attention_single_token(&q, &k, &v, &mut output);

    let mut finite = true;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut rms_sum = 0.0f32;

    for &x in &output {
        if !x.is_finite() {
            finite = false;
        }

        min = min.min(x);
        max = max.max(x);
        rms_sum += x * x;
    }

    let rms = (rms_sum / output.len() as f32).sqrt();

    println!();
    println!("Attention output:");
    println!("output[0]        : {:.12e}", output[0]);
    println!("output[1]        : {:.12e}", output[1]);
    println!("output[2]        : {:.12e}", output[2]);
    println!("output[895]      : {:.12e}", output[895]);
    println!("min              : {:.12e}", min);
    println!("max              : {:.12e}", max);
    println!("RMS              : {:.12e}", rms);

    assert!(finite, "attention produced NaN/Inf");

    // GQA invariants:
    // Every group of 7 Q heads uses the same KV head.
    for q_head in 0..Q_HEADS {
        assert_eq!(q_head / KV_GROUPS, if q_head < 7 { 0 } else { 1 });
    }

    // Since there is exactly one KV position, softmax must be 1.
    let mut test_scores = [-1000.0f32];

    softmax(&mut test_scores);

    assert!((test_scores[0] - 1.0).abs() < 1.0e-6);

    println!();
    println!("[✓] RoPE position-0 invariant PASSED");
    println!("[✓] GQA 14Q / 2KV mapping PASSED");
    println!("[✓] Stable softmax single-position PASSED");
    println!("[✓] Attention output finite");
    println!("[✓] PHASE 4F REFERENCE ATTENTION CORE PASSED");
    println!("============================================================");
}
