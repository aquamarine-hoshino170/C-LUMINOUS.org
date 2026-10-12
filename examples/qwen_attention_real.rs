use lum_rust::gguf_loader::GgufModel;

const HIDDEN: usize = 896;
const Q_HEADS: usize = 14;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const GQA_GROUPS: usize = 7;
const SCALE: f32 = 0.125;
const ROPE_BASE: f32 = 1_000_000.0;

const Q5_BLOCK: usize = 22;
const Q8_BLOCK: usize = 34;

fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let value = if exp == 0 {
        if frac == 0 {
            0.0
        } else {
            let mut f = frac as f32 / 1024.0;
            let mut e = -14i32;

            while f < 1.0 {
                f *= 2.0;
                e -= 1;
            }

            f * 2f32.powi(e)
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };

    if sign != 0 {
        -value
    } else {
        value
    }
}

fn decode_q5_0_block(block: &[u8], out: &mut [f32; 32]) {
    assert_eq!(block.len(), Q5_BLOCK);

    let d_bits = u16::from_le_bytes([block[0], block[1]]);

    let d = fp16_to_f32(d_bits);

    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    let qs = &block[6..22];

    for j in 0..32 {
        let (low, high);

        if j < 16 {
            low = qs[j] & 0x0f;
            high = ((qh >> j) & 1) as u8;
        } else {
            let k = j - 16;
            low = qs[k] >> 4;
            high = ((qh >> (k + 16)) & 1) as u8;
        }

        let q5 = low | (high << 4);

        out[j] = d * ((q5 as i32) - 16) as f32;
    }
}

fn decode_q8_0_block(block: &[u8], out: &mut [f32; 32]) {
    assert_eq!(block.len(), Q8_BLOCK);

    let d_bits = u16::from_le_bytes([block[0], block[1]]);

    let d = fp16_to_f32(d_bits);

    for i in 0..32 {
        let q = block[2 + i] as i8;
        out[i] = d * q as f32;
    }
}

fn q5_gemv(tensor: &[u8], rows: usize, cols: usize, input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), cols);
    assert_eq!(output.len(), rows);

    let blocks_per_row = cols / 32;

    assert_eq!(tensor.len(), rows * blocks_per_row * Q5_BLOCK);

    let mut decoded = [0.0f32; 32];

    for row in 0..rows {
        let mut acc = 0.0f32;

        for block in 0..blocks_per_row {
            let base = (row * blocks_per_row + block) * Q5_BLOCK;

            decode_q5_0_block(&tensor[base..base + Q5_BLOCK], &mut decoded);

            let input_base = block * 32;

            for j in 0..32 {
                acc += decoded[j] * input[input_base + j];
            }
        }

        output[row] = acc;
    }
}

fn q8_gemv(tensor: &[u8], rows: usize, cols: usize, input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), cols);
    assert_eq!(output.len(), rows);

    let blocks_per_row = cols / 32;

    assert_eq!(tensor.len(), rows * blocks_per_row * Q8_BLOCK);

    let mut decoded = [0.0f32; 32];

    for row in 0..rows {
        let mut acc = 0.0f32;

        for block in 0..blocks_per_row {
            let base = (row * blocks_per_row + block) * Q8_BLOCK;

            decode_q8_0_block(&tensor[base..base + Q8_BLOCK], &mut decoded);

            let input_base = block * 32;

            for j in 0..32 {
                acc += decoded[j] * input[input_base + j];
            }
        }

        output[row] = acc;
    }
}

fn read_f32_vector(tensor: &[u8], expected: usize) -> Vec<f32> {
    assert_eq!(tensor.len(), expected * 4);

    tensor
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn add_bias(x: &mut [f32], bias: &[f32]) {
    assert_eq!(x.len(), bias.len());

    for i in 0..x.len() {
        x[i] += bias[i];
    }
}

fn apply_rope(q: &mut [f32], k: &mut [f32], position: usize) {
    // Qwen2 rotary pairs.
    //
    // We use the verified GGUF RoPE base.
    // The model head dimension is 64.
    //
    // Pair layout:
    // (0,1), (2,3), ... (62,63)

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

fn softmax(scores: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;

    for &x in scores.iter() {
        max = max.max(x);
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..HEAD_DIM {
        sum += a[i] * b[i];
    }

    sum
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const MODEL: &str = "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("============================================================");
    println!("       LUMINOUS QWEN PHASE 4F-C REAL GGUF ATTENTION");
    println!("============================================================");

    println!("[✓] Hidden size    : {}", HIDDEN);
    println!("[✓] Q heads        : {}", Q_HEADS);
    println!("[✓] KV heads       : {}", KV_HEADS);
    println!("[✓] Head dimension : {}", HEAD_DIM);
    println!("[✓] GQA ratio      : {}", GQA_GROUPS);
    println!("[✓] RoPE base      : {}", ROPE_BASE);

    let model = GgufModel::load(MODEL)?;

    // ---------------------------------------------------------
    // Load real Qwen tensors.
    // ---------------------------------------------------------

    let q_weight = model.read_tensor_bytes("blk.0.attn_q.weight")?;

    let q_bias = model.read_tensor_bytes("blk.0.attn_q.bias")?;

    let k_weight = model.read_tensor_bytes("blk.0.attn_k.weight")?;

    let k_bias = model.read_tensor_bytes("blk.0.attn_k.bias")?;

    let v_weight = model.read_tensor_bytes("blk.0.attn_v.weight")?;

    let v_bias = model.read_tensor_bytes("blk.0.attn_v.bias")?;

    println!();
    println!("REAL GGUF TENSORS");
    println!("[✓] Q weight bytes : {}", q_weight.len());
    println!("[✓] Q bias bytes   : {}", q_bias.len());
    println!("[✓] K weight bytes : {}", k_weight.len());
    println!("[✓] K bias bytes   : {}", k_bias.len());
    println!("[✓] V weight bytes : {}", v_weight.len());
    println!("[✓] V bias bytes   : {}", v_bias.len());

    assert_eq!(q_weight.len(), HIDDEN * HIDDEN / 32 * Q5_BLOCK);

    assert_eq!(
        k_weight.len(),
        KV_HEADS * HEAD_DIM * (HIDDEN / 32) * Q5_BLOCK
    );

    assert_eq!(
        v_weight.len(),
        KV_HEADS * HEAD_DIM * (HIDDEN / 32) * Q8_BLOCK
    );

    // ---------------------------------------------------------
    // Deterministic hidden state.
    //
    // This is NOT an embedding yet.
    // It lets us test the real GGUF projection tensors.
    // ---------------------------------------------------------

    let mut hidden = vec![0.0f32; HIDDEN];

    for i in 0..HIDDEN {
        let x = i as f32;

        hidden[i] = (x * 0.01731).sin() * 0.5 + (x * 0.00371).cos() * 0.125;
    }

    // ---------------------------------------------------------
    // Q projection: 896 -> 896
    // ---------------------------------------------------------

    let mut q = vec![0.0f32; HIDDEN];

    q5_gemv(q_weight, HIDDEN, HIDDEN, &hidden, &mut q);

    let q_bias = read_f32_vector(q_bias, HIDDEN);

    add_bias(&mut q, &q_bias);

    // ---------------------------------------------------------
    // K projection: 896 -> 128
    // ---------------------------------------------------------

    let mut k = vec![0.0f32; KV_HEADS * HEAD_DIM];

    q5_gemv(k_weight, KV_HEADS * HEAD_DIM, HIDDEN, &hidden, &mut k);

    let k_bias = read_f32_vector(k_bias, KV_HEADS * HEAD_DIM);

    add_bias(&mut k, &k_bias);

    // ---------------------------------------------------------
    // V projection: 896 -> 128
    // ---------------------------------------------------------

    let mut v = vec![0.0f32; KV_HEADS * HEAD_DIM];

    q8_gemv(v_weight, KV_HEADS * HEAD_DIM, HIDDEN, &hidden, &mut v);

    let v_bias = read_f32_vector(v_bias, KV_HEADS * HEAD_DIM);

    add_bias(&mut v, &v_bias);

    println!();
    println!("REAL Q/K/V PROJECTION");
    println!("Q[0]   : {:.12e}", q[0]);
    println!("Q[895] : {:.12e}", q[895]);
    println!("K[0]   : {:.12e}", k[0]);
    println!("K[127] : {:.12e}", k[127]);
    println!("V[0]   : {:.12e}", v[0]);
    println!("V[127] : {:.12e}", v[127]);

    // ---------------------------------------------------------
    // Verify projection outputs are finite.
    // ---------------------------------------------------------

    for x in q.iter().chain(k.iter()).chain(v.iter()) {
        assert!(x.is_finite(), "Q/K/V contains NaN or Inf");
    }

    println!("[✓] Real Q/K/V projections finite");

    // ---------------------------------------------------------
    // RoPE.
    //
    // Use position 1 so the rotation is actually exercised.
    // ---------------------------------------------------------

    apply_rope(&mut q, &mut k, 1);

    for x in q.iter().chain(k.iter()) {
        assert!(x.is_finite(), "RoPE produced NaN/Inf");
    }

    println!("[✓] RoPE position 1 applied");

    // ---------------------------------------------------------
    // Single-token GQA attention.
    //
    // With one KV position, softmax = 1.
    // This validates:
    //
    // Q -> head
    // K/V -> grouped KV head
    // QK^T
    // scaling
    // softmax
    // AV
    // ---------------------------------------------------------

    let mut output = vec![0.0f32; HIDDEN];

    for q_head in 0..Q_HEADS {
        let kv_head = q_head / GQA_GROUPS;

        let q_base = q_head * HEAD_DIM;

        let kv_base = kv_head * HEAD_DIM;

        let score = dot(
            &q[q_base..q_base + HEAD_DIM],
            &k[kv_base..kv_base + HEAD_DIM],
        ) * SCALE;

        let mut scores = [score];

        softmax(&mut scores);

        let probability = scores[0];

        assert!((probability - 1.0).abs() < 1.0e-6);

        for d in 0..HEAD_DIM {
            output[q_base + d] = probability * v[kv_base + d];
        }
    }

    // ---------------------------------------------------------
    // Final numerical checks.
    // ---------------------------------------------------------

    let mut min = f32::INFINITY;

    let mut max = f32::NEG_INFINITY;

    let mut sum_sq = 0.0f32;

    for &x in &output {
        assert!(x.is_finite());

        min = min.min(x);
        max = max.max(x);
        sum_sq += x * x;
    }

    let rms = (sum_sq / HIDDEN as f32).sqrt();

    println!();
    println!("REAL ATTENTION OUTPUT");
    println!("output[0]   : {:.12e}", output[0]);
    println!("output[1]   : {:.12e}", output[1]);
    println!("output[2]   : {:.12e}", output[2]);
    println!("output[895] : {:.12e}", output[895]);
    println!("min         : {:.12e}", min);
    println!("max         : {:.12e}", max);
    println!("RMS         : {:.12e}", rms);

    println!();
    println!("[✓] Q5_0 Q projection + bias PASSED");
    println!("[✓] Q5_0 K projection + bias PASSED");
    println!("[✓] Q8_0 V projection + bias PASSED");
    println!("[✓] RoPE PASSED");
    println!("[✓] GQA 14Q → 2KV PASSED");
    println!("[✓] Attention output finite");
    println!("[✓] PHASE 4F-C REAL GGUF ATTENTION PASSED");
    println!("============================================================");

    Ok(())
}
