use lum_rust::gguf_loader::GgufModel;

const H: usize = 896;
const FFN: usize = 4864;
const QH: usize = 14;
const KVH: usize = 2;
const HD: usize = 64;
const GROUP: usize = 7;

const Q5B: usize = 22;
const Q8B: usize = 34;
const Q6B: usize = 210;
const EPS: f32 = 1.0e-6;
const ROPE_BASE: f32 = 1_000_000.0;

fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 31) as u32;
    let frac = (bits & 1023) as u32;

    let v = if exp == 0 {
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
        -v
    } else {
        v
    }
}

fn rms(x: &[f32]) -> f32 {
    let mut s = 0.0;
    for &v in x {
        s += v * v;
    }
    (s / x.len() as f32).sqrt()
}

fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut ss = 0.0;
    for &v in x {
        ss += v * v;
    }

    let inv = 1.0 / (ss / H as f32 + EPS).sqrt();

    x.iter().zip(w).map(|(&a, &b)| a * inv * b).collect()
}

// ------------------------------------------------------------
// Q5_0
// ------------------------------------------------------------

fn q5_decode(block: &[u8], out: &mut [f32; 32]) {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    let qs = &block[6..22];

    for j in 0..32 {
        let (low, high) = if j < 16 {
            (qs[j] & 0x0f, ((qh >> j) & 1) as u8)
        } else {
            let k = j - 16;
            (qs[k] >> 4, ((qh >> (k + 16)) & 1) as u8)
        };

        let q = low | (high << 4);
        out[j] = d * ((q as i32) - 16) as f32;
    }
}

fn q5_gemv(tensor: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    assert_eq!(cols % 32, 0);
    assert_eq!(tensor.len(), rows * (cols / 32) * Q5B);

    let mut out = vec![0.0; rows];
    let mut block = [0.0f32; 32];

    for r in 0..rows {
        let mut acc = 0.0;

        for b in 0..cols / 32 {
            let base = (r * (cols / 32) + b) * Q5B;

            q5_decode(&tensor[base..base + Q5B], &mut block);

            for j in 0..32 {
                acc += block[j] * input[b * 32 + j];
            }
        }

        out[r] = acc;
    }

    out
}

// ------------------------------------------------------------
// Q8_0
// ------------------------------------------------------------

fn q8_gemv(tensor: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    assert_eq!(cols % 32, 0);
    assert_eq!(tensor.len(), rows * (cols / 32) * Q8B);

    let mut out = vec![0.0; rows];

    for r in 0..rows {
        let mut acc = 0.0;

        for b in 0..cols / 32 {
            let base = (r * (cols / 32) + b) * Q8B;

            let d = fp16_to_f32(u16::from_le_bytes([tensor[base], tensor[base + 1]]));

            for j in 0..32 {
                let q = tensor[base + 2 + j] as i8;
                acc += d * q as f32 * input[b * 32 + j];
            }
        }

        out[r] = acc;
    }

    out
}

// ------------------------------------------------------------
// Q6_K canonical 210-byte block
// ------------------------------------------------------------

fn q6k_decode(block: &[u8], out: &mut [f32; 256]) {
    assert_eq!(block.len(), Q6B);

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];

    let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    // Q6_K has two 128-value halves.
    //
    // Each half uses:
    //   64 ql bytes
    //   32 qh bytes
    //    8 scale bytes
    //
    // Each 32-byte ql region contains low/high nibbles,
    // producing four groups of 32 values.

    for half in 0..2 {
        let ql0 = half * 64;
        let qh0 = half * 32;
        let sc0 = half * 8;
        let out0 = half * 128;

        for l in 0..32 {
            let is = l / 16;

            let ql_a = ql[ql0 + l];
            let ql_b = ql[ql0 + l + 32];

            let h = qh[qh0 + l];

            let q1 = (ql_a & 0x0f) | ((h & 0x03) << 4);

            let q2 = (ql_b & 0x0f) | (((h >> 2) & 0x03) << 4);

            let q3 = (ql_a >> 4) | (((h >> 4) & 0x03) << 4);

            let q4 = (ql_b >> 4) | (((h >> 6) & 0x03) << 4);

            let d1 = d * scales[sc0 + is] as i8 as f32;

            let d2 = d * scales[sc0 + is + 2] as i8 as f32;

            let d3 = d * scales[sc0 + is + 4] as i8 as f32;

            let d4 = d * scales[sc0 + is + 6] as i8 as f32;

            out[out0 + l] = d1 * (q1 as i32 - 32) as f32;

            out[out0 + l + 32] = d2 * (q2 as i32 - 32) as f32;

            out[out0 + l + 64] = d3 * (q3 as i32 - 32) as f32;

            out[out0 + l + 96] = d4 * (q4 as i32 - 32) as f32;
        }
    }
}

fn q6_gemv(tensor: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    assert_eq!(cols % 256, 0);
    assert_eq!(tensor.len(), rows * (cols / 256) * Q6B);

    let mut out = vec![0.0; rows];
    let mut block = [0.0f32; 256];

    for r in 0..rows {
        let mut acc = 0.0;

        for b in 0..cols / 256 {
            let base = (r * (cols / 256) + b) * Q6B;

            q6k_decode(&tensor[base..base + Q6B], &mut block);

            for j in 0..256 {
                acc += block[j] * input[b * 256 + j];
            }
        }

        out[r] = acc;
    }

    out
}

// ------------------------------------------------------------
// Bias
// ------------------------------------------------------------

fn add_bias(x: &mut [f32], bias: &[f32]) {
    for (a, &b) in x.iter_mut().zip(bias) {
        *a += b;
    }
}

// ------------------------------------------------------------
// RoPE
// ------------------------------------------------------------

fn rope(x: &mut [f32], position: usize) {
    for h in 0..QH {
        let base = h * HD;

        for i in (0..HD).step_by(2) {
            let theta = position as f32 / ROPE_BASE.powf(i as f32 / HD as f32);

            let c = theta.cos();
            let s = theta.sin();

            let a = x[base + i];
            let b = x[base + i + 1];

            x[base + i] = a * c - b * s;

            x[base + i + 1] = a * s + b * c;
        }
    }
}

fn rope_k(x: &mut [f32], position: usize) {
    for h in 0..KVH {
        let base = h * HD;

        for i in (0..HD).step_by(2) {
            let theta = position as f32 / ROPE_BASE.powf(i as f32 / HD as f32);

            let c = theta.cos();
            let s = theta.sin();

            let a = x[base + i];
            let b = x[base + i + 1];

            x[base + i] = a * c - b * s;

            x[base + i + 1] = a * s + b * c;
        }
    }
}

// ------------------------------------------------------------
// Single-token GQA attention
// ------------------------------------------------------------

fn attention(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; H];

    for qh in 0..QH {
        let kvh = qh / GROUP;

        let qb = qh * HD;
        let kb = kvh * HD;
        let vb = kvh * HD;

        let mut score = 0.0;

        for j in 0..HD {
            score += q[qb + j] * k[kb + j];
        }

        score *= 1.0 / (HD as f32).sqrt();

        // One-token sequence => softmax = 1.
        let _ = score;

        for j in 0..HD {
            out[qb + j] = v[vb + j];
        }
    }

    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const MODEL: &str = "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("============================================================");
    println!("          LUMINOUS QWEN COMPLETE BLOCK 0");
    println!("============================================================");

    let model = GgufModel::load(MODEL)?;

    let attn_norm = model.read_tensor_bytes("blk.0.attn_norm.weight")?;

    let q_w = model.read_tensor_bytes("blk.0.attn_q.weight")?;

    let q_b = model.read_tensor_bytes("blk.0.attn_q.bias")?;

    let k_w = model.read_tensor_bytes("blk.0.attn_k.weight")?;

    let k_b = model.read_tensor_bytes("blk.0.attn_k.bias")?;

    let v_w = model.read_tensor_bytes("blk.0.attn_v.weight")?;

    let v_b = model.read_tensor_bytes("blk.0.attn_v.bias")?;

    let o_w = model.read_tensor_bytes("blk.0.attn_output.weight")?;

    let ffn_norm = model.read_tensor_bytes("blk.0.ffn_norm.weight")?;

    let gate_w = model.read_tensor_bytes("blk.0.ffn_gate.weight")?;

    let up_w = model.read_tensor_bytes("blk.0.ffn_up.weight")?;

    let down_w = model.read_tensor_bytes("blk.0.ffn_down.weight")?;

    assert_eq!(q_w.len(), 896 * 28 * 22);
    assert_eq!(k_w.len(), 128 * 28 * 22);
    assert_eq!(v_w.len(), 128 * 28 * 34);
    assert_eq!(o_w.len(), 896 * 28 * 22);

    assert_eq!(gate_w.len(), FFN * 28 * 22);
    assert_eq!(up_w.len(), FFN * 28 * 22);
    assert_eq!(down_w.len(), H * 19 * 210);

    println!();
    println!("[✓] All Block-0 GGUF tensors loaded");
    println!("[✓] Q5_0 Q/K/O");
    println!("[✓] Q8_0 V");
    println!("[✓] Q5_0 Gate/Up");
    println!("[✓] Q6_K Down");
    println!("[✓] Qwen2.5 dimensions verified");

    // ---------------------------------------------------------
    // Deterministic hidden input
    // ---------------------------------------------------------

    let mut hidden = vec![0.0f32; H];

    for i in 0..H {
        let x = i as f32;

        hidden[i] = (x * 0.01731).sin() * 0.5 + (x * 0.00371).cos() * 0.125;
    }

    println!();
    println!("INPUT");
    println!("hidden RMS : {:.12e}", rms(&hidden));

    // ---------------------------------------------------------
    // ATTENTION SUB-BLOCK
    // ---------------------------------------------------------

    let norm1 = rmsnorm(&hidden, &decode_f32_vec(attn_norm));

    let mut q = q5_gemv(q_w, QH * HD, H, &norm1);

    let mut k = q5_gemv(k_w, KVH * HD, H, &norm1);

    let mut v = q8_gemv(v_w, KVH * HD, H, &norm1);

    add_bias(&mut q, &decode_f32_vec(q_b));

    add_bias(&mut k, &decode_f32_vec(k_b));

    add_bias(&mut v, &decode_f32_vec(v_b));

    rope(&mut q, 0);
    rope_k(&mut k, 0);

    let attn = attention(&q, &k, &v);

    let projected = q5_gemv(o_w, H, H, &attn);

    let mut after_attn = hidden.clone();

    for i in 0..H {
        after_attn[i] += projected[i];
    }

    println!();
    println!("ATTENTION SUB-BLOCK");
    println!("Q RMS          : {:.12e}", rms(&q));
    println!("K RMS          : {:.12e}", rms(&k));
    println!("V RMS          : {:.12e}", rms(&v));
    println!("Attention RMS  : {:.12e}", rms(&attn));
    println!("Projected RMS  : {:.12e}", rms(&projected));
    println!("Residual RMS   : {:.12e}", rms(&after_attn));

    assert!(after_attn.iter().all(|x| x.is_finite()));

    // ---------------------------------------------------------
    // FFN SUB-BLOCK
    // ---------------------------------------------------------

    let norm2 = rmsnorm(&after_attn, &decode_f32_vec(ffn_norm));

    let gate = q5_gemv(gate_w, FFN, H, &norm2);

    let up = q5_gemv(up_w, FFN, H, &norm2);

    let mut activated = vec![0.0f32; FFN];

    for i in 0..FFN {
        activated[i] = silu(gate[i]) * up[i];
    }

    let down = q6_gemv(down_w, H, FFN, &activated);

    let mut final_hidden = after_attn.clone();

    for i in 0..H {
        final_hidden[i] += down[i];
    }

    println!();
    println!("FFN SUB-BLOCK");
    println!("Gate RMS       : {:.12e}", rms(&gate));
    println!("Up RMS         : {:.12e}", rms(&up));
    println!("SwiGLU RMS     : {:.12e}", rms(&activated));
    println!("Down RMS       : {:.12e}", rms(&down));

    println!();
    println!("COMPLETE BLOCK 0");
    println!("Input RMS      : {:.12e}", rms(&hidden));
    println!("After Attention: {:.12e}", rms(&after_attn));
    println!("Final RMS      : {:.12e}", rms(&final_hidden));

    println!("output[0]      : {:.12e}", final_hidden[0]);

    println!("output[895]    : {:.12e}", final_hidden[895]);

    assert!(final_hidden.iter().all(|x| x.is_finite()));

    println!();
    println!("[✓] RMSNorm #1");
    println!("[✓] Q/K/V projections");
    println!("[✓] Q/K bias");
    println!("[✓] RoPE");
    println!("[✓] GQA 14Q → 2KV");
    println!("[✓] Attention");
    println!("[✓] Attention output projection");
    println!("[✓] Attention residual");
    println!("[✓] RMSNorm #2");
    println!("[✓] SwiGLU");
    println!("[✓] Q6_K FFN down projection");
    println!("[✓] FFN residual");
    println!();
    println!("============================================================");
    println!("       [✓] PHASE 4I BLOCK 0 EXECUTION COMPLETE");
    println!("============================================================");

    Ok(())
}

fn decode_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);

    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
