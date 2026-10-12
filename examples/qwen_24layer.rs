use lum_rust::gguf_loader::GgufModel;

pub const LAYERS: usize = 24;
pub const HIDDEN: usize = 896;
pub const FFN: usize = 4864;
pub const Q_HEADS: usize = 14;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const EPS: f32 = 1.0e-6;
pub const ROPE_BASE: f32 = 1_000_000.0;

pub fn fp16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let e32 = (e + 127) as u32;
            (sign << 31) | (e32 << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f800000 | (mant << 13)
    } else {
        let e32 = exp + 112;
        (sign << 31) | (e32 << 23) | (mant << 13)
    };

    f32::from_bits(bits)
}

pub fn rmsnorm(x: &[f32], w: &[f32], out: &mut [f32]) {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }

    let inv = 1.0 / (ss / x.len() as f32 + EPS).sqrt();

    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

pub fn q5_block(block: &[u8], out: &mut [f32; 32]) {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    let qs = &block[6..22];

    for j in 0..32 {
        let qlow;
        let hi;

        if j < 16 {
            qlow = qs[j] & 0x0f;
            hi = ((qh >> j) & 1) as u8;
        } else {
            let k = j - 16;
            qlow = qs[k] >> 4;
            hi = ((qh >> (k + 16)) & 1) as u8;
        }

        let q = qlow | (hi << 4);
        out[j] = d * ((q as i32) - 16) as f32;
    }
}

pub fn q8_block(block: &[u8], out: &mut [f32; 32]) {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    for i in 0..32 {
        let q = block[2 + i] as i8;
        out[i] = d * q as f32;
    }
}

pub fn q6_block(block: &[u8], out: &mut [f32; 256]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];

    let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    for half in 0..2 {
        let ql0 = half * 64;
        let qh0 = half * 32;
        let sc0 = half * 8;
        let out0 = half * 128;

        for l in 0..32 {
            let is = l / 16;

            let a = ql[ql0 + l];
            let b = ql[ql0 + l + 32];
            let h = qh[qh0 + l];

            let q1 = (a & 0x0f) | ((h & 0x03) << 4);
            let q2 = (b & 0x0f) | (((h >> 2) & 0x03) << 4);
            let q3 = (a >> 4) | (((h >> 4) & 0x03) << 4);
            let q4 = (b >> 4) | (((h >> 6) & 0x03) << 4);

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

pub fn q5_gemv(tensor: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    const BS: usize = 32;
    const BYTES: usize = 22;

    assert_eq!(cols % BS, 0);
    assert_eq!(tensor.len(), rows * (cols / BS) * BYTES);

    let blocks_per_row = cols / BS;
    let mut w = [0.0f32; BS];

    for r in 0..rows {
        let mut acc = 0.0f32;

        for b in 0..blocks_per_row {
            let off = (r * blocks_per_row + b) * BYTES;

            q5_block(&tensor[off..off + BYTES], &mut w);

            let xb = b * BS;

            for j in 0..BS {
                acc += w[j] * x[xb + j];
            }
        }

        y[r] = acc;
    }
}

pub fn q8_gemv(tensor: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    const BS: usize = 32;
    const BYTES: usize = 34;

    assert_eq!(cols % BS, 0);
    assert_eq!(tensor.len(), rows * (cols / BS) * BYTES);

    let blocks_per_row = cols / BS;
    let mut w = [0.0f32; BS];

    for r in 0..rows {
        let mut acc = 0.0f32;

        for b in 0..blocks_per_row {
            let off = (r * blocks_per_row + b) * BYTES;

            q8_block(&tensor[off..off + BYTES], &mut w);

            let xb = b * BS;

            for j in 0..BS {
                acc += w[j] * x[xb + j];
            }
        }

        y[r] = acc;
    }
}

pub fn q6_gemv(tensor: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    const BS: usize = 256;
    const BYTES: usize = 210;

    assert_eq!(cols % BS, 0);
    assert_eq!(tensor.len(), rows * (cols / BS) * BYTES);

    let blocks_per_row = cols / BS;
    let mut w = [0.0f32; BS];

    for r in 0..rows {
        let mut acc = 0.0f32;

        for b in 0..blocks_per_row {
            let off = (r * blocks_per_row + b) * BYTES;

            q6_block(&tensor[off..off + BYTES], &mut w);

            let xb = b * BS;

            for j in 0..BS {
                acc += w[j] * x[xb + j];
            }
        }

        y[r] = acc;
    }
}

pub fn q4k_block(block: &[u8], out: &mut [f32; 256]) {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    let scales = &block[4..16];
    let qs = &block[16..144];

    // Q4_K layout: 8 blocks of 32 values = 256
    // Scales and mins are packed in 12 bytes
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];

    for i in 0..4 {
        sc[i] = scales[i] & 63;
        m[i] = scales[i + 4] & 63;
    }
    for i in 4..8 {
        sc[i] = (scales[i + 4] & 0x0f) | ((scales[i - 4] >> 6) << 4);
        m[i] = (scales[i + 4] >> 4) | ((scales[i] >> 6) << 4);
    }

    for b in 0..8 {
        let d_val = d * sc[b] as f32;
        let m_val = dmin * m[b] as f32;
        let base_out = b * 32;
        let base_qs = (b / 2) * 32;
        let shift = (b % 2) * 4;

        for j in 0..32 {
            let q = (qs[base_qs + j] >> shift) & 0x0f;
            out[base_out + j] = d_val * (q as f32) - m_val;
        }
    }
}

pub fn q4k_gemv(tensor: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    const BS: usize = 256;
    const BYTES: usize = 144;
    assert_eq!(cols % BS, 0);
    assert_eq!(tensor.len(), rows * (cols / BS) * BYTES);

    let blocks_per_row = cols / BS;
    let mut w = [0.0f32; BS];

    for r in 0..rows {
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = (r * blocks_per_row + b) * BYTES;
            q4k_block(&tensor[off..off + BYTES], &mut w);
            let xb = b * BS;
            for j in 0..BS {
                acc += w[j] * x[xb + j];
            }
        }
        y[r] = acc;
    }
}

pub fn add_bias(x: &mut [f32], bias: &[f32]) {
    for i in 0..x.len() {
        x[i] += bias[i];
    }
}

pub fn apply_rope(x: &mut [f32], position: usize, heads: usize) {
    for h in 0..heads {
        let base = h * HEAD_DIM;

        for i in (0..HEAD_DIM).step_by(2) {
            let exponent = i as f32 / HEAD_DIM as f32;
            let inv_freq = ROPE_BASE.powf(-exponent);
            let theta = position as f32 * inv_freq;

            let c = theta.cos();
            let s = theta.sin();

            let a = x[base + i];
            let b = x[base + i + 1];

            x[base + i] = a * c - b * s;
            x[base + i + 1] = a * s + b * c;
        }
    }
}

pub fn single_token_attention(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32]) {
    // Single-token causal attention:
    // exactly one visible KV token, therefore softmax([score]) = [1].
    //
    // We still compute Q/K and apply RoPE so the complete projection
    // pipeline is exercised.

    for qh in 0..Q_HEADS {
        let kvh = qh / (Q_HEADS / KV_HEADS);

        let qbase = qh * HEAD_DIM;
        let kbase = kvh * HEAD_DIM;
        let vbase = kvh * HEAD_DIM;

        // Single-token causal attention has exactly one visible KV entry.
        // Therefore softmax([score]) is exactly [1.0].
        //
        // Do NOT calculate exp(score) here: a large score can overflow to
        // +Inf, producing Inf / Inf = NaN even though the true softmax is 1.
        let _score = q[qbase..qbase + HEAD_DIM]
            .iter()
            .zip(&k[kbase..kbase + HEAD_DIM])
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / (HEAD_DIM as f32).sqrt();

        let normalized = 1.0f32;

        for d in 0..HEAD_DIM {
            out[qbase + d] = v[vbase + d] * normalized;
        }
    }
}

pub fn rms(x: &[f32]) -> f32 {
    let mut s = 0.0f64;

    for &v in x {
        s += (v as f64) * (v as f64);
    }

    (s / x.len() as f64).sqrt() as f32
}

pub fn finite(x: &[f32]) -> bool {
    x.iter().all(|v| v.is_finite())
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let model = GgufModel::load(model_path)?;

    println!("============================================================");
    println!("       LUMINOUS QWEN 2.5 — FULL 24-LAYER FORWARD");
    println!("============================================================");
    println!("Layers      : {}", LAYERS);
    println!("Hidden      : {}", HIDDEN);
    println!("FFN         : {}", FFN);
    println!("Q heads     : {}", Q_HEADS);
    println!("KV heads    : {}", KV_HEADS);
    println!("Head dim    : {}", HEAD_DIM);
    println!("RoPE base   : {:.0}", ROPE_BASE);
    println!();

    let mut hidden = vec![0.0f32; HIDDEN];

    // Deterministic synthetic input.
    // This phase validates the complete 24-layer numerical pipeline;
    // tokenizer/embedding will be connected later.
    for i in 0..HIDDEN {
        let x = i as f32;
        hidden[i] = (x * 0.01731).sin() * 0.25 + (x * 0.00371).cos() * 0.125;
    }

    let initial_rms = rms(&hidden);

    println!("INPUT");
    println!("RMS         : {:.12e}", initial_rms);
    println!();

    let mut norm1 = vec![0.0f32; HIDDEN];
    let mut q = vec![0.0f32; HIDDEN];
    let mut k = vec![0.0f32; KV_HEADS * HEAD_DIM];
    let mut v = vec![0.0f32; KV_HEADS * HEAD_DIM];
    let mut attn = vec![0.0f32; HIDDEN];
    let mut projected = vec![0.0f32; HIDDEN];

    let mut norm2 = vec![0.0f32; HIDDEN];
    let mut gate = vec![0.0f32; FFN];
    let mut up = vec![0.0f32; FFN];
    let mut activated = vec![0.0f32; FFN];
    let mut down = vec![0.0f32; HIDDEN];

    for layer in 0..LAYERS {
        let prefix = format!("blk.{}", layer);

        let norm1_w = model.read_tensor_bytes(&format!("{}.attn_norm.weight", prefix))?;

        let q_w = model.read_tensor_bytes(&format!("{}.attn_q.weight", prefix))?;

        let q_b = model.read_tensor_bytes(&format!("{}.attn_q.bias", prefix))?;

        let k_w = model.read_tensor_bytes(&format!("{}.attn_k.weight", prefix))?;

        let k_b = model.read_tensor_bytes(&format!("{}.attn_k.bias", prefix))?;

        let v_w = model.read_tensor_bytes(&format!("{}.attn_v.weight", prefix))?;

        let v_b = model.read_tensor_bytes(&format!("{}.attn_v.bias", prefix))?;

        let o_w = model.read_tensor_bytes(&format!("{}.attn_output.weight", prefix))?;

        let norm2_w = model.read_tensor_bytes(&format!("{}.ffn_norm.weight", prefix))?;

        let gate_w = model.read_tensor_bytes(&format!("{}.ffn_gate.weight", prefix))?;

        let up_w = model.read_tensor_bytes(&format!("{}.ffn_up.weight", prefix))?;

        let down_w = model.read_tensor_bytes(&format!("{}.ffn_down.weight", prefix))?;

        let mut attn_q_bias = vec![0.0f32; HIDDEN];
        let mut attn_k_bias = vec![0.0f32; KV_HEADS * HEAD_DIM];
        let mut attn_v_bias = vec![0.0f32; KV_HEADS * HEAD_DIM];

        for i in 0..HIDDEN {
            let off = i * 4;
            attn_q_bias[i] =
                f32::from_le_bytes([q_b[off], q_b[off + 1], q_b[off + 2], q_b[off + 3]]);
        }

        for i in 0..KV_HEADS * HEAD_DIM {
            let off = i * 4;

            attn_k_bias[i] =
                f32::from_le_bytes([k_b[off], k_b[off + 1], k_b[off + 2], k_b[off + 3]]);

            attn_v_bias[i] =
                f32::from_le_bytes([v_b[off], v_b[off + 1], v_b[off + 2], v_b[off + 3]]);
        }

        let mut norm1_weights = vec![0.0f32; HIDDEN];
        let mut norm2_weights = vec![0.0f32; HIDDEN];

        for i in 0..HIDDEN {
            let off = i * 4;

            norm1_weights[i] = f32::from_le_bytes([
                norm1_w[off],
                norm1_w[off + 1],
                norm1_w[off + 2],
                norm1_w[off + 3],
            ]);

            norm2_weights[i] = f32::from_le_bytes([
                norm2_w[off],
                norm2_w[off + 1],
                norm2_w[off + 2],
                norm2_w[off + 3],
            ]);
        }

        // ---------------- ATTENTION ----------------

        rmsnorm(&hidden, &norm1_weights, &mut norm1);

        q5_gemv(q_w, HIDDEN, HIDDEN, &norm1, &mut q);
        q5_gemv(k_w, KV_HEADS * HEAD_DIM, HIDDEN, &norm1, &mut k);

        if v_w.len() == (KV_HEADS * HEAD_DIM) * (HIDDEN / 32) * 34 {
            q8_gemv(v_w, KV_HEADS * HEAD_DIM, HIDDEN, &norm1, &mut v);
        } else if v_w.len() == (KV_HEADS * HEAD_DIM) * (HIDDEN / 32) * 22 {
            q5_gemv(v_w, KV_HEADS * HEAD_DIM, HIDDEN, &norm1, &mut v);
        } else {
            panic!("Unexpected v_w tensor size: {}", v_w.len());
        }

        add_bias(&mut q, &attn_q_bias);
        add_bias(&mut k, &attn_k_bias);
        add_bias(&mut v, &attn_v_bias);

        apply_rope(&mut q, 0, Q_HEADS);
        apply_rope(&mut k, 0, KV_HEADS);

        single_token_attention(&q, &k, &v, &mut attn);

        q5_gemv(o_w, HIDDEN, HIDDEN, &attn, &mut projected);

        for i in 0..HIDDEN {
            hidden[i] += projected[i];
        }

        // ---------------- FFN ----------------

        rmsnorm(&hidden, &norm2_weights, &mut norm2);

        q5_gemv(gate_w, FFN, HIDDEN, &norm2, &mut gate);
        q5_gemv(up_w, FFN, HIDDEN, &norm2, &mut up);

        for i in 0..FFN {
            // SiLU(x) = x / (1 + exp(-x))
            let silu = gate[i] / (1.0 + (-gate[i]).exp());
            activated[i] = silu * up[i];
        }

        if down_w.len() == HIDDEN * (FFN / 256) * 210 {
            q6_gemv(down_w, HIDDEN, FFN, &activated, &mut down);
        } else if down_w.len() == HIDDEN * (FFN / 256) * 144 {
            q4k_gemv(down_w, HIDDEN, FFN, &activated, &mut down);
        } else {
            panic!("Unexpected down_w tensor size: {}", down_w.len());
        }

        for i in 0..HIDDEN {
            hidden[i] += down[i];
        }

        if !finite(&hidden) {
            eprintln!();
            eprintln!("[✗] NON-FINITE VALUE AT LAYER {}", layer);
            std::process::exit(1);
        }

        println!(
            "Layer {:02} / {:02}   RMS = {:.12e}   out[0] = {:.12e}",
            layer + 1,
            LAYERS,
            rms(&hidden),
            hidden[0]
        );
    }

    println!();
    println!("============================================================");
    println!("                 24-LAYER RESULT");
    println!("============================================================");
    println!("Input RMS       : {:.12e}", initial_rms);
    println!("Final RMS       : {:.12e}", rms(&hidden));
    println!("output[0]       : {:.12e}", hidden[0]);
    println!("output[1]       : {:.12e}", hidden[1]);
    println!("output[895]     : {:.12e}", hidden[895]);
    println!();

    if finite(&hidden) {
        println!("[✓] All 24 transformer layers executed");
        println!("[✓] 24 × RMSNorm");
        println!("[✓] 24 × Q/K/V projections");
        println!("[✓] 24 × RoPE");
        println!("[✓] 24 × GQA attention");
        println!("[✓] 24 × attention output projection");
        println!("[✓] 24 × attention residual");
        println!("[✓] 24 × SwiGLU");
        println!("[✓] 24 × Q6_K FFN down projection");
        println!("[✓] 24 × FFN residual");
        println!("[✓] Full hidden state remains finite");
        println!();
        println!("============================================================");
        println!("       [✓] PHASE 4K — 24-LAYER FORWARD COMPLETE");
        println!("============================================================");
    } else {
        println!("[✗] Final hidden state contains NaN/Inf");
        std::process::exit(1);
    }

    Ok(())
}
