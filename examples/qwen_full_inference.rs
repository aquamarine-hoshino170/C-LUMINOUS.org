// LUMINOUS QWEN 2.5 END-TO-END NEURAL INFERENCE ENGINE
// ARMv7 Termux Native Execution

use std::time::Instant;

#[path = "qwen_24layer.rs"]
mod qwen_backbone;

use lum_rust::gguf_loader::GgufModel;
use qwen_backbone::*;

const VOCAB: usize = 151936;
const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCKS_PER_ROW: usize = HIDDEN / Q8_0_BLOCK_SIZE; // 28 blocks

#[inline(always)]
fn fp16_to_f32_safe(h: u16) -> f32 {
    let u = h as u32;
    let sign = (u & 0x8000) << 16;
    let exp = (u & 0x7c00) >> 10;
    let mant = u & 0x03ff;

    if exp == 0x1f {
        return 0.0;
    }
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp_f = (127 - 15 + 1 - e) as u32;
        let mant_f = (m & 0x03ff) << 13;
        return f32::from_bits(sign | (exp_f << 23) | mant_f);
    }

    let exp_f = (exp + (127 - 15)) << 23;
    let mant_f = mant << 13;
    f32::from_bits(sign | exp_f | mant_f)
}

fn dot_q8_0_row(row_bytes: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..Q8_0_BLOCKS_PER_ROW {
        let offset = b * Q8_0_BLOCK_BYTES;
        let d = fp16_to_f32_safe(u16::from_le_bytes([
            row_bytes[offset],
            row_bytes[offset + 1],
        ]));
        let qs = &row_bytes[offset + 2..offset + 34];
        let x_slice = &x[b * Q8_0_BLOCK_SIZE..(b + 1) * Q8_0_BLOCK_SIZE];

        let mut block_sum = 0.0f32;
        for i in 0..32 {
            let val = (qs[i] as i8) as f32;
            block_sum += val * x_slice[i];
        }
        acc += block_sum * d;
    }
    acc
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("============================================================");
    println!(" ⚡ LUMINOUS AI ENGINE: FULL 24-LAYER + LM-HEAD PIPELINE ⚡");
    println!("============================================================");

    let start = Instant::now();
    let model = GgufModel::load(model_path)?;
    println!("[-] Model mmapped in {:.2?}", start.elapsed());

    // 1. Initial State (Deterministic synthetic input)
    let mut hidden = vec![0.0f32; HIDDEN];
    for i in 0..HIDDEN {
        let x = i as f32;
        hidden[i] = (x * 0.01731).sin() * 0.25 + (x * 0.00371).cos() * 0.125;
    }

    println!("[*] Initial Input RMS: {:.6}", rms(&hidden));

    // 2. Transformer Backbone Pass (Layer 0 to 23)
    let backbone_start = Instant::now();
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

    let mut attn_q_bias = vec![0.0f32; HIDDEN];
    let mut attn_k_bias = vec![0.0f32; KV_HEADS * HEAD_DIM];
    let mut attn_v_bias = vec![0.0f32; KV_HEADS * HEAD_DIM];

    for layer in 0..LAYERS {
        let norm1_w = model.read_tensor_bytes(&format!("blk.{}.attn_norm.weight", layer))?;
        let q_w = model.read_tensor_bytes(&format!("blk.{}.attn_q.weight", layer))?;
        let k_w = model.read_tensor_bytes(&format!("blk.{}.attn_k.weight", layer))?;
        let v_w = model.read_tensor_bytes(&format!("blk.{}.attn_v.weight", layer))?;
        let q_b = model.read_tensor_bytes(&format!("blk.{}.attn_q.bias", layer))?;
        let k_b = model.read_tensor_bytes(&format!("blk.{}.attn_k.bias", layer))?;
        let v_b = model.read_tensor_bytes(&format!("blk.{}.attn_v.bias", layer))?;
        let o_w = model.read_tensor_bytes(&format!("blk.{}.attn_output.weight", layer))?;
        let norm2_w = model.read_tensor_bytes(&format!("blk.{}.ffn_norm.weight", layer))?;
        let gate_w = model.read_tensor_bytes(&format!("blk.{}.ffn_gate.weight", layer))?;
        let up_w = model.read_tensor_bytes(&format!("blk.{}.ffn_up.weight", layer))?;
        let down_w = model.read_tensor_bytes(&format!("blk.{}.ffn_down.weight", layer))?;

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

        // Attention Sub-block
        rmsnorm(&hidden, &norm1_weights, &mut norm1);
        q5_gemv(q_w, HIDDEN, HIDDEN, &norm1, &mut q);
        q5_gemv(k_w, KV_HEADS * HEAD_DIM, HIDDEN, &norm1, &mut k);

        if v_w.len() == (KV_HEADS * HEAD_DIM) * (HIDDEN / 32) * 34 {
            q8_gemv(v_w, KV_HEADS * HEAD_DIM, HIDDEN, &norm1, &mut v);
        } else {
            q5_gemv(v_w, KV_HEADS * HEAD_DIM, HIDDEN, &norm1, &mut v);
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

        // FFN Sub-block
        rmsnorm(&hidden, &norm2_weights, &mut norm2);
        q5_gemv(gate_w, FFN, HIDDEN, &norm2, &mut gate);
        q5_gemv(up_w, FFN, HIDDEN, &norm2, &mut up);

        for i in 0..FFN {
            let silu = gate[i] / (1.0 + (-gate[i]).exp());
            activated[i] = silu * up[i];
        }

        if down_w.len() == HIDDEN * (FFN / 256) * 210 {
            q6_gemv(down_w, HIDDEN, FFN, &activated, &mut down);
        } else {
            q4k_gemv(down_w, HIDDEN, FFN, &activated, &mut down);
        }

        for i in 0..HIDDEN {
            hidden[i] += down[i];
        }
    }

    println!(
        "[-] 24 Transformer Layers completed in {:.2?}",
        backbone_start.elapsed()
    );
    println!("[*] Layer-24 Output RMS: {:.6}", rms(&hidden));

    // 3. Output RMSNorm
    let norm_bytes = model.read_tensor_bytes("output_norm.weight")?;
    let mut final_norm_w = vec![0.0f32; HIDDEN];
    for i in 0..HIDDEN {
        let off = i * 4;
        final_norm_w[i] = f32::from_le_bytes([
            norm_bytes[off],
            norm_bytes[off + 1],
            norm_bytes[off + 2],
            norm_bytes[off + 3],
        ]);
    }
    let mut normalized_hidden = vec![0.0f32; HIDDEN];
    rmsnorm(&hidden, &final_norm_w, &mut normalized_hidden);
    println!("[*] Post-OutputNorm RMS: {:.6}", rms(&normalized_hidden));

    // 4. LM-Head Projection
    println!("[*] Projecting across {} Vocab Logits...", VOCAB);
    let head_start = Instant::now();
    let out_weight_bytes = model.read_tensor_bytes("output.weight")?;
    let row_stride = Q8_0_BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES;

    let mut top_tokens: Vec<(usize, f32)> = Vec::with_capacity(VOCAB);

    for token_id in 0..VOCAB {
        let row_start = token_id * row_stride;
        let row_slice = &out_weight_bytes[row_start..row_start + row_stride];

        let mut logit = 0.0f32;
        let mut has_nan = false;

        for b in 0..Q8_0_BLOCKS_PER_ROW {
            let offset = b * Q8_0_BLOCK_BYTES;
            let d = fp16_to_f32_safe(u16::from_le_bytes([
                row_slice[offset],
                row_slice[offset + 1],
            ]));

            if d.is_nan() {
                has_nan = true;
                break;
            }

            let qs = &row_slice[offset + 2..offset + 34];
            let x_slice = &normalized_hidden[b * 32..(b + 1) * 32];

            let mut block_sum = 0.0f32;
            for i in 0..32 {
                block_sum += (qs[i] as i8 as f32) * x_slice[i];
            }
            logit += block_sum * d;
        }

        if !has_nan && logit.is_finite() {
            top_tokens.push((token_id, logit));
        }

        if token_id < 5 {
            println!("    Logit[{:>5}] = {:.6}", token_id, logit);
        }
    }

    top_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("[-] LM-Head projected in {:.2?}", head_start.elapsed());
    println!("\n============================================================");
    println!("             FINAL NEURAL PREDICTION VERDICT");
    println!("============================================================");
    println!("  Top 5 Predicted Tokens:");
    for (rank, (tok, val)) in top_tokens.iter().take(5).enumerate() {
        println!(
            "    Rank {}: Token ID = {:>6} | Logit = {:.4}",
            rank + 1,
            tok,
            val
        );
    }
    println!("  Total Inference Latency : {:.2?}", start.elapsed());
    println!("============================================================");

    Ok(())
}
