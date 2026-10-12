use lum_rust::gguf_loader::GgufModel;
use std::time::Instant;

const HIDDEN: usize = 896;
const VOCAB: usize = 151936;
const EMB_BLOCK_BYTES: usize = 22;
const EMB_BLOCKS_PER_TOKEN: usize = HIDDEN / 32;

const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCKS_PER_ROW: usize = HIDDEN / Q8_0_BLOCK_SIZE;

#[inline(always)]
fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;

    let value = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            (mant as f32) * 2.0f32.powi(-24)
        }
    } else if exp == 31 {
        0.0 // Clamp inf/nan scales safely
    } else {
        (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    };

    if sign != 0 {
        -value
    } else {
        value
    }
}

fn dequantize_q5_0_block(block: &[u8]) -> (f32, [f32; 32]) {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let d = fp16_to_f32(d_bits);

    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let mut values = [0.0f32; 32];

    for i in 0..16 {
        let low0 = block[6 + i] & 0x0f;
        let low1 = block[6 + i] >> 4;

        let high0 = ((qh >> i) & 1) as u8;
        let high1 = ((qh >> (i + 16)) & 1) as u8;

        let q0 = (low0 | (high0 << 4)) as i32 - 16;
        let q1 = (low1 | (high1 << 4)) as i32 - 16;

        values[i] = (q0 as f32) * d;
        values[i + 16] = (q1 as f32) * d;
    }

    (d, values)
}

fn decode_token_embedding(emb_bytes: &[u8], token_id: usize) -> Vec<f32> {
    let mut hidden = vec![0.0f32; HIDDEN];
    let token_offset = token_id * (EMB_BLOCKS_PER_TOKEN * EMB_BLOCK_BYTES);

    for b in 0..EMB_BLOCKS_PER_TOKEN {
        let block_start = token_offset + b * EMB_BLOCK_BYTES;
        let block = &emb_bytes[block_start..block_start + EMB_BLOCK_BYTES];
        let (_, decoded) = dequantize_q5_0_block(block);
        hidden[b * 32..(b + 1) * 32].copy_from_slice(&decoded);
    }
    hidden
}

fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let inv = 1.0f32 / (sum / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v * inv * w)
        .collect()
}

#[inline(always)]
fn dot_q8_0_row(row_bytes: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..Q8_0_BLOCKS_PER_ROW {
        let offset = b * Q8_0_BLOCK_BYTES;
        let d = fp16_to_f32(u16::from_le_bytes([
            row_bytes[offset],
            row_bytes[offset + 1],
        ]));
        let qs = &row_bytes[offset + 2..offset + 34];
        let x_slice = &x[b * Q8_0_BLOCK_SIZE..(b + 1) * Q8_0_BLOCK_SIZE];

        let mut block_sum = 0.0f32;
        for i in 0..32 {
            block_sum += (qs[i] as i8 as f32) * x_slice[i];
        }
        acc += block_sum * d;
    }
    acc
}

fn main() {
    let path = "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    println!("============================================================");
    println!(" ⚡ LUMINOUS AI ENGINE: QWEN 2.5 LM-HEAD INFERENCE ⚡");
    println!("============================================================");

    let start = Instant::now();
    let model = GgufModel::load(path).expect("GGUF load failed");
    println!("[-] Model mapped in {:.2?}", start.elapsed());

    // Prompt: Token ID 9707 (" Hello")
    let prompt_token_id = 9707;
    println!(
        "\n[1] Decoding Embedding for Token ID: {} (\" Hello\")",
        prompt_token_id
    );
    let emb_bytes = model
        .read_tensor_bytes("token_embd.weight")
        .expect("token_embd missing");
    let hidden = decode_token_embedding(&emb_bytes, prompt_token_id);

    let sum_sq: f32 = hidden.iter().map(|&v| v * v).sum();
    let hidden_rms = (sum_sq / HIDDEN as f32).sqrt();
    println!("    Hidden state initialized (RMS: {:.4e})", hidden_rms);

    // Final Output RMSNorm
    let norm_bytes = model
        .read_tensor_bytes("output_norm.weight")
        .expect("output_norm missing");
    let norm_weights: Vec<f32> = norm_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let norm_hidden = rmsnorm(&hidden, &norm_weights, 1e-6);
    let norm_rms = (norm_hidden.iter().map(|&v| v * v).sum::<f32>() / HIDDEN as f32).sqrt();
    println!("    Post-norm RMS: {:.4e}", norm_rms);

    // LM Head Projection
    println!(
        "\n[2] Projecting across {} Vocab Logits (Q8_0 LM-Head)...",
        VOCAB
    );
    let out_weight_bytes = model
        .read_tensor_bytes("output.weight")
        .expect("output.weight missing");
    let row_stride = Q8_0_BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES;

    let head_start = Instant::now();
    let mut max_logit = f32::NEG_INFINITY;
    let mut best_token = 0;

    // Collect top predictions
    let mut logits_vec: Vec<(usize, f32)> = Vec::with_capacity(VOCAB);

    for token_id in 0..VOCAB {
        let row_start = token_id * row_stride;
        let row_slice = &out_weight_bytes[row_start..row_start + row_stride];
        let logit = dot_q8_0_row(row_slice, &norm_hidden);

        if logit > max_logit {
            max_logit = logit;
            best_token = token_id;
        }
        if token_id < 1000 {
            logits_vec.push((token_id, logit));
        }
    }

    println!(
        "[-] LM-Head projection finished in {:.2?}",
        head_start.elapsed()
    );
    println!("\n============================================================");
    println!("                 INFERENCE PREDICTION");
    println!("============================================================");
    println!("  Input Prompt Token ID   : {}", prompt_token_id);
    println!("  Predicted Next Token ID : {}", best_token);
    println!("  Maximum Logit Value     : {:.4}", max_logit);
    println!("============================================================");
}
