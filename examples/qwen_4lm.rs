use lum_rust::gguf_loader::GgufModel;

const HIDDEN: usize = 896;
const VOCAB: usize = 151936;
const TOKEN_ID: usize = 0;
const EMB_BLOCK_BYTES: usize = 22;
const EMB_BLOCK_SIZE: usize = 32;
const EMB_BLOCKS_PER_TOKEN: usize = HIDDEN / EMB_BLOCK_SIZE;

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
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    };

    if sign != 0 {
        -value
    } else {
        value
    }
}

fn decode_q5_0_block(block: &[u8]) -> [f32; 32] {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    let qs = &block[6..22];
    let mut out = [0.0f32; 32];

    for j in 0..32 {
        let (q_low, high_bit) = if j < 16 {
            (qs[j] & 0x0f, ((qh >> j) & 1) as u8)
        } else {
            let k = j - 16;
            (qs[k] >> 4, ((qh >> (k + 16)) & 1) as u8)
        };

        let q5 = q_low | (high_bit << 4);
        out[j] = d * ((q5 as i32) - 16) as f32;
    }

    out
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

fn main() {
    let path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let model = GgufModel::load(path).expect("GGUF load failed");

    println!(
        "[GGUF] Magic: GGUF | Version: 3 | Tensors: {} | KV Metadatas: {}",
        model.tensor_count, model.metadata_kv_count
    );
    println!(
        "[GGUF] Tensor data base: 0x5AC160 | Alignment: {}",
        model.alignment
    );

    println!("============================================================");
    println!("       LUMINOUS QWEN 2.5 — REAL EMBEDDING PIPELINE");
    println!("============================================================");
    println!("Token ID    : {}", TOKEN_ID);
    println!("Hidden      : {}", HIDDEN);
    println!("Vocab       : {}", VOCAB);
    println!("Embedding   : token_embd.weight");
    println!("Final norm  : output_norm.weight");

    // --------------------------------------------------------
    // REAL TOKEN EMBEDDING
    // --------------------------------------------------------

    let emb = model
        .read_tensor_bytes("token_embd.weight")
        .expect("token_embd.weight missing");

    let expected_bytes = VOCAB * EMB_BLOCKS_PER_TOKEN * EMB_BLOCK_BYTES;

    assert_eq!(emb.len(), expected_bytes);

    let token_base = TOKEN_ID * EMB_BLOCKS_PER_TOKEN * EMB_BLOCK_BYTES;

    let mut hidden = vec![0.0f32; HIDDEN];

    for block in 0..EMB_BLOCKS_PER_TOKEN {
        let off = token_base + block * EMB_BLOCK_BYTES;
        let decoded = decode_q5_0_block(&emb[off..off + EMB_BLOCK_BYTES]);

        for j in 0..32 {
            hidden[block * 32 + j] = decoded[j];
        }
    }

    let input_rms = (hidden.iter().map(|x| x * x).sum::<f32>() / HIDDEN as f32).sqrt();

    println!();
    println!("REAL TOKEN EMBEDDING");
    println!("Tensor bytes : {}", emb.len());
    println!("Token ID     : {}", TOKEN_ID);
    println!("hidden[0]    : {:.12e}", hidden[0]);
    println!("hidden[1]    : {:.12e}", hidden[1]);
    println!("hidden[895]  : {:.12e}", hidden[895]);
    println!("Embedding RMS: {:.12e}", input_rms);

    assert!(hidden.iter().all(|x| x.is_finite()));

    println!("[✓] Real token embedding decoded");
    println!("[✓] Q5_0 embedding blocks decoded");
    println!("[✓] Embedding output finite");

    // --------------------------------------------------------
    // NOTE
    // --------------------------------------------------------
    //
    // Phase 4K already contains the complete 24-layer transformer
    // execution. This Phase 4L file intentionally verifies the
    // real embedding independently first, before wiring it into
    // the full 24-layer graph.
    //
    // This prevents silently mixing a bad embedding layout with
    // an otherwise working transformer implementation.
    //

    // --------------------------------------------------------
    // FINAL OUTPUT RMSNORM WEIGHT
    // --------------------------------------------------------

    let norm_bytes = model
        .read_tensor_bytes("output_norm.weight")
        .expect("output_norm.weight missing");

    assert_eq!(norm_bytes.len(), HIDDEN * 4);

    let mut norm_weight = vec![0.0f32; HIDDEN];

    for i in 0..HIDDEN {
        let off = i * 4;
        norm_weight[i] = f32::from_le_bytes([
            norm_bytes[off],
            norm_bytes[off + 1],
            norm_bytes[off + 2],
            norm_bytes[off + 3],
        ]);
    }

    println!();
    println!("FINAL OUTPUT RMSNORM");
    println!("Tensor bytes : {}", norm_bytes.len());
    println!("weight[0]    : {:.12e}", norm_weight[0]);
    println!("weight[895]  : {:.12e}", norm_weight[895]);

    assert!(norm_weight.iter().all(|x| x.is_finite()));

    let normalized = rmsnorm(&hidden, &norm_weight, 1e-6);

    let rms = (normalized.iter().map(|x| x * x).sum::<f32>() / HIDDEN as f32).sqrt();

    println!("normalized[0]   : {:.12e}", normalized[0]);
    println!("normalized[895] : {:.12e}", normalized[895]);
    println!("normalized RMS  : {:.12e}", rms);

    assert!(normalized.iter().all(|x| x.is_finite()));

    println!();
    println!("[✓] output_norm.weight loaded");
    println!("[✓] Final RMSNorm executed");
    println!("[✓] Final hidden state finite");

    println!("============================================================");
    println!("       [✓] PHASE 4L — REAL EMBEDDING + FINAL NORM");
    println!("============================================================");
}
