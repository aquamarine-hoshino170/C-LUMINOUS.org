#![cfg(target_arch = "arm")]
use lum_rust::gguf_loader::GgufModel;
use lum_rust::neural_ffi::predict_token;
use std::fs::File;
use std::io::{Seek, Write};
use std::time::Instant;

fn create_mock_gguf_weights(file_path: &str, in_dim: usize, out_dim: usize) -> std::io::Result<()> {
    let mut f = File::create(file_path)?;

    // 1. Header: Magic, Version (3), Tensor Count (2: weights & biases), Metadata Count (0)
    f.write_all(&0x46554747u32.to_le_bytes())?; // "GGUF"
    f.write_all(&3u32.to_le_bytes())?; // Version 3
    f.write_all(&2u64.to_le_bytes())?; // 2 Tensors
    f.write_all(&0u64.to_le_bytes())?; // 0 Metadata KV

    // 2. Tensor 1: "token_embeddings.weight" (out_dim x in_dim)
    let t1_name = "output.weight";
    f.write_all(&(t1_name.len() as u64).to_le_bytes())?;
    f.write_all(t1_name.as_bytes())?;
    f.write_all(&2u32.to_le_bytes())?; // 2 dims
    f.write_all(&(in_dim as u64).to_le_bytes())?;
    f.write_all(&(out_dim as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?; // Type 0 (GGML_TYPE_F32 / Raw bytes)
    f.write_all(&0u64.to_le_bytes())?; // Offset 0

    // Tensor 2: "output.bias"
    let t2_name = "output.bias";
    let weight_size = (in_dim * out_dim) as u64;
    f.write_all(&(t2_name.len() as u64).to_le_bytes())?;
    f.write_all(t2_name.as_bytes())?;
    f.write_all(&1u32.to_le_bytes())?; // 1 dim
    f.write_all(&(out_dim as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&weight_size.to_le_bytes())?; // Offset right after weights

    // Pad to 32 bytes boundary
    let cur_pos = f.stream_position()?;
    let aligned = ((cur_pos + 31) / 32) * 32;
    let pad_len = (aligned - cur_pos) as usize;
    if pad_len > 0 {
        f.write_all(&vec![0u8; pad_len])?;
    }

    // 3. Binary Payloads: Pre-trained Weights and Biases
    // Weight Matrix: Alternating +2 and -1 (INT8 Quantized)
    let mut weights = Vec::with_capacity(in_dim * out_dim);
    for i in 0..(in_dim * out_dim) {
        weights.push(if i % 2 == 0 { 2u8 } else { 0xFFu8 }); // 0xFF is -1 in two's complement
    }
    f.write_all(&weights)?;

    // Biases: Raw bytes for 16 integers, neuron 7 spiked
    let mut biases = Vec::new();
    for i in 0..out_dim {
        let val: i32 = if i == 7 { 60 } else { 10 };
        biases.extend_from_slice(&val.to_le_bytes());
    }
    // We only need the first 64 bytes of bias as INT8 representations for mock or load directly
    f.write_all(&biases)?;

    Ok(())
}

#[test]
fn test_gguf_to_neon_token_completion() {
    let gguf_path = "mini_smollm_q8.gguf";
    let in_dim = 64;
    let out_dim = 16;

    // Generate valid GGUF Binary on disk
    create_mock_gguf_weights(gguf_path, in_dim, out_dim).expect("Failed to build GGUF binary");

    println!("\n========================================================");
    println!("   LUMINOUS BARE-METAL GGUF BINARY INFERENCE ENGINE");
    println!("========================================================");

    // 1. Zero-Copy Header Parse
    let t0 = Instant::now();
    let model = GgufModel::load(gguf_path).expect("Failed to parse GGUF model");
    let parse_time = t0.elapsed();

    println!(" [GGUF Loader] Successfully parsed: {}", gguf_path);
    println!(" [GGUF Loader] Tensors found: {}", model.tensors.len());
    for (name, info) in &model.tensors {
        println!(
            "   -> Tensor '{}' | Shape: {:?} | Data Offset: 0x{:X}",
            name, info.shape, info.offset
        );
    }
    println!(" [GGUF Loader] Header Parse Time: {:.2?}", parse_time);

    // 2. Extract Raw Quantized Tensor directly into NEON Alignment
    let t_load = Instant::now();
    let weight_tensor = model
        .read_tensor_bytes("output.weight")
        .expect("Tensor load failed");
    let load_time = t_load.elapsed();
    println!(
        " [GGUF Payload] Extracted 'output.weight' ({} bytes) in {:.2?}",
        weight_tensor.len(),
        load_time
    );

    // 3. Vocabulary Table
    let vocab = [
        "<s>",
        "Hello",
        "world",
        "Luminous",
        "is",
        "a",
        "superfast",
        "AI",
        "system",
        "built",
        "on",
        "pure",
        "ARMv7",
        "assembly",
        "!",
        "</s>",
    ];

    // 4. Feed Prompt Embedding through ARMv7 Assembly NEON Inference
    let mut prompt_input = vec![0i8; in_dim];
    for i in 0..in_dim {
        prompt_input[i] = ((i % 8) + 1) as i8;
    }

    let mut biases = vec![10i32; out_dim];
    biases[7] = 60; // AI Token 7 spike

    let t_infer = Instant::now();
    let (predicted_token_id, logits, probs) = predict_token(
        &prompt_input,
        &weight_tensor,
        &biases,
        in_dim as u32,
        out_dim as u32,
    );
    let infer_time = t_infer.elapsed();

    println!(" -------------------------------------------------------");
    println!(" [ARMv7 NEON Core] Execution Time : {:.2?}", infer_time);
    println!(
        " [ARMv7 NEON Core] Top Logit       : {}",
        logits[predicted_token_id as usize]
    );
    println!(
        " [ARMv7 NEON Core] Confidence      : {}%",
        probs[predicted_token_id as usize]
    );
    println!(
        " [★ TEXT COMPLETION RESULT]        : '{}' (Token ID: {})",
        vocab[predicted_token_id as usize], predicted_token_id
    );
    println!(" ========================================================\n");

    assert_eq!(predicted_token_id, 7);
    assert_eq!(vocab[predicted_token_id as usize], "AI");

    // Cleanup generated file
    let _ = std::fs::remove_file(gguf_path);
}
