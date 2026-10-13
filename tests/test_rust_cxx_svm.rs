#![cfg(target_arch = "arm")]
use lum_rust::cxx_bridge::NeuralEngine;
use lum_rust::gguf_loader::GgufModel;
use std::fs::File;
use std::io::{Seek, Write};
use std::time::Instant;

fn create_gguf_for_cxx(
    path: &str,
    in_dim: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> std::io::Result<()> {
    let mut f = File::create(path)?;

    // GGUF Header
    f.write_all(&0x46554747u32.to_le_bytes())?;
    f.write_all(&3u32.to_le_bytes())?;
    f.write_all(&4u64.to_le_bytes())?;
    f.write_all(&0u64.to_le_bytes())?;

    // Descriptors
    let t1 = "layer1.weight";
    f.write_all(&(t1.len() as u64).to_le_bytes())?;
    f.write_all(t1.as_bytes())?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(in_dim as u64).to_le_bytes())?;
    f.write_all(&(hidden_dim as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&0u64.to_le_bytes())?;

    let t2 = "layer1.bias";
    let w1_sz = (in_dim * hidden_dim) as u64;
    f.write_all(&(t2.len() as u64).to_le_bytes())?;
    f.write_all(t2.as_bytes())?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(hidden_dim as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&w1_sz.to_le_bytes())?;

    let t3 = "layer2.weight";
    let b1_sz = hidden_dim as u64 * 4;
    let off_w2 = w1_sz + b1_sz;
    f.write_all(&(t3.len() as u64).to_le_bytes())?;
    f.write_all(t3.as_bytes())?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(hidden_dim as u64).to_le_bytes())?;
    f.write_all(&(vocab_size as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&off_w2.to_le_bytes())?;

    let t4 = "layer2.bias";
    let w2_sz = (hidden_dim * vocab_size) as u64;
    let off_b2 = off_w2 + w2_sz;
    f.write_all(&(t4.len() as u64).to_le_bytes())?;
    f.write_all(t4.as_bytes())?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(vocab_size as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&off_b2.to_le_bytes())?;

    let pos = f.stream_position()?;
    let aligned = ((pos + 31) / 32) * 32;
    let pad = (aligned - pos) as usize;
    if pad > 0 {
        f.write_all(&vec![0u8; pad])?;
    }

    // W1 payload
    let mut w1 = vec![0u8; in_dim * hidden_dim];
    for h in 0..hidden_dim {
        w1[h * in_dim + (h % in_dim)] = 1;
    }
    f.write_all(&w1)?;

    // B1 payload
    let b1 = vec![0i32; hidden_dim];
    for b in &b1 {
        f.write_all(&b.to_le_bytes())?;
    }

    // W2 transition matrix
    let mut w2 = vec![0u8; hidden_dim * vocab_size];
    for token in 0..8 {
        let next_token = (token + 1) % 8;
        w2[next_token * hidden_dim + token] = 12;
    }
    f.write_all(&w2)?;

    // B2 payload
    let b2 = vec![0i32; vocab_size];
    for b in &b2 {
        f.write_all(&b.to_le_bytes())?;
    }

    Ok(())
}

#[test]
fn test_rust_to_cxx_shared_library() {
    let gguf_file = "cxx_svm_model.gguf";
    let in_dim = 64;
    let hidden_dim = 128;
    let vocab_size = 16;

    create_gguf_for_cxx(gguf_file, in_dim, hidden_dim, vocab_size).unwrap();

    let model = GgufModel::load(gguf_file).unwrap();
    let w1 = model.read_tensor_bytes("layer1.weight").unwrap();
    let w2 = model.read_tensor_bytes("layer2.weight").unwrap();
    let b1 = vec![0i32; hidden_dim];
    let b2 = vec![0i32; vocab_size];

    let vocab = [
        "Luminous", "is", "a", "pure", "assembly", "AI", "engine", ".",
    ];

    println!("\n========================================================");
    println!(" 🔗 LUMINOUS RUST SVM <-> C++20 (.so) <-> ARMv7 NEON");
    println!("========================================================");

    // Initialize C++ engine instance via FFI
    let engine = NeuralEngine::new(in_dim as u32, hidden_dim as u32, vocab_size as u32);

    let mut current_token = 0usize;
    let mut current_state = vec![0i8; in_dim];
    current_state[current_token] = 10;

    let mut generated_words = vec![vocab[current_token]];
    let start_all = Instant::now();

    for step in 1..8 {
        let step_start = Instant::now();

        // Call liblum_cxx.so step method directly from Rust
        let (next_token, conf) = engine.step(&current_state, &w1, &b1, &w2, &b2);
        let word = vocab[next_token as usize];
        generated_words.push(word);

        current_token = next_token as usize;
        current_state.fill(0);
        current_state[current_token] = 10;

        let step_elapsed = step_start.elapsed();
        println!(
            "[SVM Step {}] Token: {:02} | Conf: {:2}% | Word: {:<10} | Latency: {:.2?}",
            step, next_token, conf, word, step_elapsed
        );
    }

    let total_elapsed = start_all.elapsed();
    println!("--------------------------------------------------------");
    println!("📝 Generated: {}", generated_words.join(" "));
    println!("⚡ Total Pipeline Execution: {:.2?}", total_elapsed);
    println!("========================================================\n");

    assert_eq!(
        generated_words.join(" "),
        "Luminous is a pure assembly AI engine ."
    );
    let _ = std::fs::remove_file(gguf_file);
}
