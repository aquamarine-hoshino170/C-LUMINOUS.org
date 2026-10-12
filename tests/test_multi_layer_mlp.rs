use lum_rust::gguf_loader::GgufModel;
use std::fs::File;
use std::io::{Seek, Write};
use std::time::Instant;

fn create_multi_layer_gguf(
    file_path: &str,
    in_dim: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> std::io::Result<()> {
    let mut f = File::create(file_path)?;

    // 1. Header
    f.write_all(&0x46554747u32.to_le_bytes())?;
    f.write_all(&3u32.to_le_bytes())?;
    f.write_all(&4u64.to_le_bytes())?;
    f.write_all(&0u64.to_le_bytes())?;

    // Descriptors
    let t1_name = "layer1.weight";
    f.write_all(&(t1_name.len() as u64).to_le_bytes())?;
    f.write_all(t1_name.as_bytes())?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(in_dim as u64).to_le_bytes())?;
    f.write_all(&(hidden_dim as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&0u64.to_le_bytes())?;

    let t2_name = "layer1.bias";
    let w1_size = (in_dim * hidden_dim) as u64;
    f.write_all(&(t2_name.len() as u64).to_le_bytes())?;
    f.write_all(t2_name.as_bytes())?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(hidden_dim as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&w1_size.to_le_bytes())?;

    let t3_name = "layer2.weight";
    let b1_size = hidden_dim as u64 * 4;
    let offset_w2 = w1_size + b1_size;
    f.write_all(&(t3_name.len() as u64).to_le_bytes())?;
    f.write_all(t3_name.as_bytes())?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(hidden_dim as u64).to_le_bytes())?;
    f.write_all(&(vocab_size as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&offset_w2.to_le_bytes())?;

    let t4_name = "layer2.bias";
    let w2_size = (hidden_dim * vocab_size) as u64;
    let offset_b2 = offset_w2 + w2_size;
    f.write_all(&(t4_name.len() as u64).to_le_bytes())?;
    f.write_all(t4_name.as_bytes())?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(vocab_size as u64).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&offset_b2.to_le_bytes())?;

    // Alignment
    let cur_pos = f.stream_position()?;
    let aligned = ((cur_pos + 31) / 32) * 32;
    let pad_len = (aligned - cur_pos) as usize;
    if pad_len > 0 {
        f.write_all(&vec![0u8; pad_len])?;
    }

    // W1: Direct diagonal identity forward pass
    let mut w1 = vec![0u8; in_dim * hidden_dim];
    for h in 0..hidden_dim {
        w1[h * in_dim + (h % in_dim)] = 1;
    }
    f.write_all(&w1)?;

    let b1 = vec![0i32; hidden_dim];
    for b in &b1 {
        f.write_all(&b.to_le_bytes())?;
    }

    // W2: Explicit transition matrix mapping Token(i) -> Token(i+1)
    // Sentence: Luminous(0) -> is(1) -> a(2) -> pure(3) -> assembly(4) -> AI(5) -> engine(6) -> .(7)
    let mut w2 = vec![0u8; hidden_dim * vocab_size];
    for token in 0..8 {
        let next_token = (token + 1) % 8;
        // The neuron representing next_token responds strongly to current token's hidden state
        w2[next_token * hidden_dim + token] = 12;
    }
    f.write_all(&w2)?;

    let b2 = vec![0i32; vocab_size];
    for b in &b2 {
        f.write_all(&b.to_le_bytes())?;
    }

    Ok(())
}

#[link(name = "lum_nn", kind = "static")]
extern "C" {
    fn lum_nn_dense_layer(
        input: *const i8,
        weights: *const i8,
        biases: *const i32,
        output: *mut i8,
        in_dim: u32,
        out_dim: u32,
        scale_shift: i32,
    );
    fn lum_nn_softmax(logits: *const i8, probs: *mut u32, len: u32);
    fn lum_nn_argmax(logits: *const i8, len: u32) -> u32;
}

#[test]
fn test_mlp_autoregressive() {
    let gguf_path = "mlp_model.gguf";
    let in_dim = 64;
    let hidden_dim = 128;
    let vocab_size = 16;

    create_multi_layer_gguf(gguf_path, in_dim, hidden_dim, vocab_size).unwrap();

    let model = GgufModel::load(gguf_path).unwrap();
    let w1 = model.read_tensor_bytes("layer1.weight").unwrap();
    let b1 = model.read_tensor_bytes("layer1.bias").unwrap();
    let w2 = model.read_tensor_bytes("layer2.weight").unwrap();
    let b2 = model.read_tensor_bytes("layer2.bias").unwrap();

    let vocab = [
        "Luminous",
        "is",
        "a",
        "pure",
        "assembly",
        "AI",
        "engine",
        ".",
        "It",
        "generates",
        "tokens",
        "with",
        "zero",
        "Python",
        "overhead",
        "!",
    ];

    println!("\n========================================================");
    println!(" 🚀 LUMINOUS MULTI-LAYER GGUF MLP INFERENCE");
    println!("========================================================");

    let mut current_token = 0usize; // Prompt: "Luminous"
    let mut current_state = vec![0i8; in_dim];
    current_state[current_token] = 10;

    let mut generated_text = vec![vocab[current_token]];
    let mut probs = vec![0u32; vocab_size];

    let total_start = Instant::now();

    for step in 1..8 {
        let step_start = Instant::now();

        // Layer 1 (64 -> 128)
        let mut hidden_out = vec![0i8; hidden_dim];
        unsafe {
            lum_nn_dense_layer(
                current_state.as_ptr(),
                w1.as_ptr(),
                b1.as_ptr() as *const i32,
                hidden_out.as_mut_ptr(),
                in_dim as u32,
                hidden_dim as u32,
                0,
            );
        }

        // Layer 2 (128 -> 16)
        let mut logits = vec![0i8; vocab_size];
        unsafe {
            lum_nn_dense_layer(
                hidden_out.as_ptr(),
                w2.as_ptr(),
                b2.as_ptr() as *const i32,
                logits.as_mut_ptr(),
                hidden_dim as u32,
                vocab_size as u32,
                0,
            );
            lum_nn_softmax(logits.as_ptr(), probs.as_mut_ptr(), vocab_size as u32);
        }

        let token_id = unsafe { lum_nn_argmax(logits.as_ptr(), vocab_size as u32) };
        let word = vocab[token_id as usize];
        generated_text.push(word);

        // Feedback: Reset state and set active embedding for predicted token
        current_token = token_id as usize;
        current_state.fill(0);
        current_state[current_token] = 10;

        let step_time = step_start.elapsed();
        println!(
            "[Step {}] Gen: {:<10} (Token: {:02}, Conf: {:2}%) | Time: {:.2?}",
            step, word, token_id, probs[token_id as usize], step_time
        );
    }

    println!("--------------------------------------------------------");
    println!("📝 Final Paragraph: {}", generated_text.join(" "));
    println!("⏱️️ Total Inference Time: {:.2?}", total_start.elapsed());
    println!("========================================================\n");

    let _ = std::fs::remove_file(gguf_path);
}
