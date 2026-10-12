use std::os::raw::{c_int, c_uint};
use std::time::Instant;

#[link(name = "lum_nn", kind = "static")]
extern "C" {
    fn lum_nn_q_k_dot(q: *const i8, k: *const i8, scores: *mut i8);
    fn lum_nn_dense_layer(
        i: *const i8,
        w: *const i8,
        b: *const i32,
        o: *mut i8,
        id: c_uint,
        od: c_uint,
        s: c_int,
    );
    fn lum_nn_softmax(logits: *const i8, probs: *mut u32, len: c_uint);
    fn lum_nn_argmax(logits: *const i8, len: c_uint) -> c_uint;
}

// Luminous VM Opcodes
#[derive(Debug)]
pub enum Opcode {
    SelfAttention,
    PredictToken,
}

pub struct LuminousVM {
    pub hidden_state: Vec<i8>,
    pub token_vocab: Vec<String>,
}

impl LuminousVM {
    pub fn new() -> Self {
        let vocab = vec![
            "I",
            "am",
            "Luminous",
            "a",
            "high",
            "performance",
            "AI",
            "engine",
            "running",
            "on",
            "bare",
            "metal",
            "ARMv7",
            "architecture",
            ".",
            "!",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            hidden_state: vec![1i8; 64], // Initial embedding state
            token_vocab: vocab,
        }
    }

    pub fn execute(&mut self, op: Opcode, weights: &[i8], biases: &[i32]) -> u32 {
        match op {
            Opcode::SelfAttention => {
                let mut attention_scores = vec![0i8; 16];
                let mut attention_probs = vec![0u32; 16];
                unsafe {
                    // Q x K^T (Query vs Key Matrix)
                    lum_nn_q_k_dot(
                        self.hidden_state.as_ptr(),
                        weights.as_ptr(),
                        attention_scores.as_mut_ptr(),
                    );
                    lum_nn_softmax(attention_scores.as_ptr(), attention_probs.as_mut_ptr(), 16);
                }
                // Simulate V-matrix integration by mutating hidden state slightly
                for i in 0..64 {
                    self.hidden_state[i] = self.hidden_state[i].saturating_add(2);
                }
                0
            }
            Opcode::PredictToken => {
                let mut logits = vec![0i8; 16];
                unsafe {
                    lum_nn_dense_layer(
                        self.hidden_state.as_ptr(),
                        weights.as_ptr(),
                        biases.as_ptr(),
                        logits.as_mut_ptr(),
                        64,
                        16,
                        1,
                    );
                    lum_nn_argmax(logits.as_ptr(), 16)
                }
            }
        }
    }
}

#[test]
fn test_autoregressive_generation() {
    println!("\n========================================================");
    println!(" ⚡ LUMINOUS AUTOREGRESSIVE TRANSFORMER LOOP (NEON) ⚡");
    println!("========================================================");

    let mut vm = LuminousVM::new();

    // Mock Weights for Attention (K-matrix) & Feed-Forward
    let attn_weights = vec![2i8; 16 * 64];
    let mut ff_weights = vec![1i8; 16 * 64];
    let mut biases = vec![5i32; 16];

    let mut generated_tokens = Vec::new();
    let start_time = Instant::now();

    // Autoregressive Loop: Generate 8 tokens
    for step in 0..8 {
        // Dynamic Bias Shift to "guide" the model to different tokens based on step
        for b in biases.iter_mut() {
            *b = 2;
        }
        biases[step * 2 % 16] = 80; // Force a spike for demonstration

        // 1. Run Attention Block
        vm.execute(Opcode::SelfAttention, &attn_weights, &[]);

        // 2. Run Token Prediction
        let token_id = vm.execute(Opcode::PredictToken, &ff_weights, &biases);

        let word = vm.token_vocab[token_id as usize].clone();
        generated_tokens.push(word.clone());

        println!(
            "[Step {}] Opcode::PredictToken executed -> Token ID: {:02} | Word: '{}'",
            step + 1,
            token_id,
            word
        );

        // 3. Autoregressive Feedback (Update hidden state based on generated token)
        for i in 0..64 {
            vm.hidden_state[i] = (token_id as i8)
                .saturating_mul(2)
                .saturating_add(i as i8 % 4);
        }
    }

    let duration = start_time.elapsed();
    println!("--------------------------------------------------------");
    println!("📝 Generated Text: {}", generated_tokens.join(" "));
    println!(
        "⏱️ Total Inference Time: {:.2?} (Bare-Metal ARMv7 Pipeline)",
        duration
    );
    println!("========================================================\n");
}
