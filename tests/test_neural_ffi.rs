#![cfg(target_arch = "arm")]
use lum_rust::neural_ffi::predict_token;

#[test]
fn test_assembly_backend_from_rust() {
    let in_dim = 64;
    let out_dim = 16;

    // 1. Prepare Input X
    let mut input = vec![0i8; in_dim];
    for i in 0..in_dim {
        input[i] = ((i % 8) + 1) as i8;
    }

    // 2. Prepare Weights W
    let mut weights = vec![0i8; in_dim * out_dim];
    for i in 0..weights.len() {
        weights[i] = if i % 2 == 0 { 2 } else { -1 };
    }

    // 3. Prepare Biases B (Target Spike on Neuron 7)
    let mut biases = vec![10i32; out_dim];
    biases[7] = 60; // Target spike

    // 4. Call Pure ARMv7 Assembly Core from Rust!
    let (predicted_token, logits, probs) =
        predict_token(&input, &weights, &biases, in_dim as u32, out_dim as u32);

    println!("\n========================================================");
    println!("  LUMINOUS SVM <-> ARMv7 NEON FFI BRIDGE VERIFIED");
    println!("========================================================");
    for i in 0..out_dim {
        println!(
            " Token [{:02}] | Logit: {:3} | Confidence: {:2}%",
            i, logits[i], probs[i]
        );
    }
    println!(
        "\n[★ RUST FFI PREDICTED] Token ID: {} (Logit: {}, Prob: {}%)\n",
        predicted_token, logits[predicted_token as usize], probs[predicted_token as usize]
    );

    assert_eq!(predicted_token, 7);
    assert_eq!(logits[7], 78);
    assert_eq!(probs[7], 23);
}
