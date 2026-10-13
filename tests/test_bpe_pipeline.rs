#![cfg(target_arch = "arm")]
use lum_rust::cxx_bridge::NeuralEngine;
use lum_rust::tokenizer::ByteLevelBpeTokenizer;
use std::time::Instant;

#[test]
fn test_dynamic_bpe_to_assembly_generation() {
    println!("\n========================================================");
    println!(" 🔤 LUMINOUS BYTE-LEVEL BPE TOKENIZER + SILICON INFERENCE");
    println!("========================================================");

    let mut tokenizer = ByteLevelBpeTokenizer::new();

    // 1. Hierarchical byte-level merges
    tokenizer.add_merge(b"u", b"m"); // "um"
    tokenizer.add_merge(b"L", b"um"); // "Lum"
    tokenizer.add_merge(b"i", b"n"); // "in"
    tokenizer.add_merge(b"in", b"o"); // "ino"
    tokenizer.add_merge(b"ino", b"u"); // "inou"
    tokenizer.add_merge(b"inou", b"s"); // "inous"
    tokenizer.add_merge(b"Lum", b"inous"); // Full token: "Luminous"

    tokenizer.add_merge(b" ", b"i"); // " i"
    tokenizer.add_merge(b" i", b"s"); // Full token: " is"

    tokenizer.add_merge(b"A", b"I"); // "AI"
    tokenizer.add_merge(b" ", b"AI"); // Full token: " AI"

    tokenizer.add_merge(b"r", b"e"); // "re"
    tokenizer.add_merge(b"re", b"a"); // "rea"
    tokenizer.add_merge(b"rea", b"d"); // "read"
    tokenizer.add_merge(b"read", b"y"); // "ready"
    tokenizer.add_merge(b"ready", b"."); // "ready."
    tokenizer.add_merge(b" ", b"ready."); // Full token: " ready."

    let prompt = "Luminous";
    let encoded_prompt = tokenizer.encode(prompt);
    println!(" [Tokenizer] Raw Prompt  : \"{}\"", prompt);
    println!(" [Tokenizer] Encoded IDs : {:?}", encoded_prompt);

    // 2. Hardware Dimensions
    let in_dim = 64usize;
    let hidden_dim = 128usize;
    let vocab_size = 16usize;

    // Weights setup: 0 -> 1 -> 2 -> 3 -> 0
    let mut w1 = vec![0i8; in_dim * hidden_dim];
    for h in 0..hidden_dim {
        w1[h * in_dim + (h % in_dim)] = 1;
    }
    let b1 = vec![0i32; hidden_dim];

    let mut w2 = vec![0i8; hidden_dim * vocab_size];
    for tok in 0..4 {
        let next_tok = (tok + 1) % 4;
        w2[next_tok * hidden_dim + tok] = 14;
    }
    let b2 = vec![0i32; vocab_size];

    let sub_vocab_bpe = vec![
        *tokenizer
            .vocab
            .get(b"Luminous".as_slice())
            .expect("Token 'Luminous' missing"),
        *tokenizer
            .vocab
            .get(b" is".as_slice())
            .expect("Token ' is' missing"),
        *tokenizer
            .vocab
            .get(b" AI".as_slice())
            .expect("Token ' AI' missing"),
        *tokenizer
            .vocab
            .get(b" ready.".as_slice())
            .expect("Token ' ready.' missing"),
    ];

    let engine = NeuralEngine::new(in_dim as u32, hidden_dim as u32, vocab_size as u32);

    let mut current_state = vec![0i8; in_dim];
    current_state[0] = 10;

    let mut generated_bpe_ids = vec![sub_vocab_bpe[0]];
    let start_time = Instant::now();

    for step in 1..4 {
        let (next_tok_idx, conf) = engine.step(&current_state, &w1, &b1, &w2, &b2);
        let bpe_id = sub_vocab_bpe[next_tok_idx as usize];
        generated_bpe_ids.push(bpe_id);

        println!(
            " [Hardware Step {}] Token Index: {} (Conf: {}%) -> BPE ID: {}",
            step, next_tok_idx, conf, bpe_id
        );

        current_state.fill(0);
        current_state[next_tok_idx as usize] = 10;
    }

    let elapsed = start_time.elapsed();
    let decoded_result = tokenizer.decode(&generated_bpe_ids);

    println!(" -------------------------------------------------------");
    println!(" [Decoded Output]       : \"{}\"", decoded_result);
    println!(" [Total Inference Time] : {:.2?}", elapsed);
    println!(" ========================================================\n");

    assert_eq!(decoded_result, "Luminous is AI ready.");
}
