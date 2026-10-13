#![cfg(target_arch = "arm")]
use lum_rust::cxx_bridge::{KVCache, NeuralEngine};
use lum_rust::tokenizer::ByteLevelBpeTokenizer;
use std::time::Instant;

#[test]
fn test_live_attention_kv_generation() {
    println!("\n========================================================");
    println!(" ⚡ LUMINOUS ATTENTION QxK^T + KV-CACHE GENERATION ⚡");
    println!("========================================================");

    let max_seq = 16u32;
    let head_dim = 64u32;
    let vocab_size = 16u32;

    // 1. Initialize BPE Tokenizer with Strict Canonical Merges
    let mut tokenizer = ByteLevelBpeTokenizer::new();

    // "Luminous"
    tokenizer.add_merge(b"u", b"m");
    tokenizer.add_merge(b"L", b"um");
    tokenizer.add_merge(b"i", b"n");
    tokenizer.add_merge(b"in", b"o");
    tokenizer.add_merge(b"ino", b"u");
    tokenizer.add_merge(b"inou", b"s");
    tokenizer.add_merge(b"Lum", b"inous");

    // " is"
    tokenizer.add_merge(b" ", b"i");
    tokenizer.add_merge(b" i", b"s");

    // " fast"
    tokenizer.add_merge(b"f", b"a");
    tokenizer.add_merge(b"fa", b"s");
    tokenizer.add_merge(b"fas", b"t");
    tokenizer.add_merge(b" ", b"fast");

    // " on"
    tokenizer.add_merge(b"o", b"n");
    tokenizer.add_merge(b" ", b"on");

    // " ARMv7"
    tokenizer.add_merge(b"A", b"R");
    tokenizer.add_merge(b"AR", b"M");
    tokenizer.add_merge(b"v", b"7");
    tokenizer.add_merge(b"ARM", b"v7");
    tokenizer.add_merge(b" ", b"ARMv7");

    // " silicon."
    tokenizer.add_merge(b"s", b"i");
    tokenizer.add_merge(b"si", b"l");
    tokenizer.add_merge(b"sil", b"i");
    tokenizer.add_merge(b"sili", b"c");
    tokenizer.add_merge(b"silic", b"o");
    tokenizer.add_merge(b"silico", b"n");
    tokenizer.add_merge(b"silicon", b".");
    tokenizer.add_merge(b" ", b"silicon.");

    let sub_vocab = vec![
        *tokenizer
            .vocab
            .get(b"Luminous".as_slice())
            .expect("Missing 'Luminous'"),
        *tokenizer
            .vocab
            .get(b" is".as_slice())
            .expect("Missing ' is'"),
        *tokenizer
            .vocab
            .get(b" fast".as_slice())
            .expect("Missing ' fast'"),
        *tokenizer
            .vocab
            .get(b" on".as_slice())
            .expect("Missing ' on'"),
        *tokenizer
            .vocab
            .get(b" ARMv7".as_slice())
            .expect("Missing ' ARMv7'"),
        *tokenizer
            .vocab
            .get(b" silicon.".as_slice())
            .expect("Missing ' silicon.'"),
    ];

    // 2. Instantiate C++ Components
    let kv_cache = KVCache::new(max_seq, head_dim);
    let engine = NeuralEngine::new(head_dim, 128, vocab_size);

    // Feed-Forward Projection Weights (16 context -> vocab_size logits)
    let mut w_ffn = vec![0i8; 16 * vocab_size as usize];
    for t in 0..6 {
        let next_t = (t + 1) % 6;
        w_ffn[next_t * 16 + t] = 16;
    }
    let b_ffn = vec![0i32; vocab_size as usize];

    // Seed the KV-cache with prompt "Luminous"
    let current_token_idx = 0usize;
    let mut q_vec = vec![0i8; head_dim as usize];
    let mut k_vec = vec![0i8; head_dim as usize];
    let v_vec = vec![1i8; head_dim as usize];

    q_vec[current_token_idx] = 10;
    k_vec[current_token_idx] = 10;
    kv_cache.append(&k_vec, &v_vec);

    let mut generated_ids = vec![sub_vocab[current_token_idx]];
    let start_all = Instant::now();

    // 3. Autoregressive Attention Inference Loop
    for step in 1..6 {
        let t0 = Instant::now();

        let (next_token_idx, conf) = engine.step_kv(&q_vec, &kv_cache, &w_ffn, &b_ffn);
        let bpe_id = sub_vocab[next_token_idx as usize];
        generated_ids.push(bpe_id);

        k_vec.fill(0);
        k_vec[next_token_idx as usize] = 10;
        kv_cache.append(&k_vec, &v_vec);

        q_vec.fill(0);
        q_vec[next_token_idx as usize] = 10;

        let elapsed = t0.elapsed();
        println!(
            " [Step {}] Cache Slots: {:02}/{} | Pred: {:<12} | Conf: {:2}% | Time: {:.2?}",
            step,
            kv_cache.len(),
            max_seq,
            tokenizer.decode(&[bpe_id]),
            conf,
            elapsed
        );
    }

    let total_time = start_all.elapsed();
    let final_sentence = tokenizer.decode(&generated_ids);

    println!(" -------------------------------------------------------");
    println!(" 📝 Complete Live Text : \"{}\"", final_sentence);
    println!(" ⏱️ Total Pipeline Time : {:.2?}", total_time);
    println!(" ========================================================\n");

    assert_eq!(final_sentence, "Luminous is fast on ARMv7 silicon.");
}
