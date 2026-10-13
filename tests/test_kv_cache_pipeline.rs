#![cfg(target_arch = "arm")]
use lum_rust::cxx_bridge::KVCache;
use std::time::Instant;

#[test]
fn test_kv_cache_ring_buffer() {
    println!("\n========================================================");
    println!(" ⚡ LUMINOUS ZERO-COPY KV-CACHE RING BUFFER (C++20)");
    println!("========================================================");

    let max_seq = 16u32;
    let head_dim = 64u32;

    let kv_cache = KVCache::new(max_seq, head_dim);

    let dummy_k = vec![1i8; head_dim as usize];
    let dummy_v = vec![2i8; head_dim as usize];

    let t0 = Instant::now();

    for step in 1..=20 {
        kv_cache.append(&dummy_k, &dummy_v);
        let current_len = kv_cache.len();
        let k_ptr = kv_cache.get_k_ptr();

        if step <= 16 {
            assert_eq!(current_len, step);
        } else {
            assert_eq!(current_len, max_seq);
        }

        println!(
            " [Token Step {:02}] Cache Size: {:02}/{} | Aligned MemPtr: {:p}",
            step, current_len, max_seq, k_ptr
        );
    }

    let append_time = t0.elapsed();
    println!(" -------------------------------------------------------");
    println!(" ⏱ Total KV-Cache Ingestion Time: {:.2?}", append_time);
    println!(
        " ⚡ Average Ingestion Latency      : {:.2?} / token",
        append_time / 20
    );
    println!(" ========================================================\n");

    assert_eq!(kv_cache.len(), max_seq);
}
