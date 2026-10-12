use std::time::Instant;

use lum_rust::gguf_loader::{q4k_reference::dequantize_block, GgufModel};

const Q4_K: u32 = 12;
const BLOCK_BYTES: usize = 144;
const BLOCK_SIZE: usize = 256;

fn main() {
    println!("============================================================");
    println!(" LUMINOUS AI — Q4_K TENSOR-LEVEL VERIFICATION");
    println!("============================================================");

    let model_path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!();
    println!("[MODEL] {}", model_path);

    let model = match GgufModel::load(model_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("[✗] GGUF load failed: {}", e);
            std::process::exit(1);
        }
    };

    println!();
    println!("------------------------------------------------------------");
    println!(" Q4_K TENSOR INVENTORY");
    println!("------------------------------------------------------------");

    let mut q4_names: Vec<String> = model
        .tensors
        .values()
        .filter(|tensor| tensor.tensor_type == Q4_K)
        .map(|tensor| tensor.name.clone())
        .collect();

    q4_names.sort();

    println!("Total tensors : {}", model.tensors.len());
    println!("Q4_K tensors  : {}", q4_names.len());

    if q4_names.is_empty() {
        eprintln!("[✗] No Q4_K tensors found.");
        std::process::exit(1);
    }

    for name in &q4_names {
        let info = &model.tensors[name];

        let elements: u64 = info.dimensions.iter().product();

        let blocks = elements / BLOCK_SIZE as u64;

        println!(
            "[Q4_K] {:50} shape={:?} elements={} blocks={}",
            name, info.dimensions, elements, blocks
        );
    }

    // Pick the first Q4_K tensor deterministically.
    let tensor_name = &q4_names[0];
    let info = &model.tensors[tensor_name];

    println!();
    println!("------------------------------------------------------------");
    println!(" SELECTED TENSOR");
    println!("------------------------------------------------------------");

    println!("Name       : {}", info.name);
    println!("Shape      : {:?}", info.dimensions);
    println!("Type       : {} (Q4_K)", info.tensor_type);
    println!("Rel offset : 0x{:X}", info.offset);

    let elements: u64 = info.dimensions.iter().product();

    if elements % BLOCK_SIZE as u64 != 0 {
        eprintln!(
            "[✗] {} elements is not divisible by {}.",
            elements, BLOCK_SIZE
        );
        std::process::exit(1);
    }

    let expected_blocks = elements / BLOCK_SIZE as u64;

    let expected_bytes = expected_blocks * BLOCK_BYTES as u64;

    println!("Elements   : {}", elements);
    println!("Blocks     : {}", expected_blocks);
    println!("Expected   : {} bytes", expected_bytes);

    println!();
    println!("------------------------------------------------------------");
    println!(" RAW TENSOR EXTRACTION");
    println!("------------------------------------------------------------");

    let raw = match model.read_tensor_bytes(tensor_name) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("[✗] Tensor extraction failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("Raw bytes  : {}", raw.len());

    if raw.len() as u64 != expected_bytes {
        eprintln!(
            "[✗] Byte-size mismatch: expected {}, got {}",
            expected_bytes,
            raw.len()
        );
        std::process::exit(1);
    }

    println!("[✓] Tensor byte size matches GGUF geometry.");

    println!();
    println!("------------------------------------------------------------");
    println!(" Q4_K DEQUANTIZATION");
    println!("------------------------------------------------------------");

    // Checking the first 64 blocks keeps the diagnostic fast
    // while still exercising real model data.
    let blocks_to_check = expected_blocks as usize;

    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    let mut values_seen = 0usize;
    let mut non_finite = 0usize;

    let decode_start = Instant::now();

    for block_index in 0..blocks_to_check {
        let start = block_index * BLOCK_BYTES;
        let end = start + BLOCK_BYTES;

        let block = &raw[start..end];

        let values = match dequantize_block(block) {
            Ok(values) => values,
            Err(e) => {
                eprintln!("[✗] Dequantization failed at block {}: {}", block_index, e);
                std::process::exit(1);
            }
        };

        let block_min = values.iter().copied().fold(f32::INFINITY, f32::min);

        let block_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if block_index < 8 {
            println!(
                "block {:>3}: min={:+.8e} max={:+.8e} first=[{:+.5e}, {:+.5e}, {:+.5e}, {:+.5e}]",
                block_index, block_min, block_max, values[0], values[1], values[2], values[3],
            );
        }

        for &value in &values {
            if !value.is_finite() {
                non_finite += 1;
            }

            global_min = global_min.min(value);
            global_max = global_max.max(value);

            sum += value as f64;
            sum_sq += (value as f64) * (value as f64);

            values_seen += 1;
        }
    }

    let decode_elapsed = decode_start.elapsed();

    let mean = sum / values_seen as f64;

    let rms = (sum_sq / values_seen as f64).sqrt();

    println!();
    println!("------------------------------------------------------------");
    println!(" Q4_K DECODE BENCHMARK");
    println!("------------------------------------------------------------");

    let seconds = decode_elapsed.as_secs_f64();
    let values_per_second = values_seen as f64 / seconds.max(f64::MIN_POSITIVE);

    let mb_per_second = (raw.len() as f64 / (1024.0 * 1024.0)) / seconds.max(f64::MIN_POSITIVE);

    println!("Decode time       : {:.6} s", seconds);
    println!(
        "Values / second   : {:.3} M",
        values_per_second / 1_000_000.0
    );
    println!("Raw throughput    : {:.3} MiB/s", mb_per_second);

    println!();
    println!("------------------------------------------------------------");
    println!(" Q4_K TENSOR RESULTS");
    println!("------------------------------------------------------------");
    println!("------------------------------------------------------------");

    println!("Blocks checked : {}", blocks_to_check);
    println!("Values checked : {}", values_seen);
    println!("Global min     : {:+.8e}", global_min);
    println!("Global max     : {:+.8e}", global_max);
    println!("Mean           : {:+.8e}", mean);
    println!("RMS            : {:.8e}", rms);
    println!("NaN / Inf      : {}", non_finite);

    println!();

    if values_seen == blocks_to_check * BLOCK_SIZE
        && non_finite == 0
        && global_min.is_finite()
        && global_max.is_finite()
        && rms.is_finite()
    {
        println!("============================================================");
        println!(" [✓] Q4_K TENSOR-LEVEL VERIFICATION PASSED");
        println!("============================================================");
        println!("[✓] Real Qwen GGUF tensor");
        println!("[✓] Q4_K tensor identified");
        println!("[✓] Raw tensor bytes extracted");
        println!("[✓] 144-byte block geometry");
        println!("[✓] 256 values/block");
        println!("[✓] FP32 dequantization");
        println!("[✓] No NaN/Inf");
        println!();
        println!("NEXT → Q4_K full-tensor verification → NEON MatMul");
    } else {
        println!("[✗] Q4_K tensor verification FAILED");
        std::process::exit(1);
    }
}
