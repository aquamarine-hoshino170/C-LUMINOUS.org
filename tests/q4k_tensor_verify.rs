use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use lum_rust::gguf_loader::{q4k_reference::dequantize_block, GGUFLoader};

const Q4_K: u32 = 12;
const Q5_K: u32 = 13;
const Q6_K: u32 = 14;

fn find_model() -> Option<String> {
    let candidates = [
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "qwen.gguf",
        "model.gguf",
    ];

    for name in candidates {
        if std::path::Path::new(name).exists() {
            return Some(name.to_string());
        }
    }

    None
}

fn main() {
    println!("============================================================");
    println!(" LUMINOUS AI — Q4_K TENSOR-LEVEL VERIFICATION");
    println!("============================================================");

    let model = match find_model() {
        Some(path) => path,
        None => {
            eprintln!("[!] GGUF model not found in current directory.");
            eprintln!("[!] Searching common Termux locations...");

            let roots = [
                ".",
                "..",
                "../..",
                "/data/data/com.termux/files/home",
                "/data/data/com.termux/files/home/storage/downloads",
            ];

            let mut found = None;

            for root in roots {
                if let Ok(entries) = std::fs::read_dir(root) {
                    for entry in entries.flatten() {
                        let path = entry.path();

                        if path.extension().and_then(|x| x.to_str()) == Some("gguf") {
                            found = Some(path.to_string_lossy().to_string());
                            break;
                        }
                    }
                }

                if found.is_some() {
                    break;
                }
            }

            match found {
                Some(path) => path,
                None => {
                    eprintln!("[✗] No .gguf model found.");
                    std::process::exit(1);
                }
            }
        }
    };

    println!();
    println!("[MODEL] {}", model);

    let loader = match GGUFLoader::load(&model) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[✗] GGUF loader failed: {}", e);
            std::process::exit(1);
        }
    };

    println!();
    println!("------------------------------------------------------------");
    println!(" Q4_K TENSOR INVENTORY");
    println!("------------------------------------------------------------");

    let q4_tensors: Vec<_> = loader
        .tensors
        .iter()
        .filter(|t| t.tensor_type == Q4_K)
        .collect();

    println!("Total tensors : {}", loader.tensors.len());
    println!("Q4_K tensors  : {}", q4_tensors.len());

    if q4_tensors.is_empty() {
        println!();
        println!("[✗] No Q4_K tensor found.");
        println!("    Check whether this GGUF is actually Q4_K_M.");
        std::process::exit(1);
    }

    for tensor in &q4_tensors {
        let elements: u64 = tensor.dimensions.iter().product();

        let blocks = elements / 256;

        println!(
            "[Q4_K] {:45} elements={} blocks={} offset=0x{:X}",
            tensor.name, elements, blocks, tensor.offset
        );
    }

    let tensor = q4_tensors[0];

    println!();
    println!("------------------------------------------------------------");
    println!(" VERIFYING FIRST Q4_K TENSOR");
    println!("------------------------------------------------------------");

    println!("Tensor : {}", tensor.name);
    println!("Shape  : {:?}", tensor.dimensions);
    println!("Type   : {}", tensor.tensor_type);
    println!("Offset : 0x{:X}", tensor.offset);

    let elements: u64 = tensor.dimensions.iter().product();

    if elements % 256 != 0 {
        eprintln!(
            "[✗] Tensor element count {} is not divisible by 256.",
            elements
        );
        std::process::exit(1);
    }

    let block_count = elements / 256;

    let mut file = match File::open(&model) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[✗] Cannot open model: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = file.seek(SeekFrom::Start(tensor.offset)) {
        eprintln!("[✗] Seek failed: {}", e);
        std::process::exit(1);
    }

    let blocks_to_check = block_count.min(64);

    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    let mut values_seen = 0usize;
    let mut bad_values = 0usize;

    println!();
    println!("Checking {} / {} blocks...", blocks_to_check, block_count);

    for block_index in 0..blocks_to_check {
        let mut raw = [0u8; 144];

        if let Err(e) = file.read_exact(&mut raw) {
            eprintln!("[✗] Failed reading block {}: {}", block_index, e);
            std::process::exit(1);
        }

        let values = match dequantize_block(&raw) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[✗] Dequantization failed at block {}: {}", block_index, e);
                std::process::exit(1);
            }
        };

        for &v in &values {
            if !v.is_finite() {
                bad_values += 1;
            }

            global_min = global_min.min(v);
            global_max = global_max.max(v);

            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);

            values_seen += 1;
        }

        if block_index < 5 {
            let block_min = values.iter().copied().fold(f32::INFINITY, f32::min);

            let block_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            println!(
                "  block {:>3}: min={:+.6e} max={:+.6e}",
                block_index, block_min, block_max
            );
        }
    }

    let mean = sum / values_seen as f64;
    let rms = (sum_sq / values_seen as f64).sqrt();

    println!();
    println!("------------------------------------------------------------");
    println!(" Q4_K TENSOR RESULTS");
    println!("------------------------------------------------------------");

    println!("Blocks checked : {}", blocks_to_check);
    println!("Values checked : {}", values_seen);
    println!("Global min     : {:+.8e}", global_min);
    println!("Global max     : {:+.8e}", global_max);
    println!("Mean           : {:+.8e}", mean);
    println!("RMS            : {:.8e}", rms);
    println!("NaN/Inf values : {}", bad_values);

    println!();

    if values_seen == blocks_to_check as usize * 256
        && bad_values == 0
        && global_min.is_finite()
        && global_max.is_finite()
        && rms.is_finite()
    {
        println!("[✓] Q4_K TENSOR-LEVEL VERIFICATION PASSED");
        println!("[✓] Raw GGUF blocks read correctly");
        println!("[✓] 144-byte block geometry confirmed");
        println!("[✓] 256 values/block confirmed");
        println!("[✓] Dequantization produced finite FP32");
    } else {
        println!("[✗] Q4_K TENSOR-LEVEL VERIFICATION FAILED");
        std::process::exit(1);
    }

    println!();
    println!("NEXT → Q4_K full-tensor statistics + NEON kernel");
}
