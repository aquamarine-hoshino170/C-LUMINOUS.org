use lum_rust::gguf_loader::GgufModel;
use std::path::Path;

#[test]
fn test_qwen_tensor_extraction_v02() {
    let model_path = "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    if !Path::new(model_path).exists() {
        println!("⚠️ Model not found: {}", model_path);
        return;
    }

    println!("\n========================================================");
    println!("   LUMINOUS GGUF TENSOR EXTRACTION v0.2");
    println!("========================================================");

    let model = GgufModel::load(model_path).expect("GGUF load failed");

    let name = "token_embd.weight";
    let info = model
        .tensors
        .get(name)
        .expect("token_embd.weight not found");

    let (start, end) = model
        .tensor_range(name)
        .expect("Failed to calculate tensor range");

    let bytes = model
        .read_tensor_bytes(name)
        .expect("Failed to extract tensor");

    println!(" Tensor : {}", info.name);
    println!(" Dims   : {:?}", info.dimensions);
    println!(" Type   : {}", info.tensor_type);
    println!(" Offset : 0x{:X} (relative)", info.offset);
    println!(" Start  : 0x{:X} (absolute)", start);
    println!(" End    : 0x{:X}", end);
    println!(" Bytes  : {}", bytes.len());
    println!(" File   : {}", model.mmap.len());
    println!(
        " Bounds : {}",
        if end <= model.mmap.len() as u64 {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );

    println!(" First 32 bytes:");
    for (i, b) in bytes.iter().take(32).enumerate() {
        if i % 16 == 0 {
            print!("  ");
        }
        print!("{:02X} ", b);
        if i % 16 == 15 {
            println!();
        }
    }
    println!();

    assert!(!bytes.is_empty());
    assert!(end <= model.mmap.len() as u64);
}
