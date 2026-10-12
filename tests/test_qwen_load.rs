use lum_rust::gguf_loader::GgufModel;
use std::path::Path;

#[test]
fn test_load_qwen_metadata() {
    let model_path = "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    if !Path::new(model_path).exists() {
        println!(
            "⚠️ Model file not found at: {}. Please download it first!",
            model_path
        );
        return;
    }

    println!("\n========================================================");
    println!(" ⚡ LUMINOUS GGUF PARSER: QWEN-2.5-0.5B-INSTRUCT ⚡");
    println!("========================================================");

    let model = GgufModel::load(model_path).expect("Failed to load Qwen GGUF");

    println!(
        " ✅ Successfully Mapped {} Tensors into Luminous Memory!",
        model.tensor_count
    );

    // Inspect critical transformer layers
    let key_tensors = [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "output.weight",
    ];

    println!("\n 🔍 Inspecting Key Architectural Weights:");
    for name in &key_tensors {
        if let Some(info) = model.tensors.get(*name) {
            println!(
                "  [Found] {:<26} | Dim: {:?} | Type ID: {}",
                info.name, info.dimensions, info.tensor_type
            );
        } else {
            println!("  [Missing] {}", name);
        }
    }
    println!("========================================================\n");
}
