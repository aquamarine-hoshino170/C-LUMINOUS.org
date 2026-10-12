use lum_rust::gguf_loader::GgufModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("============================================================");
    println!(" LUMINOUS AI — PHASE 4A QWEN TENSOR INVENTORY");
    println!("============================================================");

    let model = GgufModel::load(path)?;

    println!("Tensor count : {}", model.tensor_count);
    println!("KV metadata  : {}", model.metadata_kv_count);
    println!();

    let mut names: Vec<_> = model.tensors.keys().collect();
    names.sort();

    for name in names {
        let t = &model.tensors[name];

        if name.starts_with("blk.0.")
            || name == "token_embd.weight"
            || name == "output.weight"
            || name == "output_norm.weight"
        {
            println!(
                "{:<45} shape={:?} type={}",
                name, t.dimensions, t.tensor_type
            );
        }
    }

    println!();
    println!("============================================================");
    println!(" BLOCK 0 TENSOR MAP");
    println!("============================================================");

    for name in model.tensors.keys() {
        if name.starts_with("blk.0.") {
            let t = &model.tensors[name];

            let (start, end) = model.tensor_range(name)?;

            println!(
                "{:<45} shape={:?} type={} bytes={}",
                name,
                t.dimensions,
                t.tensor_type,
                end - start
            );
        }
    }

    println!();
    println!("[✓] Tensor inventory complete");

    Ok(())
}
