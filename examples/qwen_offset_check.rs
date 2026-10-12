use lum_rust::gguf_loader::GgufModel;

fn main() {
    let path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let model = GgufModel::load(path).unwrap();

    println!("============================================================");
    println!("             QWEN GGUF OFFSET VERIFICATION");
    println!("============================================================");

    for name in [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
        "output.weight",
    ] {
        let info = model.tensors.get(name).unwrap();

        let (start, end) = model.tensor_range(name).unwrap();
        let bytes = model.read_tensor_bytes(name).unwrap();

        println!();
        println!("Tensor : {}", name);
        println!("Shape  : {:?}", info.dimensions);
        println!("Type   : {}", info.tensor_type);
        println!("Offset : 0x{:X}", info.offset);
        println!("Range  : 0x{:X} .. 0x{:X}", start, end);
        println!("Bytes  : {}", bytes.len());

        print!("First 16 bytes: ");
        for b in bytes.iter().take(16) {
            print!("{:02x} ", b);
        }
        println!();
    }

    println!();
    println!("[✓] GGUF OFFSET VERIFICATION COMPLETE");
}
