use std::time::Instant;

use lum_rust::gguf_loader::GgufModel;

extern "C" {
    fn luminous_q4k_neon_dot(blocks: *const u8, input: *const f32, n_blocks: usize) -> f32;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" LUMINOUS AI — PHASE 3 Q4_K NEON GEMV");
    println!("============================================================");

    let model_path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let model = GgufModel::load(model_path)?;

    let tensor_name = "blk.11.ffn_down.weight";

    let info = model.tensors.get(tensor_name).ok_or("tensor not found")?;

    let elements: usize = info.dimensions.iter().product::<u64>() as usize;

    let rows = info.dimensions[0] as usize;
    let cols = info.dimensions[1] as usize;

    let raw = model.read_tensor_bytes(tensor_name)?;

    println!("[✓] Tensor: {}", tensor_name);
    println!("[✓] Shape: {} × {}", rows, cols);
    println!("[✓] Elements: {}", elements);
    println!("[✓] Raw bytes: {}", raw.len());

    /*
     * For this tensor:
     *
     * [4864, 896]
     *
     * Each row contains 896 values.
     *
     * Q4_K block = 256 values = 144 bytes.
     *
     * Therefore:
     *
     * 896 / 256 = 3.5
     *
     * This tensor is NOT row-aligned to a single Q4_K block.
     *
     * So this first GEMV benchmark deliberately operates
     * on complete 256-value Q4_K blocks.
     */

    const BLOCK_VALUES: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let n_blocks = raw.len() / BLOCK_BYTES;

    println!("[✓] Q4_K blocks: {}", n_blocks);

    /*
     * Deterministic input vector.
     */
    let input: Vec<f32> = (0..BLOCK_VALUES)
        .map(|i| ((i as f32) * 0.00137).sin() * 0.25)
        .collect();

    /*
     * Reference:
     * use the existing Rust Q4_K decoder.
     */
    let decoded: [f32; BLOCK_VALUES];

    {
        let block: &[u8; 144] = raw[0..144].try_into()?;

        decoded = lum_rust::gguf_loader::q4k_reference::dequantize_block(block)?;
    }

    let scalar_start = Instant::now();

    let mut scalar = 0.0f32;

    for i in 0..BLOCK_VALUES {
        scalar += decoded[i] * input[i];
    }

    let scalar_time = scalar_start.elapsed();

    /*
     * NEON
     */
    let neon_start = Instant::now();

    let neon = unsafe { luminous_q4k_neon_dot(raw.as_ptr(), input.as_ptr(), 1) };

    let neon_time = neon_start.elapsed();

    let abs_error = (scalar - neon).abs();

    let rel_error = abs_error / scalar.abs().max(1e-12);

    println!();
    println!("------------------------------------------------------------");
    println!(" SINGLE Q4_K BLOCK GEMV");
    println!("------------------------------------------------------------");

    println!("Scalar result : {:.10}", scalar);
    println!("NEON result   : {:.10}", neon);

    println!("Scalar time   : {:?}", scalar_time);
    println!("NEON time     : {:?}", neon_time);

    println!();
    println!("Absolute error : {:.12e}", abs_error);
    println!("Relative error : {:.12e}", rel_error);

    if abs_error <= 1.0e-4 && rel_error <= 2.0e-5 {
        println!("[✓] Q4_K GEMV NUMERICAL VALIDATION PASSED");
    } else {
        println!("[!] Q4_K GEMV VALIDATION FAILED");
        std::process::exit(1);
    }

    println!();
    println!("============================================================");
    println!(" PHASE 3A COMPLETE");
    println!("============================================================");

    Ok(())
}
