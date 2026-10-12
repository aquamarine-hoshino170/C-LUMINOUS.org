use std::error::Error;
use std::time::Instant;

use lum_rust::gguf_loader::{q4k_reference, GgufModel};

extern "C" {
    fn luminous_q4k_neon_dot(blocks: *const u8, input: *const f32, n_blocks: usize) -> f32;
}

const MODEL_PATH: &str =
    "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

const ITERATIONS: usize = 20;

fn scalar_reference(weights: &[f32], input: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..weights.len() {
        sum += weights[i] * input[i];
    }

    sum
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("============================================================");
    println!(" LUMINOUS AI — Q4_K ON-THE-FLY ARMv7 NEON");
    println!("============================================================");

    let model = GgufModel::load(MODEL_PATH)?;

    let info = model
        .tensors
        .get("blk.11.ffn_down.weight")
        .ok_or("tensor not found")?;

    let raw = model.read_tensor_bytes("blk.11.ffn_down.weight")?;

    let elements = info.dimensions.iter().product::<u64>() as usize;

    let blocks = elements / q4k_reference::BLOCK_SIZE;

    println!("[✓] Qwen GGUF loaded");
    println!("[✓] Tensor: blk.11.ffn_down.weight");
    println!("[✓] Elements: {}", elements);
    println!("[✓] Blocks: {}", blocks);
    println!("[✓] Raw Q4_K bytes: {}", raw.len());

    println!();
    println!("------------------------------------------------------------");
    println!(" BUILD INPUT");
    println!("------------------------------------------------------------");

    let mut input = Vec::<f32>::with_capacity(elements);

    for i in 0..elements {
        let x = i as f32 * 0.00137;

        input.push(x.sin() * 0.25);
    }

    /*
     * Reference FP32 tensor.
     */
    let mut weights = Vec::<f32>::with_capacity(elements);

    for b in 0..blocks {
        let start = b * q4k_reference::BLOCK_BYTES;

        let end = start + q4k_reference::BLOCK_BYTES;

        let mut block = [0u8; q4k_reference::BLOCK_BYTES];

        block.copy_from_slice(&raw[start..end]);

        let values = q4k_reference::dequantize_block(&block)?;

        weights.extend_from_slice(&values);
    }

    println!("[✓] Reference FP32 tensor: {} values", weights.len());

    println!();
    println!("------------------------------------------------------------");
    println!(" SCALAR REFERENCE");
    println!("------------------------------------------------------------");

    let reference = scalar_reference(&weights, &input);

    let scalar_start = Instant::now();

    let mut scalar_result = 0.0f32;

    for _ in 0..ITERATIONS {
        scalar_result = scalar_reference(&weights, &input);
    }

    let scalar_time = scalar_start.elapsed().as_secs_f64();

    let scalar_m = (elements * ITERATIONS) as f64 / scalar_time / 1e6;

    println!("Reference dot : {:.10}", reference);

    println!("Scalar result : {:.10}", scalar_result);

    println!("Values/sec    : {:.3} M", scalar_m);

    println!();
    println!("------------------------------------------------------------");
    println!(" Q4_K RAW → ARMv7 NEON ON-THE-FLY");
    println!("------------------------------------------------------------");

    let neon_start = Instant::now();

    let mut neon_result = 0.0f32;

    for _ in 0..ITERATIONS {
        neon_result = unsafe { luminous_q4k_neon_dot(raw.as_ptr(), input.as_ptr(), blocks) };
    }

    let neon_time = neon_start.elapsed().as_secs_f64();

    let neon_m = (elements * ITERATIONS) as f64 / neon_time / 1e6;

    let speedup = scalar_time / neon_time;

    let abs_error = (neon_result - reference).abs();

    let rel_error = abs_error / reference.abs().max(1e-12);

    println!("NEON result   : {:.10}", neon_result);

    println!("Values/sec    : {:.3} M", neon_m);

    println!("Speedup       : {:.2}x", speedup);

    println!();
    println!("------------------------------------------------------------");
    println!(" NUMERICAL VERIFICATION");
    println!("------------------------------------------------------------");

    println!("Absolute error : {:.12e}", abs_error);

    println!("Relative error : {:.12e}", rel_error);

    /*
     * Initial acceptance tolerance.
     * The purpose here is numerical equivalence,
     * not bit-exact equality.
     */
    let abs_tol = 1.0e-4f32;
    let rel_tol = 2.0e-5f32;

    if abs_error <= abs_tol && rel_error <= rel_tol {
        println!("[✓] ON-THE-FLY Q4_K NUMERICAL VALIDATION PASSED");
    } else {
        println!("[!] ON-THE-FLY Q4_K NUMERICAL VALIDATION FAILED");

        println!("[!] Do NOT optimize further until dequantization matches");
    }

    println!();
    println!("============================================================");
    println!(" PHASE 2 COMPLETE");
    println!("============================================================");

    Ok(())
}
