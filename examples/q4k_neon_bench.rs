use std::error::Error;
use std::time::Instant;

use lum_rust::gguf_loader::{q4k_reference, GgufModel};

extern "C" {
    fn luminous_neon_dot(a: *const f32, b: *const f32, n: usize) -> f32;
}

const MODEL_PATH: &str =
    "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

const ITERATIONS: usize = 20;

fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("============================================================");
    println!(" LUMINOUS AI — Q4_K ARMv7 NEON MATMUL BENCHMARK");
    println!("============================================================");
    println!("[ARCH] arm");
    println!("[TARGET] ARMv7");
    println!("[NEON] C/Clang NEON kernel");

    println!();
    println!("------------------------------------------------------------");
    println!(" LOAD QWEN GGUF");
    println!("------------------------------------------------------------");

    let model = GgufModel::load(MODEL_PATH)?;

    println!("Tensor     : blk.11.ffn_down.weight");

    let info = model
        .tensors
        .get("blk.11.ffn_down.weight")
        .ok_or("target tensor not found")?;

    let (start, end) = model.tensor_range("blk.11.ffn_down.weight")?;
    let raw = model.read_tensor_bytes("blk.11.ffn_down.weight")?;

    let elements: usize = info.dimensions.iter().product::<u64>() as usize;

    let blocks = elements / q4k_reference::BLOCK_SIZE;

    println!("Shape      : {:?}", info.dimensions);
    println!("Elements   : {}", elements);
    println!("Blocks     : {}", blocks);
    println!("Raw bytes  : {}", raw.len());

    println!();
    println!("------------------------------------------------------------");
    println!(" FULL TENSOR DEQUANTIZATION");
    println!("------------------------------------------------------------");

    let mut weights = Vec::<f32>::with_capacity(elements);

    for block_index in 0..blocks {
        let begin = block_index * q4k_reference::BLOCK_BYTES;

        let end_block = begin + q4k_reference::BLOCK_BYTES;

        let block_bytes = &raw[begin..end_block];

        let mut block = [0u8; q4k_reference::BLOCK_BYTES];
        block.copy_from_slice(block_bytes);

        let values = q4k_reference::dequantize_block(&block)?;

        weights.extend_from_slice(&values);
    }

    println!("Decoded values : {}", weights.len());

    println!();
    println!("------------------------------------------------------------");
    println!(" CREATE INPUT VECTOR");
    println!("------------------------------------------------------------");

    let mut input = Vec::<f32>::with_capacity(elements);

    for i in 0..elements {
        let x = i as f32 * 0.00137;
        input.push(x.sin() * 0.25);
    }

    println!("Input values : {}", input.len());

    println!();
    println!("------------------------------------------------------------");
    println!(" SCALAR REFERENCE");
    println!("------------------------------------------------------------");

    let reference = scalar_dot(&weights, &input);

    println!("Reference dot : {:.10}", reference);

    let scalar_start = Instant::now();

    let mut scalar_result = 0.0f32;

    for _ in 0..ITERATIONS {
        scalar_result = scalar_dot(&weights, &input);
    }

    let scalar_time = scalar_start.elapsed().as_secs_f64();

    let scalar_values_per_sec = (elements * ITERATIONS) as f64 / scalar_time / 1e6;

    println!("Scalar result : {:.10}", scalar_result);
    println!("Total time    : {:.6} s", scalar_time);
    println!("Values/sec    : {:.3} M", scalar_values_per_sec);

    println!();
    println!("------------------------------------------------------------");
    println!(" ARMv7 NEON — C/CLANG");
    println!("------------------------------------------------------------");

    let neon_start = Instant::now();

    let mut neon_result = 0.0f32;

    for _ in 0..ITERATIONS {
        unsafe {
            neon_result = luminous_neon_dot(weights.as_ptr(), input.as_ptr(), elements);
        }
    }

    let neon_time = neon_start.elapsed().as_secs_f64();

    let neon_values_per_sec = (elements * ITERATIONS) as f64 / neon_time / 1e6;

    let speedup = scalar_time / neon_time;

    println!("NEON result   : {:.10}", neon_result);
    println!("NEON result   : {:.10}", neon_result);
    println!("Total time    : {:.6} s", neon_time);
    println!("Values/sec    : {:.3} M", neon_values_per_sec);
    println!("Speedup       : {:.2}x", speedup);

    println!();
    println!("------------------------------------------------------------");
    println!(" NUMERICAL VERIFICATION");
    println!("------------------------------------------------------------");

    let abs_error = (neon_result - reference).abs();
    let rel_error = abs_error / reference.abs().max(1e-12);

    // SIMD accumulation changes floating-point summation order.
    // Therefore bit-exact equality is not required.
    let abs_tol: f32 = 1.0e-4;
    let rel_tol: f32 = 2.0e-5;

    println!("Absolute error : {:.12e}", abs_error);
    println!("Relative error : {:.12e}", rel_error);
    println!("Absolute tol   : {:.12e}", abs_tol);
    println!("Relative tol   : {:.12e}", rel_tol);

    if abs_error <= abs_tol && rel_error <= rel_tol {
        println!("[✓] Scalar / NEON numerical agreement PASSED");
        println!("[✓] SIMD reduction difference is within tolerance");
    } else {
        println!("[!] Scalar / NEON numerical agreement FAILED");
        println!("[!] Difference exceeds numerical tolerance");
    }

    println!();
    println!("------------------------------------------------------------");
    println!(" GGUF / NEON VALIDATION SUMMARY");
    println!("------------------------------------------------------------");
    println!("[✓] Real Qwen GGUF loaded");
    println!("[✓] Q4_K tensor loaded");
    println!("[✓] Full tensor dequantized");
    println!("[✓] ARMv7 C/Clang NEON kernel executed");

    if abs_error <= abs_tol && rel_error <= rel_tol {
        println!("[✓] Numerical validation PASSED");
    } else {
        println!("[!] Numerical validation FAILED");
    }

    println!();
    println!("Tensor byte range: 0x{:X}..0x{:X}", start, end);
    println!("============================================================");

    Ok(())
}
