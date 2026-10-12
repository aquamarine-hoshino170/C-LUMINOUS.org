use std::time::Instant;

use lum_rust::gguf_loader::GgufModel;

extern "C" {
    fn luminous_q4k_neon_gemv(
        blocks: *const u8,
        input: *const f32,
        output: *mut f32,
        rows: usize,
        cols: usize,
    );
}

const BLOCK_VALUES: usize = 256;
const BLOCK_BYTES: usize = 144;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" LUMINOUS AI — PHASE 3B FULL Q4_K NEON GEMV");
    println!("============================================================");

    let model_path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let model = GgufModel::load(model_path)?;

    let tensor_name = "blk.11.ffn_down.weight";

    let info = model.tensors.get(tensor_name).ok_or("tensor not found")?;

    let rows = info.dimensions[0] as usize;
    let cols = info.dimensions[1] as usize;

    let elements = rows * cols;

    let raw = model.read_tensor_bytes(tensor_name)?;

    let blocks = raw.len() / BLOCK_BYTES;

    println!("[✓] Tensor: {}", tensor_name);
    println!("[✓] Shape: {} × {}", rows, cols);
    println!("[✓] Elements: {}", elements);
    println!("[✓] Q4_K blocks: {}", blocks);
    println!("[✓] Raw bytes: {}", raw.len());

    assert_eq!(elements, 4_358_144);
    assert_eq!(raw.len(), blocks * BLOCK_BYTES);

    /*
     * Deterministic 896-element input.
     */
    let input: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.00137).sin() * 0.25)
        .collect();

    /*
     * --------------------------------------------------------
     * RUST REFERENCE FULL GEMV
     * --------------------------------------------------------
     *
     * Q4_K blocks are contiguous in the flattened tensor.
     *
     * Since:
     *
     *     896 % 256 != 0
     *
     * a Q4_K block can cross a matrix-row boundary.
     *
     * We therefore map every decoded value using:
     *
     *     global_index
     *         -> row
     *         -> column
     */
    println!();
    println!("------------------------------------------------------------");
    println!(" RUST REFERENCE FULL GEMV");
    println!("------------------------------------------------------------");

    let ref_start = Instant::now();

    let mut reference = vec![0.0f32; rows];

    for b in 0..blocks {
        let begin = b * BLOCK_BYTES;
        let end = begin + BLOCK_BYTES;

        let block: &[u8; BLOCK_BYTES] = raw[begin..end].try_into()?;

        let decoded = lum_rust::gguf_loader::q4k_reference::dequantize_block(block)?;

        let global_base = b * BLOCK_VALUES;

        for j in 0..BLOCK_VALUES {
            let global = global_base + j;

            if global >= elements {
                break;
            }

            let row = global / cols;

            let col = global % cols;

            reference[row] += decoded[j] * input[col];
        }
    }

    let ref_time = ref_start.elapsed();

    println!("[✓] Reference output generated");
    println!("Reference time : {:?}", ref_time);

    /*
     * --------------------------------------------------------
     * ARMv7 NEON FULL GEMV
     * --------------------------------------------------------
     */
    println!();
    println!("------------------------------------------------------------");
    println!(" RAW Q4_K → ARMv7 NEON FULL GEMV");
    println!("------------------------------------------------------------");

    let mut output = vec![0.0f32; rows];

    let neon_start = Instant::now();

    unsafe {
        luminous_q4k_neon_gemv(
            raw.as_ptr(),
            input.as_ptr(),
            output.as_mut_ptr(),
            rows,
            cols,
        );
    }

    let neon_time = neon_start.elapsed();

    /*
     * --------------------------------------------------------
     * NUMERICAL VERIFICATION
     * --------------------------------------------------------
     */
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut sum_abs_error = 0.0f64;

    let mut max_index = 0usize;

    for i in 0..rows {
        let abs = (reference[i] - output[i]).abs();

        let rel = abs / reference[i].abs().max(1.0e-12);

        if abs > max_abs_error {
            max_abs_error = abs;
            max_index = i;
        }

        if rel > max_rel_error {
            max_rel_error = rel;
        }

        sum_abs_error += abs as f64;
    }

    let mean_abs_error = sum_abs_error / rows as f64;

    let speedup = ref_time.as_secs_f64() / neon_time.as_secs_f64();

    println!();
    println!("------------------------------------------------------------");
    println!(" RESULTS");
    println!("------------------------------------------------------------");

    println!("Reference[0] : {:.10}", reference[0]);
    println!("NEON[0]      : {:.10}", output[0]);

    println!("Reference[last] : {:.10}", reference[rows - 1]);
    println!("NEON[last]      : {:.10}", output[rows - 1]);

    println!();
    println!("Reference time : {:?}", ref_time);
    println!("NEON time      : {:?}", neon_time);
    println!("Speedup        : {:.2}x", speedup);

    println!();
    println!("Max abs error  : {:.12e}", max_abs_error);
    println!("Max rel error  : {:.12e}", max_rel_error);
    println!("Mean abs error : {:.12e}", mean_abs_error);
    println!("Worst row      : {}", max_index);

    /*
     * Robust floating-point validation.
     *
     * Pure relative error becomes unstable when the reference
     * value is close to zero.
     *
     * Therefore validate using:
     *
     *   abs_error <= ATOL + RTOL * |reference|
     *
     * This still catches real Q4_K mapping/dequantization bugs
     * while allowing normal FP32 reduction-order differences.
     */
    const ATOL: f32 = 2.0e-4;
    const RTOL: f32 = 5.0e-5;

    let mut mixed_failures: usize = 0;
    let mut worst_ratio: f32 = 0.0;
    let mut worst_ratio_row: usize = 0;

    for i in 0..rows {
        let abs_error = (reference[i] - output[i]).abs();

        let allowed = ATOL + RTOL * reference[i].abs();

        let ratio = if allowed > 0.0 {
            abs_error / allowed
        } else {
            0.0
        };

        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_ratio_row = i;
        }

        if abs_error > allowed {
            mixed_failures += 1;
        }
    }

    println!();
    println!("Mixed tolerance ATOL : {:.2e}", ATOL);
    println!("Mixed tolerance RTOL : {:.2e}", RTOL);
    println!("Mixed failures       : {}", mixed_failures);
    println!("Worst tolerance ratio: {:.6}", worst_ratio);
    println!("Worst tolerance row  : {}", worst_ratio_row);

    if mixed_failures == 0 {
        println!();
        println!("[✓] FULL Q4_K GEMV NUMERICAL VALIDATION PASSED");
        println!("[✓] Q4_K row/block mapping validated");
        println!("[✓] ARMv7 NEON vector accumulation validated");
    } else {
        println!();
        println!("[!] FULL Q4_K GEMV NUMERICAL VALIDATION FAILED");
        println!("[!] Numerical error exceeds mixed tolerance.");

        std::process::exit(1);
    }

    println!();
    println!("============================================================");
    println!(" PHASE 3B COMPLETE");
    println!("============================================================");

    Ok(())
}
