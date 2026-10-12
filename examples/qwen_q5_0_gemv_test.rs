use lum_rust::gguf_loader::GgufModel;

const MODEL: &str =
    "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

const ROWS: usize = 896;
const COLS: usize = 896;
const BLOCK_BYTES: usize = 22;
const BLOCK_VALUES: usize = 32;

unsafe extern "C" {
    fn luminous_q5_0_gemv_scalar(
        data: *const u8,
        rows: usize,
        cols: usize,
        input: *const f32,
        output: *mut f32,
    );

    fn luminous_q5_0_gemv(
        data: *const u8,
        rows: usize,
        cols: usize,
        input: *const f32,
        output: *mut f32,
    );
}

fn half_to_float(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x03ff) as u32;

    let bits: u32;

    if exp == 0 {
        if frac == 0 {
            bits = sign;
        } else {
            let mut f = frac;
            let mut e = -14i32;

            while (f & 0x0400) == 0 {
                f <<= 1;
                e -= 1;
            }

            f &= 0x03ff;

            let e32 = (e + 127) as u32;

            bits = sign | (e32 << 23) | (f << 13);
        }
    } else if exp == 31 {
        bits = sign | 0x7f800000 | (frac << 13);
    } else {
        let e32 = (exp - 15 + 127) as u32;

        bits = sign | (e32 << 23) | (frac << 13);
    }

    f32::from_bits(bits)
}

fn q5_0_decode_block(block: &[u8], out: &mut [f32; 32]) {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);

    let d = half_to_float(d_bits);

    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    let qs = &block[6..22];

    for j in 0..16 {
        let q0 = (qs[j] & 0x0f) as i32 | ((((qh >> j) & 1) as i32) << 4);

        let q1 = (qs[j] >> 4) as i32 | ((((qh >> (j + 16)) & 1) as i32) << 4);

        out[j] = d * (q0 - 16) as f32;

        out[j + 16] = d * (q1 - 16) as f32;
    }
}

fn reference_gemv(data: &[u8], input: &[f32], output: &mut [f32]) {
    let blocks = ROWS * COLS / BLOCK_VALUES;

    let mut decoded = [0.0f32; 32];

    for b in 0..blocks {
        let start = b * BLOCK_BYTES;

        q5_0_decode_block(&data[start..start + BLOCK_BYTES], &mut decoded);

        let base = b * BLOCK_VALUES;

        for j in 0..BLOCK_VALUES {
            let global = base + j;

            let row = global / COLS;

            let col = global % COLS;

            output[row] += decoded[j] * input[col];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" LUMINOUS AI — PHASE 4C Q5_0 GEMV");
    println!("============================================================");

    let model = GgufModel::load(MODEL)?;

    let data = model.read_tensor_bytes("blk.0.attn_q.weight")?;

    let expected = ROWS * COLS / BLOCK_VALUES;

    assert_eq!(data.len(), expected * BLOCK_BYTES);

    println!("[✓] Tensor: blk.0.attn_q.weight");

    println!("[✓] Shape: {} × {}", ROWS, COLS);

    println!("[✓] Q5_0 blocks: {}", expected);

    println!("[✓] Raw bytes: {}", data.len());

    /*
     * Deterministic input.
     */
    let mut input = vec![0.0f32; COLS];

    for i in 0..COLS {
        let x = i as f32;

        input[i] = (x * 0.01371).sin() * 0.25 + (x * 0.00491).cos() * 0.125;
    }

    let mut reference = vec![0.0f32; ROWS];

    let mut neon = vec![0.0f32; ROWS];

    println!();
    println!("------------------------------------------------------------");
    println!(" Q5_0 REFERENCE GEMV");
    println!("------------------------------------------------------------");

    reference_gemv(data, &input, &mut reference);

    println!("Reference[0]   : {:.12e}", reference[0]);

    println!("Reference[last]: {:.12e}", reference[ROWS - 1]);

    println!();
    println!("------------------------------------------------------------");
    println!(" Q5_0 → ARMv7 NEON GEMV");
    println!("------------------------------------------------------------");

    unsafe {
        luminous_q5_0_gemv(data.as_ptr(), ROWS, COLS, input.as_ptr(), neon.as_mut_ptr());
    }

    println!("NEON[0]        : {:.12e}", neon[0]);

    println!("NEON[last]     : {:.12e}", neon[ROWS - 1]);

    let mut max_abs = 0.0f32;

    let mut max_rel = 0.0f32;

    let mut mean_abs = 0.0f64;

    let mut worst = 0usize;

    for i in 0..ROWS {
        let abs = (reference[i] - neon[i]).abs();

        let rel = abs / reference[i].abs().max(1.0e-12);

        if abs > max_abs {
            max_abs = abs;
            worst = i;
        }

        max_rel = max_rel.max(rel);
        mean_abs += abs as f64;
    }

    mean_abs /= ROWS as f64;

    println!();
    println!("------------------------------------------------------------");
    println!(" RESULTS");
    println!("------------------------------------------------------------");

    println!("Max abs error : {:.12e}", max_abs);

    println!("Max rel error : {:.12e}", max_rel);

    println!("Mean abs error: {:.12e}", mean_abs);

    println!("Worst row     : {}", worst);

    /*
     * Mixed tolerance.
     */
    const ATOL: f32 = 2.0e-4;
    const RTOL: f32 = 5.0e-5;

    let mut failures = 0usize;

    for i in 0..ROWS {
        let abs = (reference[i] - neon[i]).abs();

        let allowed = ATOL + RTOL * reference[i].abs();

        if abs > allowed {
            failures += 1;
        }
    }

    println!("Mixed failures: {}", failures);

    if failures != 0 {
        println!();
        println!("[!] Q5_0 GEMV NUMERICAL VALIDATION FAILED");
        std::process::exit(1);
    }

    println!();
    println!("[✓] Q5_0 GEMV NUMERICAL VALIDATION PASSED");

    println!();
    println!("============================================================");
    println!(" PHASE 4C-B PERFORMANCE BENCHMARK");
    println!("============================================================");

    use std::time::Instant;

    const ITERATIONS: usize = 1000;

    let mut scalar_out = vec![0.0f32; ROWS];

    let mut neon_out = vec![0.0f32; ROWS];

    /*
     * Warm-up.
     */
    unsafe {
        luminous_q5_0_gemv_scalar(
            data.as_ptr(),
            ROWS,
            COLS,
            input.as_ptr(),
            scalar_out.as_mut_ptr(),
        );

        luminous_q5_0_gemv(
            data.as_ptr(),
            ROWS,
            COLS,
            input.as_ptr(),
            neon_out.as_mut_ptr(),
        );
    }

    let scalar_start = Instant::now();

    for _ in 0..ITERATIONS {
        unsafe {
            luminous_q5_0_gemv_scalar(
                data.as_ptr(),
                ROWS,
                COLS,
                input.as_ptr(),
                scalar_out.as_mut_ptr(),
            );
        }
    }

    let scalar_time = scalar_start.elapsed();

    let neon_start = Instant::now();

    for _ in 0..ITERATIONS {
        unsafe {
            luminous_q5_0_gemv(
                data.as_ptr(),
                ROWS,
                COLS,
                input.as_ptr(),
                neon_out.as_mut_ptr(),
            );
        }
    }

    let neon_time = neon_start.elapsed();

    let scalar_ns = scalar_time.as_secs_f64() * 1.0e9;

    let neon_ns = neon_time.as_secs_f64() * 1.0e9;

    let scalar_per = scalar_ns / ITERATIONS as f64;

    let neon_per = neon_ns / ITERATIONS as f64;

    let speedup = scalar_per / neon_per;

    /*
     * Numerical comparison between scalar and NEON.
     */
    let mut max_abs = 0.0f32;

    let mut mean_abs = 0.0f64;

    for i in 0..ROWS {
        let err = (scalar_out[i] - neon_out[i]).abs();

        if err > max_abs {
            max_abs = err;
        }

        mean_abs += err as f64;
    }

    mean_abs /= ROWS as f64;

    /*
     * Q5_0 throughput.
     *
     * One GEMV processes the entire 896×896 matrix.
     * Count one multiply-add per matrix element.
     */
    let operations = (ROWS * COLS) as f64;

    let scalar_mops = operations / (scalar_per / 1.0e6);

    let neon_mops = operations / (neon_per / 1.0e6);

    println!();
    println!("Iterations       : {}", ITERATIONS);

    println!(
        "Scalar total     : {:.6} ms",
        scalar_time.as_secs_f64() * 1000.0
    );

    println!(
        "NEON total       : {:.6} ms",
        neon_time.as_secs_f64() * 1000.0
    );

    println!("Scalar / GEMV    : {:.6} us", scalar_per / 1000.0);

    println!("NEON / GEMV      : {:.6} us", neon_per / 1000.0);

    println!("Speedup          : {:.2}x", speedup);

    println!("Scalar throughput: {:.2} M elements/s", scalar_mops);

    println!("NEON throughput  : {:.2} M elements/s", neon_mops);

    println!("Scalar vs NEON max abs: {:.12e}", max_abs);

    println!("Scalar vs NEON mean abs: {:.12e}", mean_abs);

    if max_abs > 2.0e-4 {
        println!("[!] PERFORMANCE KERNEL VALIDATION FAILED");
        std::process::exit(1);
    }

    println!();
    println!("[✓] SCALAR vs NEON NUMERICAL VALIDATION PASSED");

    println!();
    println!("============================================================");
    println!(" PHASE 4C-B PERFORMANCE COMPLETE");
    println!("============================================================");

    println!();
    println!("============================================================");
    println!(" PHASE 4C Q5_0 CORRECTNESS COMPLETE");
    println!("============================================================");

    Ok(())
}
