use lum_rust::gguf_loader::GgufModel;

use std::time::Instant;

const MODEL: &str =
    "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

const ROWS: usize = 896;
const COLS: usize = 128;

const BLOCK_VALUES: usize = 32;
const BLOCK_BYTES: usize = 34;

unsafe extern "C" {
    fn luminous_q8_0_gemv_scalar(
        data: *const u8,
        rows: usize,
        cols: usize,
        input: *const f32,
        output: *mut f32,
    );

    fn luminous_q8_0_gemv(
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

            bits = sign | (((e + 127) as u32) << 23) | (f << 13);
        }
    } else if exp == 31 {
        bits = sign | 0x7f800000 | (frac << 13);
    } else {
        bits = sign | (((exp - 15 + 127) as u32) << 23) | (frac << 13);
    }

    f32::from_bits(bits)
}

fn reference_gemv(data: &[u8], input: &[f32], output: &mut [f32]) {
    let blocks = ROWS * COLS / BLOCK_VALUES;

    for b in 0..blocks {
        let start = b * BLOCK_BYTES;

        let block = &data[start..start + BLOCK_BYTES];

        let d = half_to_float(u16::from_le_bytes([block[0], block[1]]));

        let qs = &block[2..34];

        let base = b * BLOCK_VALUES;

        for j in 0..BLOCK_VALUES {
            let global = base + j;

            let row = global / COLS;

            let col = global % COLS;

            let q = qs[j] as i8;

            output[row] += d * q as f32 * input[col];
        }
    }
}

fn validate(reference: &[f32], candidate: &[f32], label: &str) {
    const ATOL: f32 = 2.0e-5;
    const RTOL: f32 = 5.0e-5;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mean_abs = 0.0f64;
    let mut failures = 0usize;
    let mut worst = 0usize;

    for i in 0..reference.len() {
        let abs = (reference[i] - candidate[i]).abs();

        let rel = abs / reference[i].abs().max(1.0e-12);

        let allowed = ATOL + RTOL * reference[i].abs();

        if abs > max_abs {
            max_abs = abs;
            worst = i;
        }

        max_rel = max_rel.max(rel);
        mean_abs += abs as f64;

        if abs > allowed {
            failures += 1;
        }
    }

    mean_abs /= reference.len() as f64;

    println!("{} max abs     : {:.12e}", label, max_abs);

    println!("{} max rel     : {:.12e}", label, max_rel);

    println!("{} mean abs    : {:.12e}", label, mean_abs);

    println!("{} worst row   : {}", label, worst);

    println!("{} failures    : {}", label, failures);

    assert_eq!(failures, 0, "{} numerical validation failed", label);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" LUMINOUS AI — PHASE 4D Q8_0 GEMV");
    println!("============================================================");

    let model = GgufModel::load(MODEL)?;

    let data = model.read_tensor_bytes("blk.0.attn_v.weight")?;

    let blocks = ROWS * COLS / BLOCK_VALUES;

    let expected_bytes = blocks * BLOCK_BYTES;

    assert_eq!(data.len(), expected_bytes);

    println!("[✓] Tensor: blk.0.attn_v.weight");

    println!("[✓] Shape: {} × {}", ROWS, COLS);

    println!("[✓] Q8_0 blocks: {}", blocks);

    println!("[✓] Raw bytes: {}", data.len());

    let mut input = vec![0.0f32; COLS];

    for i in 0..COLS {
        let x = i as f32;

        input[i] = (x * 0.01937).sin() * 0.25 + (x * 0.00611).cos() * 0.125;
    }

    let mut reference = vec![0.0f32; ROWS];

    let mut scalar = vec![0.0f32; ROWS];

    let mut neon = vec![0.0f32; ROWS];

    println!();
    println!("------------------------------------------------------------");
    println!(" Q8_0 REFERENCE GEMV");
    println!("------------------------------------------------------------");

    reference_gemv(data, &input, &mut reference);

    println!("Reference[0]    : {:.12e}", reference[0]);

    println!("Reference[last] : {:.12e}", reference[ROWS - 1]);

    println!();
    println!("------------------------------------------------------------");
    println!(" Q8_0 SCALAR / NEON VALIDATION");
    println!("------------------------------------------------------------");

    unsafe {
        luminous_q8_0_gemv_scalar(
            data.as_ptr(),
            ROWS,
            COLS,
            input.as_ptr(),
            scalar.as_mut_ptr(),
        );

        luminous_q8_0_gemv(data.as_ptr(), ROWS, COLS, input.as_ptr(), neon.as_mut_ptr());
    }

    println!("Scalar[0]       : {:.12e}", scalar[0]);

    println!("NEON[0]         : {:.12e}", neon[0]);

    println!("Scalar[last]    : {:.12e}", scalar[ROWS - 1]);

    println!("NEON[last]      : {:.12e}", neon[ROWS - 1]);

    validate(&reference, &scalar, "Reference/Scalar");

    validate(&reference, &neon, "Reference/NEON");

    println!();
    println!("[✓] Q8_0 NUMERICAL VALIDATION PASSED");

    /*
     * Performance.
     */
    const ITERATIONS: usize = 5000;

    println!();
    println!("============================================================");
    println!(" PHASE 4D PERFORMANCE BENCHMARK");
    println!("============================================================");

    unsafe {
        luminous_q8_0_gemv_scalar(
            data.as_ptr(),
            ROWS,
            COLS,
            input.as_ptr(),
            scalar.as_mut_ptr(),
        );

        luminous_q8_0_gemv(data.as_ptr(), ROWS, COLS, input.as_ptr(), neon.as_mut_ptr());
    }

    let scalar_start = Instant::now();

    for _ in 0..ITERATIONS {
        unsafe {
            luminous_q8_0_gemv_scalar(
                data.as_ptr(),
                ROWS,
                COLS,
                input.as_ptr(),
                scalar.as_mut_ptr(),
            );
        }
    }

    let scalar_time = scalar_start.elapsed();

    let neon_start = Instant::now();

    for _ in 0..ITERATIONS {
        unsafe {
            luminous_q8_0_gemv(data.as_ptr(), ROWS, COLS, input.as_ptr(), neon.as_mut_ptr());
        }
    }

    let neon_time = neon_start.elapsed();

    let scalar_us = scalar_time.as_secs_f64() * 1.0e6 / ITERATIONS as f64;

    let neon_us = neon_time.as_secs_f64() * 1.0e6 / ITERATIONS as f64;

    let speedup = scalar_us / neon_us;

    let elements = (ROWS * COLS) as f64;

    let scalar_mps = elements / (scalar_us / 1.0e6) / 1.0e6;

    let neon_mps = elements / (neon_us / 1.0e6) / 1.0e6;

    println!("Iterations       : {}", ITERATIONS);

    println!("Scalar / GEMV    : {:.3} us", scalar_us);

    println!("NEON / GEMV      : {:.3} us", neon_us);

    println!("Speedup          : {:.2}x", speedup);

    println!("Scalar throughput: {:.2} M elements/s", scalar_mps);

    println!("NEON throughput  : {:.2} M elements/s", neon_mps);

    println!();
    println!("[✓] PHASE 4D Q8_0 COMPLETE");

    println!("============================================================");

    Ok(())
}
