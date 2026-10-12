use lum_rust::gguf_loader::GgufModel;
use std::time::Instant;

const ROWS: usize = 128;
const COLS: usize = 896;
const BLOCK_SIZE: usize = 32;
const BLOCK_BYTES: usize = 22;

unsafe extern "C" {
    fn luminous_q5_0_gemv(
        tensor: *const u8,
        rows: usize,
        cols: usize,
        input: *const f32,
        output: *mut f32,
    );

    fn luminous_q5_0_gemv_scalar(
        tensor: *const u8,
        rows: usize,
        cols: usize,
        input: *const f32,
        output: *mut f32,
    );

    fn luminous_q5_0_debug_block(block: *const u8, out32: *mut f32);

    fn luminous_q5_0_debug_dot32(block: *const u8, input: *const f32) -> f32;

}

fn fp16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = -14i32;

            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }

            m &= 0x3ff;
            let exp32 = (e + 127) as u32;

            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let exp32 = (exp - 15 + 127) as u32;
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };

    f32::from_bits(bits)
}

fn reference_gemv(tensor: &[u8], input: &[f32], output: &mut [f32]) {
    const BLOCKS_PER_ROW: usize = COLS / BLOCK_SIZE;

    assert_eq!(tensor.len(), ROWS * BLOCKS_PER_ROW * BLOCK_BYTES);

    assert_eq!(input.len(), COLS);
    assert_eq!(output.len(), ROWS);

    for row in 0..ROWS {
        let mut acc = 0.0f32;

        for block in 0..BLOCKS_PER_ROW {
            let block_index = row * BLOCKS_PER_ROW + block;

            let base = block_index * BLOCK_BYTES;

            let d_bits = u16::from_le_bytes([tensor[base], tensor[base + 1]]);

            let d = fp16_to_f32(d_bits);

            let qh = u32::from_le_bytes([
                tensor[base + 2],
                tensor[base + 3],
                tensor[base + 4],
                tensor[base + 5],
            ]);

            let qs = &tensor[base + 6..base + 22];

            let input_base = block * BLOCK_SIZE;

            for j in 0..BLOCK_SIZE {
                let q_low: u8;

                let high_bit: u8;

                if j < 16 {
                    q_low = qs[j] & 0x0f;
                    high_bit = ((qh >> j) & 1) as u8;
                } else {
                    let k = j - 16;

                    q_low = qs[k] >> 4;
                    high_bit = ((qh >> (k + 16)) & 1) as u8;
                }

                let q5 = q_low | (high_bit << 4);

                let value = d * ((q5 as i32) - 16) as f32;

                acc += value * input[input_base + j];
            }
        }

        output[row] = acc;
    }
}

fn validate(name: &str, reference: &[f32], actual: &[f32]) {
    const ATOL: f32 = 2.0e-5;
    const RTOL: f32 = 5.0e-5;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mean_abs = 0.0f32;
    let mut failures = 0usize;
    let mut worst_row = 0usize;
    let mut worst_ratio = 0.0f32;

    for i in 0..reference.len() {
        let abs = (reference[i] - actual[i]).abs();
        let rel = abs / reference[i].abs().max(1.0e-12);
        let allowed = ATOL + RTOL * reference[i].abs();
        let ratio = abs / allowed;

        if abs > max_abs {
            max_abs = abs;
            worst_row = i;
        }

        if rel > max_rel {
            max_rel = rel;
        }

        mean_abs += abs;

        if ratio > worst_ratio {
            worst_ratio = ratio;
        }

        if abs > allowed {
            failures += 1;
        }
    }

    mean_abs /= reference.len() as f32;

    println!("{name} max abs       : {max_abs:.12e}");
    println!("{name} max rel       : {max_rel:.12e}");
    println!("{name} mean abs      : {mean_abs:.12e}");
    println!("{name} worst row     : {worst_row}");
    println!("{name} worst ratio   : {worst_ratio:.6e}");
    println!("{name} failures      : {failures}");

    assert_eq!(failures, 0, "{name} numerical validation failed");
}

fn main() {
    println!("============================================================");
    println!(" LUMINOUS AI — PHASE 4E Q5_0 K PROJECTION");
    println!("============================================================");

    let path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let model = GgufModel::load(path).expect("failed to load GGUF");

    let tensor = model
        .read_tensor_bytes("blk.0.attn_k.weight")
        .expect("K tensor not found");

    println!("[✓] Tensor: blk.0.attn_k.weight");
    println!("[✓] GGUF Shape: [896, 128]");
    println!("[✓] GEMV logical shape: 128 × 896 → 128");
    println!("[✓] Q5_0 blocks: {}", tensor.len() / BLOCK_BYTES);
    println!("[✓] Raw bytes: {}", tensor.len());

    assert_eq!(tensor.len(), ROWS * (COLS / BLOCK_SIZE) * BLOCK_BYTES);

    println!();
    println!("------------------------------------------------------------");
    println!(" Q5_0 FIRST BLOCK CROSS-CHECK");
    println!("------------------------------------------------------------");

    print!("Raw[0..22]: ");
    for b in &tensor[0..22] {
        print!("{:02x} ", b);
    }
    println!();

    let d_bits = u16::from_le_bytes([tensor[0], tensor[1]]);

    let d = fp16_to_f32(d_bits);

    let qh = u32::from_le_bytes([tensor[2], tensor[3], tensor[4], tensor[5]]);

    println!("Rust d_bits : 0x{:04x}", d_bits);
    println!("Rust d      : {:.12e}", d);
    println!("Rust qh     : 0x{:08x}", qh);

    let qs = &tensor[6..22];

    let mut rust32 = [0.0f32; 32];

    for j in 0..16 {
        let q0 = (qs[j] & 0x0f) | ((((qh >> j) & 1) as u8) << 4);

        let q1 = (qs[j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4);

        rust32[j] = d * ((q0 as i32) - 16) as f32;

        rust32[j + 16] = d * ((q1 as i32) - 16) as f32;
    }

    let mut c32 = [0.0f32; 32];

    unsafe {
        luminous_q5_0_debug_block(tensor.as_ptr(), c32.as_mut_ptr());
    }

    println!();
    println!("idx        Rust decode          C decode          abs diff");

    let mut block_max = 0.0f32;

    for i in 0..32 {
        let diff = (rust32[i] - c32[i]).abs();

        if diff > block_max {
            block_max = diff;
        }

        println!(
            "{:02}   {:.12e}   {:.12e}   {:.3e}",
            i, rust32[i], c32[i], diff
        );
    }

    println!();
    println!("First-block max abs diff: {:.12e}", block_max);

    if block_max == 0.0 {
        println!("[✓] Rust/C Q5_0 block decoder MATCH");
    } else {
        println!("[!] Rust/C Q5_0 block decoder MISMATCH");
    }

    let mut input = vec![0.0f32; COLS];

    for i in 0..COLS {
        let x = i as f32;
        input[i] = (x * 0.01371).sin() * 0.25 + (x * 0.00731).cos() * 0.125;
    }

    let mut reference = vec![0.0f32; ROWS];
    let mut scalar = vec![0.0f32; ROWS];
    let mut neon = vec![0.0f32; ROWS];

    println!();
    println!("------------------------------------------------------------");
    println!(" Q5_0 FFI / INPUT POINTER DIAGNOSTIC");
    println!("------------------------------------------------------------");

    println!("Rust input[0..8]:");
    for i in 0..8 {
        println!("  input[{}] = {:.12e}", i, input[i]);
    }

    let mut rust_dot32 = 0.0f32;

    for j in 0..32 {
        rust_dot32 += rust32[j] * input[j];
    }

    let c_dot32 = unsafe { luminous_q5_0_debug_dot32(tensor.as_ptr(), input.as_ptr()) };

    println!();
    println!("Rust first-block dot32 : {:.12e}", rust_dot32);
    println!("C first-block dot32    : {:.12e}", c_dot32);
    println!(
        "Dot32 abs difference   : {:.12e}",
        (rust_dot32 - c_dot32).abs()
    );

    if (rust_dot32 - c_dot32).abs() < 1.0e-7 {
        println!("[✓] FFI input + first-block GEMV MATCH");
    } else {
        println!("[!] FFI/input mismatch detected");
    }

    println!();
    println!("------------------------------------------------------------");

    const BLOCKS_PER_ROW: usize = COLS / BLOCK_SIZE;

    /*
     * IMPORTANT:
     * Use exactly the same decoder formula as reference_gemv().
     * No alternate decoder and no C-derived values.
     */
    let mut rust_block_dots = vec![0.0f32; BLOCKS_PER_ROW];

    for b in 0..BLOCKS_PER_ROW {
        let base = b * BLOCK_BYTES;

        let d_bits = u16::from_le_bytes([tensor[base], tensor[base + 1]]);

        let d = fp16_to_f32(d_bits);

        let qh = u32::from_le_bytes([
            tensor[base + 2],
            tensor[base + 3],
            tensor[base + 4],
            tensor[base + 5],
        ]);

        let qs = &tensor[base + 6..base + 22];

        let mut dot = 0.0f32;

        for j in 0..32 {
            let q_low = if j < 16 {
                qs[j] & 0x0f
            } else {
                qs[j - 16] >> 4
            };

            let high_bit = if j < 16 {
                ((qh >> j) & 1) as u8
            } else {
                ((qh >> (j + 16)) & 1) as u8
            };

            let q5 = q_low | (high_bit << 4);

            let value = d * ((q5 as i32) - 16) as f32;

            dot += value * input[b * 32 + j];
        }

        rust_block_dots[b] = dot;
    }

    /*
     * Reconstruct row-0 total from those exact block dots.
     */
    let rust_row0_from_blocks: f32 = rust_block_dots.iter().copied().sum();

    /*
     * The FIRST BLOCK diagnostic already calculated rust32
     * from exactly block 0. Recompute its dot independently
     * from rust32 and compare.
     */
    let rust_first_from_decoded: f32 = (0..32).map(|j| rust32[j] * input[j]).sum();

    println!(
        "Rust first block from decoded[32] : {:.12e}",
        rust_first_from_decoded
    );

    println!(
        "Rust first block from block loop   : {:.12e}",
        rust_block_dots[0]
    );

    println!(
        "Difference between Rust calculations: {:.12e}",
        (rust_first_from_decoded - rust_block_dots[0]).abs()
    );

    println!();
    println!(
        "Rust row-0 sum from 28 blocks: {:.12e}",
        rust_row0_from_blocks
    );

    println!();
    println!("block        Rust block dot");

    for b in 0..BLOCKS_PER_ROW {
        println!("{:02}      {:.12e}", b, rust_block_dots[b]);
    }

    if (rust_first_from_decoded - rust_block_dots[0]).abs() < 1.0e-7 {
        println!();
        println!("[✓] Rust block-0 calculations agree");
    } else {
        println!();
        println!("[!] RUST REFERENCE BUG: block-0 calculations disagree");
    }

    println!();
    println!("------------------------------------------------------------");
    println!(" Q5_0 K REFERENCE GEMV");
    println!("------------------------------------------------------------");

    reference_gemv(tensor, &input, &mut reference);

    println!("Reference[0]    : {:.12e}", reference[0]);
    println!("Reference[last] : {:.12e}", reference[ROWS - 1]);

    unsafe {
        luminous_q5_0_gemv_scalar(
            tensor.as_ptr(),
            ROWS,
            COLS,
            input.as_ptr(),
            scalar.as_mut_ptr(),
        );

        luminous_q5_0_gemv(
            tensor.as_ptr(),
            ROWS,
            COLS,
            input.as_ptr(),
            neon.as_mut_ptr(),
        );
    }

    println!();
    println!("------------------------------------------------------------");
    println!(" Q5_0 K SCALAR / NEON VALIDATION");
    println!("------------------------------------------------------------");

    println!("Scalar[0]       : {:.12e}", scalar[0]);
    println!("NEON[0]         : {:.12e}", neon[0]);
    println!("Scalar[last]    : {:.12e}", scalar[ROWS - 1]);
    println!("NEON[last]      : {:.12e}", neon[ROWS - 1]);

    validate("Reference/Scalar", &reference, &scalar);
    validate("Reference/NEON", &reference, &neon);

    println!();
    println!("[✓] Q5_0 K PROJECTION NUMERICAL VALIDATION PASSED");

    const ITERS: usize = 5000;

    let start = Instant::now();

    for _ in 0..ITERS {
        unsafe {
            luminous_q5_0_gemv_scalar(
                tensor.as_ptr(),
                ROWS,
                COLS,
                input.as_ptr(),
                scalar.as_mut_ptr(),
            );
        }
    }

    let scalar_time = start.elapsed();

    let start = Instant::now();

    for _ in 0..ITERS {
        unsafe {
            luminous_q5_0_gemv(
                tensor.as_ptr(),
                ROWS,
                COLS,
                input.as_ptr(),
                neon.as_mut_ptr(),
            );
        }
    }

    let neon_time = start.elapsed();

    let scalar_us = scalar_time.as_secs_f64() * 1e6 / ITERS as f64;

    let neon_us = neon_time.as_secs_f64() * 1e6 / ITERS as f64;

    let speedup = scalar_us / neon_us;

    let elements = (ROWS * COLS) as f64;

    let scalar_m = elements / (scalar_us * 1e-6) / 1e6;

    let neon_m = elements / (neon_us * 1e-6) / 1e6;

    println!();
    println!("============================================================");
    println!(" PHASE 4E PERFORMANCE BENCHMARK");
    println!("============================================================");
    println!("Iterations       : {ITERS}");
    println!("Scalar / GEMV    : {scalar_us:.3} us");
    println!("NEON / GEMV      : {neon_us:.3} us");
    println!("Speedup          : {speedup:.2}x");
    println!("Scalar throughput: {scalar_m:.2} M elements/s");
    println!("NEON throughput  : {neon_m:.2} M elements/s");

    println!();
    println!("[✓] PHASE 4E Q5_0 K PROJECTION COMPLETE");
    println!("============================================================");
}
