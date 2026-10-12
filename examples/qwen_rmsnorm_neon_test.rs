use lum_rust::gguf_loader::GgufModel;

use std::time::Instant;

const MODEL: &str =
    "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

const HIDDEN: usize = 896;
const EPS: f32 = 1.0e-6;

unsafe extern "C" {
    fn luminous_rmsnorm_neon(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        n: usize,
        eps: f32,
    );
}

fn f32_slice(bytes: &[u8]) -> &[f32] {
    assert_eq!(bytes.len() % 4, 0);

    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

fn rmsnorm_reference(input: &[f32], weight: &[f32], output: &mut [f32]) {
    let mut sum_sq = 0.0f32;

    for &x in input {
        sum_sq += x * x;
    }

    let inv_rms = 1.0f32 / ((sum_sq / input.len() as f32) + EPS).sqrt();

    for i in 0..input.len() {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" LUMINOUS AI — PHASE 4B ARMv7 NEON RMSNORM");
    println!("============================================================");

    let model = GgufModel::load(MODEL)?;

    let weight_bytes = model.read_tensor_bytes("blk.0.attn_norm.weight")?;

    let weight = f32_slice(weight_bytes);

    assert_eq!(weight.len(), HIDDEN);

    println!("[✓] Tensor: blk.0.attn_norm.weight");
    println!("[✓] Hidden size: {}", HIDDEN);

    /*
     * Same deterministic input as reference test.
     */
    let mut input = vec![0.0f32; HIDDEN];

    for i in 0..HIDDEN {
        let x = i as f32;

        input[i] = (x * 0.01731).sin() * 0.5 + (x * 0.00371).cos() * 0.125;
    }

    let mut reference = vec![0.0f32; HIDDEN];
    let mut neon = vec![0.0f32; HIDDEN];

    rmsnorm_reference(&input, weight, &mut reference);

    /*
     * Warmup.
     */
    unsafe {
        luminous_rmsnorm_neon(
            input.as_ptr(),
            weight.as_ptr(),
            neon.as_mut_ptr(),
            HIDDEN,
            EPS,
        );
    }

    println!();
    println!("------------------------------------------------------------");
    println!(" NUMERICAL VALIDATION");
    println!("------------------------------------------------------------");

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mean_abs = 0.0f64;
    let mut worst = 0usize;

    for i in 0..HIDDEN {
        let abs_err = (reference[i] - neon[i]).abs();

        let rel_err = abs_err / reference[i].abs().max(1.0e-12);

        if abs_err > max_abs {
            max_abs = abs_err;
            worst = i;
        }

        max_rel = max_rel.max(rel_err);
        mean_abs += abs_err as f64;
    }

    mean_abs /= HIDDEN as f64;

    println!("Reference[0] : {:.12e}", reference[0]);
    println!("NEON[0]      : {:.12e}", neon[0]);

    println!("Reference[895] : {:.12e}", reference[895]);
    println!("NEON[895]      : {:.12e}", neon[895]);

    println!("Max abs error : {:.12e}", max_abs);
    println!("Max rel error : {:.12e}", max_rel);
    println!("Mean abs error: {:.12e}", mean_abs);
    println!("Worst index   : {}", worst);

    /*
     * Mixed tolerance because the NEON reduction order
     * is intentionally different from the scalar reference.
     */
    const ATOL: f32 = 2.0e-5;
    const RTOL: f32 = 5.0e-5;

    let mut failures = 0usize;

    for i in 0..HIDDEN {
        let abs_err = (reference[i] - neon[i]).abs();

        let allowed = ATOL + RTOL * reference[i].abs();

        if abs_err > allowed {
            failures += 1;
        }
    }

    println!("Mixed failures: {}", failures);

    if failures != 0 {
        println!();
        println!("[!] RMSNorm NEON VALIDATION FAILED");
        std::process::exit(1);
    }

    println!();
    println!("[✓] RMSNorm NEON NUMERICAL VALIDATION PASSED");

    /*
     * Benchmark.
     */
    let iterations = 100_000usize;

    let start = Instant::now();

    for _ in 0..iterations {
        unsafe {
            luminous_rmsnorm_neon(
                input.as_ptr(),
                weight.as_ptr(),
                neon.as_mut_ptr(),
                HIDDEN,
                EPS,
            );
        }
    }

    let elapsed = start.elapsed();

    let per_call = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;

    println!();
    println!("------------------------------------------------------------");
    println!(" NEON BENCHMARK");
    println!("------------------------------------------------------------");

    println!("Iterations     : {}", iterations);
    println!("Total time     : {:?}", elapsed);
    println!("Time / RMSNorm : {:.4} us", per_call);

    println!();
    println!("============================================================");
    println!(" PHASE 4B-NEON COMPLETE");
    println!("============================================================");

    Ok(())
}
