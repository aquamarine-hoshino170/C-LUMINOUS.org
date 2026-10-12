use lum_rust::gguf_loader::GgufModel;

const MODEL: &str =
    "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

const HIDDEN: usize = 896;
const EPS: f32 = 1.0e-6;

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
    println!(" LUMINOUS AI — PHASE 4B QWEN RMSNORM");
    println!("============================================================");

    let model = GgufModel::load(MODEL)?;

    let weight_bytes = model.read_tensor_bytes("blk.0.attn_norm.weight")?;

    let weight = f32_slice(weight_bytes);

    assert_eq!(weight.len(), HIDDEN);

    println!("[✓] Tensor: blk.0.attn_norm.weight");
    println!("[✓] Hidden size: {}", HIDDEN);
    println!("[✓] Weight bytes: {}", weight_bytes.len());

    /*
     * Deterministic test vector.
     *
     * This is deliberately not random so that future NEON,
     * C, Rust, and transformer implementations can all compare
     * against exactly the same input.
     */
    let mut input = vec![0.0f32; HIDDEN];

    for i in 0..HIDDEN {
        let x = i as f32;
        input[i] = (x * 0.01731).sin() * 0.5 + (x * 0.00371).cos() * 0.125;
    }

    let mut output = vec![0.0f32; HIDDEN];

    rmsnorm_reference(&input, weight, &mut output);

    println!();
    println!("------------------------------------------------------------");
    println!(" RMSNORM REFERENCE");
    println!("------------------------------------------------------------");

    let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / HIDDEN as f32;

    let rms = (mean_sq + EPS).sqrt();

    println!("RMS            : {:.12e}", rms);
    println!("inv RMS        : {:.12e}", 1.0 / rms);

    println!("Output[0]      : {:.12e}", output[0]);
    println!("Output[1]      : {:.12e}", output[1]);
    println!("Output[2]      : {:.12e}", output[2]);
    println!("Output[895]    : {:.12e}", output[895]);

    let mut max_abs = 0.0f32;
    let mut min_out = f32::INFINITY;
    let mut max_out = f32::NEG_INFINITY;

    for &v in &output {
        assert!(v.is_finite());

        max_abs = max_abs.max(v.abs());
        min_out = min_out.min(v);
        max_out = max_out.max(v);
    }

    println!();
    println!("Min output     : {:.12e}", min_out);
    println!("Max output     : {:.12e}", max_out);
    println!("Max |output|   : {:.12e}", max_abs);

    /*
     * Verify the normalization identity:
     *
     * mean((output / weight)^2) ≈ 1
     *
     * The model weights are not necessarily 1, so we undo
     * them for this mathematical sanity check.
     */
    let mut normalized_sq = 0.0f32;

    for i in 0..HIDDEN {
        let w = weight[i];

        if w.abs() > 1.0e-12 {
            let n = output[i] / w;
            normalized_sq += n * n;
        }
    }

    let normalized_rms = (normalized_sq / HIDDEN as f32).sqrt();

    println!("Normalized RMS : {:.12e}", normalized_rms);

    if !normalized_rms.is_finite() {
        return Err("RMSNorm produced non-finite output".into());
    }

    if (normalized_rms - 1.0).abs() > 1.0e-5 {
        return Err(format!("RMSNorm sanity check failed: {}", normalized_rms).into());
    }

    println!();
    println!("[✓] RMSNorm reference calculation passed");
    println!("[✓] Output is finite");
    println!("[✓] RMS normalization identity passed");

    println!();
    println!("============================================================");
    println!(" PHASE 4B REFERENCE COMPLETE");
    println!("============================================================");

    Ok(())
}
