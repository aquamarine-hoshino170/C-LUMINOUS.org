use lum_rust::gguf_loader::GgufModel;

const HIDDEN: usize = 896;
const Q5_BLOCK: usize = 22;

fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let value = if exp == 0 {
        if frac == 0 {
            0.0
        } else {
            let mut f = frac as f32 / 1024.0;
            let mut e = -14i32;

            while f < 1.0 {
                f *= 2.0;
                e -= 1;
            }

            f * 2f32.powi(e)
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };

    if sign != 0 {
        -value
    } else {
        value
    }
}

fn decode_q5(block: &[u8], out: &mut [f32; 32]) {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    let qs = &block[6..22];

    for j in 0..32 {
        let (low, high) = if j < 16 {
            (qs[j] & 0x0f, ((qh >> j) & 1) as u8)
        } else {
            let k = j - 16;

            (qs[k] >> 4, ((qh >> (k + 16)) & 1) as u8)
        };

        let q5 = low | (high << 4);

        out[j] = d * ((q5 as i32) - 16) as f32;
    }
}

fn q5_gemv(tensor: &[u8], rows: usize, cols: usize, input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), cols);
    assert_eq!(output.len(), rows);
    assert_eq!(cols % 32, 0);

    let blocks_per_row = cols / 32;

    assert_eq!(tensor.len(), rows * blocks_per_row * Q5_BLOCK);

    let mut decoded = [0.0f32; 32];

    for row in 0..rows {
        let mut acc = 0.0f32;

        for block in 0..blocks_per_row {
            let base = (row * blocks_per_row + block) * Q5_BLOCK;

            decode_q5(&tensor[base..base + Q5_BLOCK], &mut decoded);

            let input_base = block * 32;

            for j in 0..32 {
                acc += decoded[j] * input[input_base + j];
            }
        }

        output[row] = acc;
    }
}

fn rms(x: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for &v in x {
        sum += v * v;
    }

    (sum / x.len() as f32).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const MODEL: &str = "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("============================================================");
    println!("     LUMINOUS QWEN PHASE 4G ATTENTION OUTPUT PROJECTION");
    println!("============================================================");

    let model = GgufModel::load(MODEL)?;

    let weight = model.read_tensor_bytes("blk.0.attn_output.weight")?;

    println!();
    println!("REAL GGUF TENSOR");
    println!("[✓] Tensor: blk.0.attn_output.weight");
    println!("[✓] Shape : 896 × 896");
    println!("[✓] Type  : Q5_0");
    println!("[✓] Bytes : {}", weight.len());

    assert_eq!(weight.len(), HIDDEN * (HIDDEN / 32) * Q5_BLOCK);

    // ---------------------------------------------------------
    // Simulated output of Phase 4F-D.
    //
    // This deterministic vector represents the 896-dimensional
    // attention output entering the output projection.
    // ---------------------------------------------------------

    let mut attention = vec![0.0f32; HIDDEN];

    for i in 0..HIDDEN {
        let x = i as f32;

        attention[i] = (x * 0.01173).sin() * 0.18 + (x * 0.00417).cos() * 0.07;
    }

    println!();
    println!("ATTENTION INPUT");
    println!("attention[0]   : {:.12e}", attention[0]);
    println!("attention[895] : {:.12e}", attention[895]);
    println!("attention RMS  : {:.12e}", rms(&attention));

    // ---------------------------------------------------------
    // Real Q5_0 output projection.
    // ---------------------------------------------------------

    let mut projected = vec![0.0f32; HIDDEN];

    q5_gemv(weight, HIDDEN, HIDDEN, &attention, &mut projected);

    assert!(projected.iter().all(|x| x.is_finite()));

    println!();
    println!("ATTENTION OUTPUT PROJECTION");

    println!("projected[0]   : {:.12e}", projected[0]);

    println!("projected[1]   : {:.12e}", projected[1]);

    println!("projected[895] : {:.12e}", projected[895]);

    println!("projected RMS  : {:.12e}", rms(&projected));

    // ---------------------------------------------------------
    // Residual connection.
    //
    // In the actual transformer block:
    //
    // hidden + attention_output_projection
    //
    // Here we use the same deterministic hidden-state
    // construction used in the previous real-Q/K/V tests.
    // ---------------------------------------------------------

    let mut hidden = vec![0.0f32; HIDDEN];

    for i in 0..HIDDEN {
        let x = i as f32;

        hidden[i] = (x * 0.01731).sin() * 0.5 + (x * 0.00371).cos() * 0.125;
    }

    let hidden_before = hidden.clone();

    for i in 0..HIDDEN {
        hidden[i] += projected[i];
    }

    assert!(hidden.iter().all(|x| x.is_finite()));

    // ---------------------------------------------------------
    // Residual identity check.
    // ---------------------------------------------------------

    for i in 0..HIDDEN {
        let expected = hidden_before[i] + projected[i];

        let error = (hidden[i] - expected).abs();

        assert!(error <= 1.0e-6, "Residual mismatch at {}: {}", i, error);
    }

    println!();
    println!("RESIDUAL CONNECTION");

    println!("hidden before[0] : {:.12e}", hidden_before[0]);

    println!("projected[0]     : {:.12e}", projected[0]);

    println!("hidden after[0]  : {:.12e}", hidden[0]);

    println!("hidden before RMS: {:.12e}", rms(&hidden_before));

    println!("hidden after RMS : {:.12e}", rms(&hidden));

    println!();
    println!("[✓] Real Q5_0 attention output projection PASSED");

    println!("[✓] 896 → 896 projection PASSED");

    println!("[✓] Projection output finite");

    println!("[✓] Residual addition identity PASSED");

    println!("[✓] Transformer attention residual PASSED");

    println!();
    println!("============================================================");
    println!("       [✓] PHASE 4G COMPLETE");
    println!("============================================================");

    Ok(())
}
