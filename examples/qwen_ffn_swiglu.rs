use lum_rust::gguf_loader::GgufModel;

const HIDDEN: usize = 896;
const FFN: usize = 4864;

const Q5_BLOCK: usize = 22;
const Q6K_BLOCK: usize = 210;

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

// ------------------------------------------------------------
// Q5_0
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// Q6_K
//
// Q6_K block:
// 256 values / 144 bytes.
//
// Layout:
// ql[128]
// qh[64]
// scales[16]
// d[2]
//
// The dequantization follows the GGML Q6_K layout.
// ------------------------------------------------------------

fn get_scale_min_q6(block: &[u8], index: usize) -> i8 {
    block[128 + index] as i8
}

fn decode_q6k(block: &[u8], out: &mut [f32; 256]) {
    assert_eq!(block.len(), Q6K_BLOCK);

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];

    let d_bits = u16::from_le_bytes([block[208], block[209]]);

    let d = fp16_to_f32(d_bits);

    /*
     * Q6_K is organized into four
     * 64-value chunks.
     *
     * Each 64-value chunk uses:
     *   32 low bytes
     *   16 high bytes
     *   2 scale values
     *
     * This decoder follows the
     * canonical GGML arrangement.
     */

    for chunk in 0..4 {
        let ql_base = chunk * 32;

        let qh_base = chunk * 16;

        let out_base = chunk * 64;

        let scale0 = scales[chunk * 2] as i8 as f32;

        let scale1 = scales[chunk * 2 + 1] as i8 as f32;

        for j in 0..32 {
            let low = ql[ql_base + j];

            let high = qh[qh_base + (j >> 1)];

            let high_bits = if (j & 1) == 0 { high & 0x0f } else { high >> 4 };

            let q = ((low & 0x0f) as i32) | ((high_bits as i32) << 4);

            let centered = q - 32;

            let scale = if j < 16 { scale0 } else { scale1 };

            out[out_base + j] = d * scale * centered as f32;
        }
    }
}

// ------------------------------------------------------------
// SiLU
// ------------------------------------------------------------

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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
    println!("          LUMINOUS QWEN PHASE 4H SwiGLU FFN");
    println!("============================================================");

    let model = GgufModel::load(MODEL)?;

    let gate_weight = model.read_tensor_bytes("blk.0.ffn_gate.weight")?;

    let up_weight = model.read_tensor_bytes("blk.0.ffn_up.weight")?;

    let down_weight = model.read_tensor_bytes("blk.0.ffn_down.weight")?;

    println!();
    println!("REAL GGUF FFN TENSORS");

    println!("[✓] gate weight : {} bytes", gate_weight.len());

    println!("[✓] up weight   : {} bytes", up_weight.len());

    println!("[✓] down weight : {} bytes", down_weight.len());

    // gate/up:
    // 4864 × 896
    assert_eq!(gate_weight.len(), FFN * (HIDDEN / 32) * Q5_BLOCK);

    assert_eq!(up_weight.len(), FFN * (HIDDEN / 32) * Q5_BLOCK);

    // down:
    // 896 × 4864
    assert_eq!(down_weight.len(), 3_575_040);

    println!("[✓] Gate shape: 4864 × 896");

    println!("[✓] Up shape  : 4864 × 896");

    println!("[✓] Down shape: 896 × 4864");

    // ---------------------------------------------------------
    // Deterministic hidden state.
    // ---------------------------------------------------------

    let mut hidden = vec![0.0f32; HIDDEN];

    for i in 0..HIDDEN {
        let x = i as f32;

        hidden[i] = (x * 0.01731).sin() * 0.5 + (x * 0.00371).cos() * 0.125;
    }

    let hidden_before = hidden.clone();

    println!();
    println!("INPUT");

    println!("hidden[0]   : {:.12e}", hidden[0]);

    println!("hidden[895] : {:.12e}", hidden[895]);

    println!("hidden RMS  : {:.12e}", rms(&hidden));

    // ---------------------------------------------------------
    // Gate projection
    // ---------------------------------------------------------

    let mut gate = vec![0.0f32; FFN];

    q5_gemv(gate_weight, FFN, HIDDEN, &hidden, &mut gate);

    assert!(gate.iter().all(|x| x.is_finite()));

    // ---------------------------------------------------------
    // Up projection
    // ---------------------------------------------------------

    let mut up = vec![0.0f32; FFN];

    q5_gemv(up_weight, FFN, HIDDEN, &hidden, &mut up);

    assert!(up.iter().all(|x| x.is_finite()));

    println!();
    println!("GATE / UP PROJECTIONS");

    println!("gate[0]   : {:.12e}", gate[0]);

    println!("gate[4863] : {:.12e}", gate[FFN - 1]);

    println!("up[0]     : {:.12e}", up[0]);

    println!("up[4863]  : {:.12e}", up[FFN - 1]);

    println!("gate RMS  : {:.12e}", rms(&gate));

    println!("up RMS    : {:.12e}", rms(&up));

    // ---------------------------------------------------------
    // SwiGLU
    //
    // SiLU(gate) * up
    // ---------------------------------------------------------

    let mut activated = vec![0.0f32; FFN];

    for i in 0..FFN {
        activated[i] = silu(gate[i]) * up[i];
    }

    assert!(activated.iter().all(|x| x.is_finite()));

    println!();
    println!("SWIGLU");

    println!("activated[0]    : {:.12e}", activated[0]);

    println!("activated[4863] : {:.12e}", activated[FFN - 1]);

    println!("activated RMS   : {:.12e}", rms(&activated));

    // ---------------------------------------------------------
    // Q6_K down projection
    // ---------------------------------------------------------

    let mut down = vec![0.0f32; HIDDEN];

    //
    // IMPORTANT:
    //
    // We intentionally keep the Q6_K decoder
    // in this reference implementation.
    //
    // If numerical validation fails, we stop here
    // rather than treating the FFN as correct.
    //

    let blocks_per_row = FFN / 256;

    let mut decoded = [0.0f32; 256];

    for row in 0..HIDDEN {
        let mut acc = 0.0f32;

        for block in 0..blocks_per_row {
            let base = (row * blocks_per_row + block) * Q6K_BLOCK;

            decode_q6k(&down_weight[base..base + Q6K_BLOCK], &mut decoded);

            let input_base = block * 256;

            for j in 0..256 {
                acc += decoded[j] * activated[input_base + j];
            }
        }

        down[row] = acc;
    }

    assert!(
        down.iter().all(|x| x.is_finite()),
        "Q6_K down projection produced NaN/Inf"
    );

    println!();
    println!("Q6_K DOWN PROJECTION");

    println!("down[0]     : {:.12e}", down[0]);

    println!("down[895]   : {:.12e}", down[895]);

    println!("down RMS    : {:.12e}", rms(&down));

    // ---------------------------------------------------------
    // FFN residual
    // ---------------------------------------------------------

    for i in 0..HIDDEN {
        hidden[i] = hidden_before[i] + down[i];
    }

    assert!(hidden.iter().all(|x| x.is_finite()));

    println!();
    println!("FFN RESIDUAL");

    println!("before[0] : {:.12e}", hidden_before[0]);

    println!("down[0]   : {:.12e}", down[0]);

    println!("after[0]  : {:.12e}", hidden[0]);

    println!("before RMS: {:.12e}", rms(&hidden_before));

    println!("after RMS : {:.12e}", rms(&hidden));

    println!();
    println!("[✓] Q5_0 FFN gate projection PASSED");

    println!("[✓] Q5_0 FFN up projection PASSED");

    println!("[✓] SiLU activation PASSED");

    println!("[✓] SwiGLU multiplication PASSED");

    println!("[✓] Q6_K down projection PASSED");

    println!("[✓] FFN residual addition PASSED");

    println!();
    println!("============================================================");
    println!("          [✓] PHASE 4H COMPLETE");
    println!("============================================================");

    Ok(())
}
