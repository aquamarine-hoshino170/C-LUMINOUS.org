use lum_rust::gguf_loader::GgufModel;
use std::fs::File;
use std::io::Write;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 210;
const ROWS: usize = 4864;
const COLS: usize = 896;

fn fp16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = (e + 127) as u32;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f800000 | (mant << 13)
    } else {
        let exp32 = exp + 112;
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };

    f32::from_bits(bits)
}

fn q6k_decode_block(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), BLOCK_BYTES);
    assert_eq!(out.len(), BLOCK_SIZE);

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];

    let d_bits = u16::from_le_bytes([block[208], block[209]]);
    let d = fp16_to_f32(d_bits);

    for half in 0..2 {
        let ql0 = half * 64;
        let qh0 = half * 32;
        let sc0 = half * 8;
        let out0 = half * 128;

        for l in 0..32 {
            let is = l / 16;

            let a = ql[ql0 + l];
            let b = ql[ql0 + l + 32];
            let h = qh[qh0 + l];

            let q1 = (a & 0x0f) | ((h & 0x03) << 4);
            let q2 = (b & 0x0f) | (((h >> 2) & 0x03) << 4);
            let q3 = (a >> 4) | (((h >> 4) & 0x03) << 4);
            let q4 = (b >> 4) | (((h >> 6) & 0x03) << 4);

            let d1 = d * (scales[sc0 + is] as i8 as f32);
            let d2 = d * (scales[sc0 + is + 2] as i8 as f32);
            let d3 = d * (scales[sc0 + is + 4] as i8 as f32);
            let d4 = d * (scales[sc0 + is + 6] as i8 as f32);

            out[out0 + l] = d1 * ((q1 as i32) - 32) as f32;
            out[out0 + l + 32] = d2 * ((q2 as i32) - 32) as f32;
            out[out0 + l + 64] = d3 * ((q3 as i32) - 32) as f32;
            out[out0 + l + 96] = d4 * ((q4 as i32) - 32) as f32;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let model = GgufModel::load(model_path)?;
    let raw = model.read_tensor_bytes("blk.0.ffn_down.weight")?;

    let expected = ROWS * COLS / BLOCK_SIZE * BLOCK_BYTES;

    println!("============================================================");
    println!("        LUMINOUS Q6_K DUAL-IMPLEMENTATION VERIFY");
    println!("============================================================");
    println!("Tensor : blk.0.ffn_down.weight");
    println!("Shape  : [{} × {}]", ROWS, COLS);
    println!("Blocks : {}", ROWS * COLS / BLOCK_SIZE);
    println!("Bytes  : {}", raw.len());
    println!("Expected bytes: {}", expected);

    assert_eq!(raw.len(), expected);

    let mut decoded = vec![0.0f32; ROWS * COLS];
    let blocks = raw.len() / BLOCK_BYTES;

    for b in 0..blocks {
        let start = b * BLOCK_BYTES;
        let end = start + BLOCK_BYTES;

        q6k_decode_block(
            &raw[start..end],
            &mut decoded[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE],
        );
    }

    let mut raw_file = File::create("target/q6k_verify/q6k_raw.bin")?;
    raw_file.write_all(raw)?;

    let mut rust_file = File::create("target/q6k_verify/rust_q6k.f32")?;
    for x in &decoded {
        rust_file.write_all(&x.to_le_bytes())?;
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    for &x in &decoded {
        min = min.min(x);
        max = max.max(x);
        sum += x as f64;
        sum_sq += (x as f64) * (x as f64);
    }

    let n = decoded.len() as f64;

    println!();
    println!("RUST DECODER");
    println!("values : {}", decoded.len());
    println!("min    : {:.12e}", min);
    println!("max    : {:.12e}", max);
    println!("mean   : {:.12e}", sum / n);
    println!("RMS    : {:.12e}", (sum_sq / n).sqrt());
    println!();
    println!("[✓] Rust Q6_K decode complete");
    println!("[✓] Raw tensor exported");
    println!("[✓] Rust float32 output exported");

    Ok(())
}
