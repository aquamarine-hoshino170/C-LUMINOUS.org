use lum_rust::gguf_loader::GgufModel;

fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;

    let v = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            (mant as f32) * 2.0f32.powi(-24)
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    };

    if sign != 0 {
        -v
    } else {
        v
    }
}

fn decode_q5(block: &[u8]) -> [f32; 32] {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    let qs = &block[6..22];

    let mut out = [0.0f32; 32];

    for j in 0..32 {
        let (low, high) = if j < 16 {
            (qs[j] & 0xf, ((qh >> j) & 1) as u8)
        } else {
            let k = j - 16;
            (qs[k] >> 4, ((qh >> (k + 16)) & 1) as u8)
        };

        let q = low | (high << 4);
        out[j] = d * ((q as i32) - 16) as f32;
    }

    out
}

fn main() {
    let path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let model = GgufModel::load(path).unwrap();

    println!("============================================================");
    println!("          QWEN GGUF TENSOR SANITY CHECK");
    println!("============================================================");

    // --------------------------------------------------------
    // TOKEN EMBEDDING
    // --------------------------------------------------------

    let emb = model.read_tensor_bytes("token_embd.weight").unwrap();

    println!();
    println!("TOKEN EMBEDDING");
    println!("bytes = {}", emb.len());

    println!("raw first 32 bytes:");

    for i in 0..32 {
        print!("{:02x} ", emb[i]);
    }

    println!();

    let d_bits = u16::from_le_bytes([emb[0], emb[1]]);
    let d = fp16_to_f32(d_bits);

    println!("d bits = 0x{:04x}", d_bits);
    println!("d      = {:.12e}", d);

    let first = decode_q5(&emb[0..22]);

    println!();
    println!("First Q5_0 block:");

    for i in 0..32 {
        println!("x[{:02}] = {:.12e}", i, first[i]);
    }

    let rms = (first.iter().map(|x| x * x).sum::<f32>() / 32.0).sqrt();

    println!();
    println!("first-block RMS = {:.12e}", rms);

    // --------------------------------------------------------
    // OUTPUT NORM
    // --------------------------------------------------------

    let norm = model.read_tensor_bytes("output_norm.weight").unwrap();

    println!();
    println!("OUTPUT NORM");
    println!("bytes = {}", norm.len());

    println!("first 32 raw bytes:");

    for i in 0..32 {
        print!("{:02x} ", norm[i]);
    }

    println!();

    for i in 0..8 {
        let off = i * 4;

        let v = f32::from_le_bytes([norm[off], norm[off + 1], norm[off + 2], norm[off + 3]]);

        println!("norm[{}] = {:.12e}", i, v);
    }

    println!();
    println!("[✓] TENSOR SANITY INSPECTION COMPLETE");
}
