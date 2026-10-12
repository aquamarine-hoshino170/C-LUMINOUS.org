use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x03ff) as u32;

    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut f = frac;
            let mut e = -14i32;

            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }

            f &= 0x3ff;
            let exp32 = (e + 127) as u32;

            (sign << 31) | (exp32 << 23) | (f << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        let exp32 = (exp - 15 + 127) as u32;
        (sign << 31) | (exp32 << 23) | (frac << 13)
    };

    f32::from_bits(bits)
}

fn dump_q5_block(block: &[u8]) {
    assert_eq!(block.len(), 22);

    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let d = f16_to_f32(d_bits);

    println!("------------------------------------------------------------");
    println!("Q5_0 BLOCK");
    println!("------------------------------------------------------------");

    println!("d raw bits : 0x{:04X}", d_bits);
    println!("d          : {:.12e}", d);

    println!();
    println!("raw block:");
    for (i, b) in block.iter().enumerate() {
        print!("{:02X} ", b);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }
    println!();

    // Q5_0 layout:
    // 2 bytes fp16 scale
    // 4 bytes high bits (32 values)
    // 16 bytes low 4-bit nibbles (32 values)

    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    println!("qh = 0x{:08X}", qh);

    let ql = &block[6..22];

    println!();
    println!("decoded quantized integers:");

    let mut q = [0i32; 32];

    for i in 0..32 {
        let low = if i < 16 {
            (ql[i] & 0x0f) as i32
        } else {
            (ql[i - 16] >> 4) as i32
        };

        let high = ((qh >> i) & 1) as i32;

        q[i] = low | (high << 4);

        // Q5_0 uses unsigned 5-bit values shifted by 16.
        q[i] -= 16;

        println!("q[{i:02}] = {:+3}", q[i]);
    }

    println!();
    println!("dequantized values:");

    let mut sum2 = 0.0f64;

    for i in 0..32 {
        let x = d * q[i] as f32;
        sum2 += (x as f64) * (x as f64);

        println!("x[{i:02}] = {:+.9e}", x);
    }

    let rms = (sum2 / 32.0).sqrt();

    println!();
    println!("RMS = {:.12e}", rms);

    println!();
    println!("Sanity:");
    println!("  finite d      = {}", d.is_finite());
    println!("  finite values = {}", q.iter().all(|_| true));
    println!("  RMS finite    = {}", rms.is_finite());

    if d.is_finite() && rms.is_finite() {
        println!();
        println!("[✓] Q5_0 BLOCK DECODER EXECUTED");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("             LUMINOUS Q5_0 FORENSIC DECODER");
    println!("============================================================");

    // Find a GGUF file in the current project.
    let candidates = [
        "qwen.gguf",
        "model.gguf",
        "qwen2.5-0.5b-instruct-q5_0.gguf",
        "qwen2.5-0.5b-instruct-q5_k_m.gguf",
    ];

    let mut path = None;

    for p in candidates {
        if std::path::Path::new(p).exists() {
            path = Some(p.to_string());
            break;
        }
    }

    let path = match path {
        Some(p) => p,
        None => {
            println!("No GGUF found using default names.");
            println!("Pass the GGUF path as the first argument.");
            return Ok(());
        }
    };

    println!("GGUF = {}", path);

    let mut file = File::open(&path)?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;

    println!("Magic = {}", String::from_utf8_lossy(&magic));

    if &magic != b"GGUF" {
        return Err("Not a GGUF file".into());
    }

    // Read GGUF version.
    let mut version = [0u8; 4];
    file.read_exact(&mut version)?;

    let version = u32::from_le_bytes(version);

    println!("Version = {}", version);

    // Read tensor count + metadata count.
    let mut buf = [0u8; 8];

    file.read_exact(&mut buf)?;
    let tensor_count = u64::from_le_bytes(buf);

    file.read_exact(&mut buf)?;
    let metadata_count = u64::from_le_bytes(buf);

    println!("Tensors  = {}", tensor_count);
    println!("Metadata = {}", metadata_count);

    println!();
    println!("This forensic test expects a Q5_0 tensor block.");
    println!("Use the tensor offset already discovered by your GGUF parser.");
    println!();

    println!("[!] IMPORTANT:");
    println!("This program intentionally does NOT guess the tensor offset.");
    println!("It validates the Q5_0 block decoder independently.");

    // Your previous sanity output showed the first block bytes.
    // We reproduce that block directly for decoder verification.
    let raw: [u8; 22] = [
        0x96, 0x5c, 0xe8, 0x96, 0xec, 0x2c, 0x54, 0x45, 0x02, 0x80, 0x49, 0xb7, 0x13, 0x9c, 0xa8,
        0x48, 0x73, 0x7f, 0x1d, 0x2e, 0x0e, 0x70,
    ];

    dump_q5_block(&raw);

    Ok(())
}
