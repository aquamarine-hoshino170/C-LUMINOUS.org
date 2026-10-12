use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

fn read_u8(f: &mut File) -> std::io::Result<u8> {
    let mut b = [0u8; 1];
    f.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32(f: &mut File) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(f: &mut File) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn skip_string(f: &mut File) -> std::io::Result<()> {
    let n = read_u64(f)? as i64;
    f.seek(SeekFrom::Current(n))?;
    Ok(())
}

fn skip_value(f: &mut File, ty: u32) -> std::io::Result<()> {
    match ty {
        // UINT8
        0 => {
            read_u8(f)?;
        }

        // INT8
        1 => {
            read_u8(f)?;
        }

        // UINT16
        2 => {
            f.seek(SeekFrom::Current(2))?;
        }

        // INT16
        3 => {
            f.seek(SeekFrom::Current(2))?;
        }

        // UINT32
        4 => {
            f.seek(SeekFrom::Current(4))?;
        }

        // INT32
        5 => {
            f.seek(SeekFrom::Current(4))?;
        }

        // FLOAT32
        6 => {
            f.seek(SeekFrom::Current(4))?;
        }

        // BOOL
        7 => {
            read_u8(f)?;
        }

        // STRING
        8 => {
            skip_string(f)?;
        }

        // ARRAY
        9 => {
            let elem_ty = read_u32(f)?;
            let n = read_u64(f)?;

            for _ in 0..n {
                skip_value(f, elem_ty)?;
            }
        }

        // UINT64
        10 => {
            f.seek(SeekFrom::Current(8))?;
        }

        // INT64
        11 => {
            f.seek(SeekFrom::Current(8))?;
        }

        // FLOAT64
        12 => {
            f.seek(SeekFrom::Current(8))?;
        }

        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown GGUF metadata type: {}", ty),
            ));
        }
    }

    Ok(())
}

fn f16_to_f32(x: u16) -> f32 {
    let sign = ((x >> 15) as u32) << 31;
    let exp = ((x >> 10) & 0x1f) as i32;
    let frac = (x & 0x03ff) as u32;

    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut f = frac;
            let mut e = -14i32;

            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }

            f &= 0x3ff;

            sign | (((e + 127) as u32) << 23) | (f << 13)
        }
    } else if exp == 31 {
        sign | (0xff << 23) | (frac << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (frac << 13)
    };

    f32::from_bits(bits)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path =
        "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let mut f = File::open(path)?;

    println!("============================================================");
    println!("             LUMINOUS REAL Q4_K DEQUANTIZER");
    println!("============================================================");
    println!("FILE = {}", path);
    println!();

    // ---------------------------------------------------------
    // GGUF HEADER
    // ---------------------------------------------------------

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;

    let version = read_u32(&mut f)?;
    let tensor_count = read_u64(&mut f)?;
    let metadata_count = read_u64(&mut f)?;

    println!("Magic    = {}", String::from_utf8_lossy(&magic));
    println!("Version  = {}", version);
    println!("Tensors  = {}", tensor_count);
    println!("Metadata = {}", metadata_count);

    if &magic != b"GGUF" {
        return Err("Invalid GGUF magic".into());
    }

    // ---------------------------------------------------------
    // METADATA
    // ---------------------------------------------------------

    for _ in 0..metadata_count {
        skip_string(&mut f)?;
        let ty = read_u32(&mut f)?;
        skip_value(&mut f, ty)?;
    }

    println!();
    println!("Metadata skipped successfully.");

    // ---------------------------------------------------------
    // FIND TARGET TENSOR
    // ---------------------------------------------------------

    let target_name = "blk.11.ffn_down.weight";

    let mut target_offset = None;
    let mut target_type = None;
    let mut target_dims = Vec::new();

    for _ in 0..tensor_count {
        let name_len = read_u64(&mut f)? as usize;

        let mut name_buf = vec![0u8; name_len];
        f.read_exact(&mut name_buf)?;

        let name = String::from_utf8_lossy(&name_buf).to_string();

        let ndim = read_u32(&mut f)?;

        let mut dims = Vec::with_capacity(ndim as usize);

        for _ in 0..ndim {
            dims.push(read_u64(&mut f)?);
        }

        let ty = read_u32(&mut f)?;
        let offset = read_u64(&mut f)?;

        if name == target_name {
            target_offset = Some(offset);
            target_type = Some(ty);
            target_dims = dims;
        }
    }

    let offset = target_offset.ok_or("Target tensor not found")?;
    let ty = target_type.unwrap();

    println!();
    println!("============================================================");
    println!("TARGET TENSOR");
    println!("============================================================");
    println!("name   = {}", target_name);
    println!("dims   = {:?}", target_dims);
    println!("type   = {} (Q4_K expected = 12)", ty);
    println!("offset = {}", offset);

    if ty != 12 {
        return Err(format!("Expected Q4_K type 12, got {}", ty).into());
    }

    // ---------------------------------------------------------
    // TENSOR DATA BASE
    // ---------------------------------------------------------

    let descriptor_end = f.stream_position()?;

    // GGUF default alignment here is 32 bytes.
    let data_base = (descriptor_end + 31) & !31;
    let absolute = data_base + offset;

    println!();
    println!("descriptor end = {}", descriptor_end);
    println!("data base      = {}", data_base);
    println!("absolute       = {}", absolute);

    // ---------------------------------------------------------
    // READ FIRST Q4_K BLOCK
    // ---------------------------------------------------------

    f.seek(SeekFrom::Start(absolute))?;

    let mut block = [0u8; 144];
    f.read_exact(&mut block)?;

    println!();
    println!("============================================================");
    println!("FIRST REAL Q4_K BLOCK");
    println!("============================================================");

    for row in 0..9 {
        print!("{:04X}: ", row * 16);

        for col in 0..16 {
            print!("{:02X} ", block[row * 16 + col]);
        }

        println!();
    }

    // ---------------------------------------------------------
    // Q4_K HEADER
    // ---------------------------------------------------------

    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let dmin_bits = u16::from_le_bytes([block[2], block[3]]);

    let d = f16_to_f32(d_bits);
    let dmin = f16_to_f32(dmin_bits);

    println!();
    println!("------------------------------------------------------------");
    println!("Q4_K HEADER");
    println!("------------------------------------------------------------");

    println!("d bits    = 0x{:04X}", d_bits);
    println!("d         = {:.12e}", d);

    println!("dmin bits = 0x{:04X}", dmin_bits);
    println!("dmin      = {:.12e}", dmin);

    // ---------------------------------------------------------
    // PACKED DATA
    // ---------------------------------------------------------

    let scales = &block[4..16];
    let qs = &block[16..144];

    println!();
    println!("scales[12]:");

    for (i, byte) in scales.iter().enumerate() {
        print!("{:02X} ", byte);

        if (i + 1) % 12 == 0 {
            println!();
        }
    }

    println!();
    println!("first 32 q bytes:");

    for i in 0..32 {
        print!("{:02X} ", qs[i]);

        if (i + 1) % 16 == 0 {
            println!();
        }
    }

    // ---------------------------------------------------------
    // RAW Q4 NIBBLE DISTRIBUTION
    // ---------------------------------------------------------

    let mut q_values = [0u8; 256];

    for i in 0..256 {
        let byte = qs[i / 2];

        q_values[i] = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
    }

    let mut min_q = 15u8;
    let mut max_q = 0u8;
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;

    for &q in &q_values {
        min_q = min_q.min(q);
        max_q = max_q.max(q);

        let x = q as f64;

        sum += x;
        sum2 += x * x;
    }

    let mean = sum / 256.0;
    let rms = (sum2 / 256.0).sqrt();

    println!();
    println!("============================================================");
    println!("RAW Q4 NIBBLE SANITY");
    println!("============================================================");

    println!("count = 256");
    println!("min   = {}", min_q);
    println!("max   = {}", max_q);
    println!("mean  = {:.9}", mean);
    println!("RMS   = {:.9}", rms);

    println!();
    println!("first 32 q values:");

    for i in 0..32 {
        println!("q[{i:03}] = {}", q_values[i]);
    }

    // ---------------------------------------------------------
    // BASIC VALIDATION
    // ---------------------------------------------------------

    println!();
    println!("============================================================");
    println!("VALIDATION");
    println!("============================================================");

    println!("d finite       = {}", d.is_finite());
    println!("dmin finite    = {}", dmin.is_finite());
    println!("d > 0          = {}", d > 0.0);
    println!("dmin >= 0      = {}", dmin >= 0.0);
    println!("Q4 range       = {}..{}", min_q, max_q);

    if d.is_finite() && dmin.is_finite() && d > 0.0 && dmin >= 0.0 && min_q <= 15 && max_q <= 15 {
        println!();
        println!("[✓] REAL Q4_K BLOCK VALIDATED");
    } else {
        println!();
        println!("[!] SANITY CHECK FAILED");
    }

    println!();
    println!("NEXT STEP:");
    println!("Implement exact GGML Q4_K scale/min reconstruction.");
    println!("Then produce 256 real FP32 dequantized weights.");

    Ok(())
}
