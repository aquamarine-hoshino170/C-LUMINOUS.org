use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

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

fn read_string(f: &mut File) -> std::io::Result<String> {
    let len = read_u64(f)? as usize;
    let mut b = vec![0u8; len];
    f.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn skip_value(f: &mut File, ty: u32) -> std::io::Result<()> {
    match ty {
        0 => {
            let _ = read_u8(f)?;
        }
        1 => {
            let _ = read_i8(f)?;
        }
        2 => {
            let _ = read_u16(f)?;
        }
        3 => {
            let _ = read_i16(f)?;
        }
        4 => {
            let _ = read_u32(f)?;
        }
        5 => {
            let _ = read_i32(f)?;
        }
        6 => {
            let _ = read_f32(f)?;
        }
        7 => {
            let _ = read_bool(f)?;
        }
        8 => {
            let _ = read_string(f)?;
        }
        9 => {
            let arr_ty = read_u32(f)?;
            let n = read_u64(f)?;
            for _ in 0..n {
                skip_value(f, arr_ty)?;
            }
        }
        10 => {
            let _ = read_u64(f)?;
        }
        11 => {
            let _ = read_i64(f)?;
        }
        12 => {
            let _ = read_f64(f)?;
        }
        _ => {}
    }
    Ok(())
}

fn read_u8(f: &mut File) -> std::io::Result<u8> {
    let mut b = [0u8; 1];
    f.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_i8(f: &mut File) -> std::io::Result<i8> {
    Ok(read_u8(f)? as i8)
}

fn read_u16(f: &mut File) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    f.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16(f: &mut File) -> std::io::Result<i16> {
    Ok(read_u16(f)? as i16)
}

fn read_i32(f: &mut File) -> std::io::Result<i32> {
    Ok(read_u32(f)? as i32)
}

fn read_i64(f: &mut File) -> std::io::Result<i64> {
    Ok(read_u64(f)? as i64)
}

fn read_f32(f: &mut File) -> std::io::Result<f32> {
    Ok(f32::from_bits(read_u32(f)?))
}

fn read_f64(f: &mut File) -> std::io::Result<f64> {
    Ok(f64::from_bits(read_u64(f)?))
}

fn read_bool(f: &mut File) -> std::io::Result<bool> {
    Ok(read_u8(f)? != 0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("============================================================");
    println!("        LUMINOUS Q4_K_M REAL GGUF FORENSIC TEST");
    println!("============================================================");
    println!("FILE: {}", path);
    println!();

    let mut f = File::open(path)?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;

    println!("Magic: {}", String::from_utf8_lossy(&magic));

    let version = read_u32(&mut f)?;
    let tensor_count = read_u64(&mut f)?;
    let metadata_count = read_u64(&mut f)?;

    println!("Version: {}", version);
    println!("Tensors: {}", tensor_count);
    println!("Metadata: {}", metadata_count);
    println!();

    // Metadata
    println!("---------------- METADATA ----------------");

    for i in 0..metadata_count {
        let key = read_string(&mut f)?;
        let ty = read_u32(&mut f)?;

        if key.contains("alignment")
            || key.contains("general.architecture")
            || key.contains("general.name")
            || key.contains("general.file_type")
            || key.contains("tokenizer")
        {
            println!("[{}] {} (type={})", i, key, ty);
        }

        skip_value(&mut f, ty)?;
    }

    println!();
    println!("---------------- TENSOR INFO ----------------");

    let mut first_q4 = None;

    for i in 0..tensor_count {
        let name = read_string(&mut f)?;
        let n_dims = read_u32(&mut f)?;

        let mut dims = Vec::new();

        for _ in 0..n_dims {
            dims.push(read_u64(&mut f)?);
        }

        let ty = read_u32(&mut f)?;
        let offset = read_u64(&mut f)?;

        let type_name = match ty {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            _ => "OTHER",
        };

        if i < 12 || type_name.contains("Q4_K") {
            println!(
                "#{:03} {:50} dims={:?} type={}({}) offset={}",
                i, name, dims, type_name, ty, offset
            );
        }

        if first_q4.is_none() && ty == 12 {
            first_q4 = Some((name, dims, offset));
        }
    }

    println!();

    let Some((name, dims, offset)) = first_q4 else {
        println!("[!] No Q4_K tensor found.");
        return Ok(());
    };

    println!("============================================================");
    println!("FIRST Q4_K TENSOR");
    println!("============================================================");
    println!("name   = {}", name);
    println!("dims   = {:?}", dims);
    println!("offset = {}", offset);

    println!();
    println!("Reading first 144 bytes from tensor...");

    // GGUF tensor data base is normally aligned after tensor metadata.
    // Find the current file position and align to 32 bytes.
    let tensor_data_base = (f.stream_position()? + 31) & !31;

    println!("computed tensor data base = {}", tensor_data_base);

    let absolute = tensor_data_base + offset;

    println!("absolute tensor position = {}", absolute);

    f.seek(SeekFrom::Start(absolute))?;

    let mut raw = [0u8; 144];
    f.read_exact(&mut raw)?;

    println!();
    println!("RAW Q4_K FIRST 144 BYTES:");

    for row in 0..9 {
        print!("{:04X}: ", row * 16);

        for col in 0..16 {
            print!("{:02X} ", raw[row * 16 + col]);
        }

        println!();
    }

    println!();
    println!("Q4_K block size = 144 bytes");
    println!("Expected layout:");
    println!("  d      : 2 bytes");
    println!("  dmin   : 2 bytes");
    println!("  scales : 12 bytes");
    println!("  qs     : 128 bytes");
    println!();

    let d_bits = u16::from_le_bytes([raw[0], raw[1]]);
    let dmin_bits = u16::from_le_bytes([raw[2], raw[3]]);

    println!("d bits    = 0x{:04X}", d_bits);
    println!("dmin bits = 0x{:04X}", dmin_bits);

    println!();
    println!("[✓] REAL Q4_K TENSOR BLOCK INSPECTED");
    println!("============================================================");

    Ok(())
}
