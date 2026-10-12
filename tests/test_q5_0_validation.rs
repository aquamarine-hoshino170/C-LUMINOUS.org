use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

// কোনো ডিপেন্ডেন্সি ছাড়া পিওর রাস্টে f16 থেকে f32 কনভার্টার
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x0001) as u32;
    let exp = ((bits >> 10) & 0x001F) as u32;
    let mant = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // সাবনরমাল সংখ্যা
            let mut m = mant;
            let mut e = 0;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e += 1;
            }
            (sign << 31) | ((127 - 15 + 1 - e) << 23) | ((m & 0x03FF) << 13)
        }
    } else if exp == 31 {
        // ইনফিনিটি বা NaN
        (sign << 31) | 0x7F800000 | (mant << 13)
    } else {
        // স্বাভাবিক সংখ্যা
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };

    f32::from_bits(f32_bits)
}

fn dequantize_q5_0_block(bytes: &[u8; 22]) -> (f32, [f32; 32]) {
    // 1. FP16 Scale 'd' আনপ্যাকিং
    let d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let d = f16_to_f32(d_bits);

    // 2. ৩২-বিট হাই বিটস 'qh'
    let qh = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);

    // 3. ১৬-বাইটের লোয়ার ও আপার নিবল 'qs'
    let qs = &bytes[6..22];

    let mut values = [0.0f32; 32];
    for i in 0..16 {
        let byte = qs[i];
        let low0 = byte & 0x0F;
        let low1 = byte >> 4;

        let high0 = ((qh >> i) & 1) as u8;
        let high1 = ((qh >> (i + 16)) & 1) as u8;

        let q0 = (low0 | (high0 << 4)) as i32 - 16;
        let q1 = (low1 | (high1 << 4)) as i32 - 16;

        values[i] = (q0 as f32) * d;
        values[i + 16] = (q1 as f32) * d;
    }

    (d, values)
}

#[test]
fn test_qwen_token_embd_q5_0() {
    let model_path = "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let mut file = File::open(model_path).expect("মডেল ফাইল ওপেন করা যায়নি");

    // পাইথনে ভেরিফাইড অফসেট: 0x8f9d560
    let abs_start = 0x8f9d560;
    file.seek(SeekFrom::Start(abs_start)).expect("Seek failed");

    let mut block = [0u8; 22];
    file.read_exact(&mut block).expect("Read failed");

    let (d, decoded) = dequantize_q5_0_block(&block);

    println!("\n==========================================");
    println!("[RUST TEST] Scale d  : {}", d);
    println!("[RUST TEST] x[00]    : {}", decoded[0]);
    println!("[RUST TEST] x[01]    : {}", decoded[1]);
    println!("[RUST TEST] x[30]    : {}", decoded[30]);
    println!("[RUST TEST] x[31]    : {}", decoded[31]);

    // পাইথন ভ্যালুর সাথে মিলিয়ে দেখা
    assert_eq!(d, 293.5);
    assert_eq!(decoded[0], -3522.0);
    assert_eq!(decoded[1], -3228.5);
    assert_eq!(decoded[30], -4696.0);
    assert_eq!(decoded[31], -2641.5);

    let sum_sq: f32 = decoded.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / 32.0).sqrt();
    println!("[RUST TEST] Block RMS: {}", rms);
    println!("==========================================");

    assert!((rms - 2618.4699).abs() < 1e-2);
}
