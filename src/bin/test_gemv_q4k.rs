#![allow(
    clippy::needless_range_loop,
    clippy::upper_case_acronyms,
    clippy::len_without_is_empty
)]
#[inline(always)]
fn fp16_to_fp32(h: u16) -> f32 {
    let sign = ((h >> 15) & 0x0001) as u32;
    let exp = ((h >> 10) & 0x001F) as u32;
    let mant = (h & 0x03FF) as u32;

    let f_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = 0;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp_f32 = (127 - 15 - e + 1) as u32;
            let mant_f32 = (m & 0x03FF) << 13;
            (sign << 31) | (exp_f32 << 23) | mant_f32
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        let exp_f32 = (exp + (127 - 15)) << 23;
        let mant_f32 = mant << 13;
        (sign << 31) | exp_f32 | mant_f32
    };

    f32::from_bits(f_bits)
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

#[inline(always)]
pub fn extract_q4k_scales_mins(scales: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];

    for i in 0..4 {
        sc[i] = scales[i] & 63;
        m[i] = scales[i + 4] & 63;
    }

    for i in 0..4 {
        let sc_hi = (scales[i] >> 6) & 3;
        let m_hi = (scales[i + 4] >> 6) & 3;

        sc[i + 4] = (scales[i + 8] & 0x0F) | (sc_hi << 4);
        m[i + 4] = ((scales[i + 8] >> 4) & 0x0F) | (m_hi << 4);
    }

    (sc, m)
}

/// Computes dot-product directly on a 256-element BlockQ4_K and an FP32 vector slice
#[inline(always)]
pub fn vec_dot_q4_k(block: &BlockQ4_K, x: &[f32]) -> f32 {
    let d = fp16_to_fp32(block.d);
    let dmin = fp16_to_fp32(block.dmin);
    let (sc, m) = extract_q4k_scales_mins(&block.scales);

    let mut sum = 0.0f32;
    let mut q_offset = 0;

    for sb_pair in 0..4 {
        let sb0 = sb_pair * 2;
        let sb1 = sb_pair * 2 + 1;

        let d0 = d * (sc[sb0] as f32);
        let dm0 = dmin * (m[sb0] as f32);

        let d1 = d * (sc[sb1] as f32);
        let dm1 = dmin * (m[sb1] as f32);

        let x0 = &x[sb0 * 32..(sb0 + 1) * 32];
        let x1 = &x[sb1 * 32..(sb1 + 1) * 32];

        for j in 0..32 {
            let byte = block.qs[q_offset + j];
            let q0 = (byte & 0x0F) as f32;
            let q1 = (byte >> 4) as f32;

            let w0 = d0 * q0 - dm0;
            let w1 = d1 * q1 - dm1;

            sum += w0 * x0[j] + w1 * x1[j];
        }
        q_offset += 32;
    }

    sum
}

fn main() {
    let target_block = BlockQ4_K {
        d: 0x0172,
        dmin: 0x0964,
        scales: [
            0xBA, 0xFF, 0xF4, 0xF3, 0xBE, 0xFF, 0xE2, 0xF5, 0x8D, 0x55, 0x54, 0x62,
        ],
        qs: [
            0x8A, 0x96, 0x50, 0x79, 0x78, 0xA6, 0x97, 0xC5, 0x76, 0xBB, 0x2D, 0x98, 0xD5, 0x96,
            0x93, 0xC0, 0xA7, 0x56, 0xBF, 0x48, 0xED, 0x58, 0x89, 0xA9, 0xE6, 0xAC, 0x09, 0x96,
            0xCC, 0x74, 0xDB, 0x38, 0x41, 0xC4, 0x02, 0xC5, 0x83, 0xF5, 0x96, 0xC7, 0x86, 0x5B,
            0x64, 0x95, 0xDC, 0x90, 0xC7, 0x62, 0x84, 0x42, 0x47, 0x5E, 0x3B, 0x65, 0x70, 0xA4,
            0x43, 0x96, 0xE9, 0x22, 0xB0, 0xE1, 0xB8, 0x70, 0x83, 0xF6, 0x49, 0x93, 0x96, 0xD2,
            0x84, 0x4A, 0x74, 0x7F, 0x59, 0x68, 0x92, 0x62, 0x94, 0x33, 0xC9, 0x5B, 0x59, 0x37,
            0x70, 0xFD, 0x91, 0x96, 0x84, 0x6B, 0x85, 0x01, 0x59, 0xD3, 0xB3, 0xB8, 0xCB, 0x87,
            0xD5, 0x66, 0x97, 0x48, 0x80, 0x05, 0xD4, 0x4B, 0xF4, 0x8F, 0xF9, 0xDB, 0xF5, 0xC8,
            0xB7, 0x95, 0xD8, 0x77, 0x68, 0x0C, 0xA8, 0x76, 0xCA, 0x59, 0x81, 0xA7, 0x42, 0xA1,
            0x39, 0xA8,
        ],
    };

    // 256-element dummy activation vector (normalized)
    let mut x = [1.0f32; 256];
    for i in 0..256 {
        x[i] = ((i as f32) / 256.0) - 0.5;
    }

    let dot_prod = vec_dot_q4_k(&target_block, &x);
    println!("=== GEMV Q4_K DOT-PRODUCT RESULT ===");
    println!("Dot Product with test activation = {:+.7}", dot_prod);
}
