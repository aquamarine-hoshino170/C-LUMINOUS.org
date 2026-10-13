use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    /// Offset relative to the beginning of GGUF tensor-data.
    pub offset: u64,
}

pub struct GgufModel {
    pub mmap: Mmap,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub tensors: HashMap<String, TensorInfo>,
    /// Absolute file offset where tensor data begins.
    pub tensor_data_offset: u64,
    /// GGUF alignment. Normally 32 unless general.alignment says otherwise.
    pub alignment: u64,
}

impl GgufModel {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 24 {
            return Err("File too small to be a valid GGUF".into());
        }

        let magic = &mmap[0..4];
        if magic != b"GGUF" {
            return Err("Invalid GGUF magic bytes".into());
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into()?);
        let tensor_count = u64::from_le_bytes(mmap[8..16].try_into()?);
        let metadata_kv_count = u64::from_le_bytes(mmap[16..24].try_into()?);

        println!(
            " [GGUF] Magic: GGUF | Version: {} | Tensors: {} | KV Metadatas: {}",
            version, tensor_count, metadata_kv_count
        );

        let mut cursor = Cursor::new(&mmap[24..]);

        // GGUF tensor data alignment defaults to 32 bytes.
        // general.alignment is a UINT32/UINT64 metadata field in GGUF files.
        let mut alignment: u64 = 32;

        for i in 0..metadata_kv_count {
            let key = match Self::read_string(&mut cursor) {
                Ok(k) => k,
                Err(e) => {
                    return Err(
                        format!("Error reading key {}/{}: {}", i + 1, metadata_kv_count, e).into(),
                    )
                }
            };
            let val_type = Self::read_u32(&mut cursor)?;

            if key == "general.alignment" {
                match val_type {
                    4 => {
                        alignment = Self::read_u32(&mut cursor)? as u64;
                    }
                    10 => {
                        alignment = Self::read_u64(&mut cursor)?;
                    }
                    _ => {
                        Self::skip_val(&mut cursor, val_type)?;
                    }
                }

                if alignment == 0 || !alignment.is_power_of_two() {
                    return Err(format!("Invalid GGUF tensor alignment: {}", alignment).into());
                }
            } else if let Err(e) = Self::skip_val(&mut cursor, val_type) {
                return Err(format!(
                    "Error skipping value for key '{}' (type {}): {}",
                    key, val_type, e
                )
                .into());
            }
        }

        let mut tensors = HashMap::new();
        for i in 0..tensor_count {
            let name = match Self::read_string(&mut cursor) {
                Ok(n) => n,
                Err(e) => {
                    return Err(format!(
                        "Error parsing tensor name {}/{}: {}",
                        i + 1,
                        tensor_count,
                        e
                    )
                    .into())
                }
            };
            let n_dims = Self::read_u32(&mut cursor)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(Self::read_u64(&mut cursor)?);
            }
            let tensor_type = Self::read_u32(&mut cursor)?;
            let offset = Self::read_u64(&mut cursor)?;

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dimensions: dims,
                    tensor_type,
                    offset,
                },
            );
        }

        // cursor is positioned immediately after the tensor-info table.
        // GGUF tensor data starts at the next alignment boundary.
        let header_end = 24u64
            .checked_add(cursor.position())
            .ok_or("GGUF header size overflow")?;

        let tensor_data_offset = Self::align_up(header_end, alignment)?;

        let data_start = tensor_data_offset as usize;
        if data_start > mmap.len() {
            return Err(format!(
                "Tensor data offset {} is outside file ({} bytes)",
                tensor_data_offset,
                mmap.len()
            )
            .into());
        }

        println!(
            " [GGUF] Tensor data base: 0x{:X} | Alignment: {}",
            tensor_data_offset, alignment
        );

        Ok(Self {
            mmap,
            tensor_count,
            metadata_kv_count,
            tensors,
            tensor_data_offset,
            alignment,
        })
    }

    fn align_up(value: u64, alignment: u64) -> Result<u64, Box<dyn std::error::Error>> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err("Invalid alignment".into());
        }

        let mask = alignment - 1;
        value
            .checked_add(mask)
            .map(|v| v & !mask)
            .ok_or_else(|| "Alignment overflow".into())
    }

    /// Number of bytes occupied by one tensor according to its GGML type.
    fn tensor_byte_size(info: &TensorInfo) -> Result<u64, Box<dyn std::error::Error>> {
        let elements = info
            .dimensions
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or("Tensor element-count overflow")?;

        // GGML/GGUF type IDs:
        // 0 = F32
        // 1 = F16
        // 2 = Q4_0  (32 elements -> 18 bytes)
        // 3 = Q4_1  (32 elements -> 20 bytes)
        // 6 = Q5_0  (32 elements -> 22 bytes)
        // 7 = Q5_1  (32 elements -> 24 bytes)
        // 8 = Q8_0  (32 elements -> 34 bytes)
        // 9 = Q8_1  (32 elements -> 36 bytes)
        // 10 = Q2_K (256 elements -> 84 bytes)
        // 11 = Q3_K (256 elements -> 110 bytes)
        // 12 = Q4_K (256 elements -> 144 bytes)
        // 13 = Q5_K (256 elements -> 176 bytes)
        // 14 = Q6_K (256 elements -> 210 bytes)
        // 15 = Q8_K (256 elements -> 292 bytes)

        let (block_size, block_bytes): (u64, u64) = match info.tensor_type {
            0 => {
                return elements
                    .checked_mul(4)
                    .ok_or_else(|| "F32 tensor byte-size overflow".into())
            }

            1 => {
                return elements
                    .checked_mul(2)
                    .ok_or_else(|| "F16 tensor byte-size overflow".into())
            }

            2 => (32, 18),
            3 => (32, 20),
            6 => (32, 22),
            7 => (32, 24),
            8 => (32, 34),
            9 => (32, 36),

            10 => (256, 84),
            11 => (256, 110),
            12 => (256, 144),
            13 => (256, 176),
            14 => (256, 210),
            15 => (256, 292),

            other => {
                return Err(format!(
                    "Unsupported GGML/GGUF tensor type {} for tensor '{}'",
                    other, info.name
                )
                .into());
            }
        };

        if elements % block_size != 0 {
            return Err(format!(
                "Tensor '{}' has {} elements, not divisible by block size {} for type {}",
                info.name, elements, block_size, info.tensor_type
            )
            .into());
        }

        let blocks = elements / block_size;

        blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| "Quantized tensor byte-size overflow".into())
    }

    pub fn tensor_range(&self, name: &str) -> Result<(u64, u64), Box<dyn std::error::Error>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Tensor not found: {}", name))?;

        let start = self
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or("Tensor absolute offset overflow")?;

        let size = Self::tensor_byte_size(info)?;

        let end = start
            .checked_add(size)
            .ok_or("Tensor end offset overflow")?;

        if end > self.mmap.len() as u64 {
            return Err(format!(
                "Tensor '{}' exceeds file bounds: 0x{:X}..0x{:X}, file size 0x{:X}",
                name,
                start,
                end,
                self.mmap.len()
            )
            .into());
        }

        Ok((start, end))
    }

    /// Zero-copy view of the raw tensor bytes.
    pub fn read_tensor_bytes(&self, name: &str) -> Result<&[u8], Box<dyn std::error::Error>> {
        let (start, end) = self.tensor_range(name)?;
        Ok(&self.mmap[start as usize..end as usize])
    }

    fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, Box<dyn std::error::Error>> {
        let len = Self::read_u64(cursor)? as usize;
        let pos = cursor.position() as usize;
        let slice = cursor.get_ref();
        if pos + len > slice.len() {
            return Err("Unexpected EOF while reading string".into());
        }
        let s = String::from_utf8_lossy(&slice[pos..pos + len]).to_string();
        cursor.seek(SeekFrom::Current(len as i64))?;
        Ok(s)
    }

    fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32, Box<dyn std::error::Error>> {
        let mut buf = [0u8; 4];
        cursor.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64, Box<dyn std::error::Error>> {
        let mut buf = [0u8; 8];
        cursor.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn skip_val(
        cursor: &mut Cursor<&[u8]>,
        val_type: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // GGUF metadata value types:
        // 0 = UINT8
        // 1 = INT8
        // 2 = UINT16
        // 3 = INT16
        // 4 = UINT32
        // 5 = INT32
        // 6 = FLOAT32
        // 7 = BOOL
        // 8 = STRING
        // 9 = ARRAY
        // 10 = UINT64
        // 11 = INT64
        // 12 = FLOAT64

        match val_type {
            // 1-byte values
            0 | 1 | 7 => {
                cursor.seek(SeekFrom::Current(1))?;
            }

            // 2-byte values
            2 | 3 => {
                cursor.seek(SeekFrom::Current(2))?;
            }

            // 4-byte values
            4..=6 => {
                cursor.seek(SeekFrom::Current(4))?;
            }

            // STRING
            8 => {
                let len = Self::read_u64(cursor)?;

                let pos = cursor.position() as usize;
                let total = cursor.get_ref().len();

                let end = pos
                    .checked_add(len as usize)
                    .ok_or("GGUF string length overflow")?;

                if end > total {
                    return Err(format!(
                        "Unexpected EOF while skipping STRING: {} bytes requested, {} available",
                        len,
                        total.saturating_sub(pos)
                    )
                    .into());
                }

                cursor.seek(SeekFrom::Current(len as i64))?;
            }

            // ARRAY
            9 => {
                let arr_type = Self::read_u32(cursor)?;
                let arr_len = Self::read_u64(cursor)?;

                match arr_type {
                    // 1-byte array elements
                    0 | 1 | 7 => {
                        let bytes = arr_len;
                        cursor.seek(SeekFrom::Current(bytes as i64))?;
                    }

                    // 2-byte array elements
                    2 | 3 => {
                        let bytes = arr_len.checked_mul(2).ok_or("GGUF array size overflow")?;

                        cursor.seek(SeekFrom::Current(bytes as i64))?;
                    }

                    // 4-byte array elements
                    4..=6 => {
                        let bytes = arr_len.checked_mul(4).ok_or("GGUF array size overflow")?;

                        cursor.seek(SeekFrom::Current(bytes as i64))?;
                    }

                    // STRING array
                    8 => {
                        for _ in 0..arr_len {
                            let slen = Self::read_u64(cursor)?;

                            let pos = cursor.position() as usize;
                            let total = cursor.get_ref().len();

                            let end = pos
                                .checked_add(slen as usize)
                                .ok_or("GGUF string-array length overflow")?;

                            if end > total {
                                return Err("Unexpected EOF inside GGUF STRING array".into());
                            }

                            cursor.seek(SeekFrom::Current(slen as i64))?;
                        }
                    }

                    // UINT64 / INT64 / FLOAT64
                    10..=12 => {
                        let bytes = arr_len.checked_mul(8).ok_or("GGUF array size overflow")?;

                        cursor.seek(SeekFrom::Current(bytes as i64))?;
                    }

                    _ => {
                        return Err(format!("Unknown GGUF array value type: {}", arr_type).into());
                    }
                }
            }

            // 8-byte values
            10..=12 => {
                cursor.seek(SeekFrom::Current(8))?;
            }

            _ => {
                return Err(format!("Unsupported GGUF metadata value type: {}", val_type).into());
            }
        }

        Ok(())
    }
}

// ============================================================
// Q4_K REFERENCE DEQUANTIZER
// ============================================================
//
// Correctness-first scalar implementation.
// Q4_K block:
//   256 values
//   144 bytes
//
// This module is deliberately separate from the GGUF loader.
// Once numerical correctness is established, this implementation
// becomes the reference against which the NEON kernel is tested.
// ============================================================

pub mod q4k_reference {
    pub const BLOCK_SIZE: usize = 256;
    pub const BLOCK_BYTES: usize = 144;

    #[inline]
    fn read_u16_le(data: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes([data[offset], data[offset + 1]])
    }

    // Reference Q4_K scale/min unpacking.
    //
    // Groups 0..3:
    //   d = scales[j] & 63
    //   m = scales[j+4] & 63
    //
    // Groups 4..7:
    //   low 4 bits come from q[j+4]
    //   high 2 bits come from q[j-4] / q[j]
    #[inline]
    pub fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    /// Decode one complete Q4_K super-block.
    ///
    /// Layout:
    ///   [0..2]     fp16 d
    ///   [2..4]     fp16 dmin
    ///   [4..16]    12-byte packed scales/mins
    ///   [16..144]  128 packed 4-bit quants
    pub fn dequantize_block(block: &[u8]) -> Result<[f32; BLOCK_SIZE], String> {
        if block.len() != BLOCK_BYTES {
            return Err(format!(
                "Q4_K: expected {} bytes, got {}",
                BLOCK_BYTES,
                block.len()
            ));
        }

        let d = half_to_f32(read_u16_le(block, 0));
        let dmin = half_to_f32(read_u16_le(block, 2));

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&block[4..16]);

        let q = &block[16..144];

        let mut output = [0.0f32; BLOCK_SIZE];

        let mut out = 0usize;
        let mut q_offset = 0usize;
        let mut group = 0usize;

        // Q4_K is processed as four 64-value sections.
        //
        // Every section contains:
        //   first 32 values  = low nibbles
        //   next  32 values  = high nibbles
        //
        // Two scale/min groups are used per 64-value section.
        for _section in 0..4 {
            let (sc0, m0) = get_scale_min_k4(group, &scales);

            let (sc1, m1) = get_scale_min_k4(group + 1, &scales);

            let d0 = d * sc0 as f32;
            let m0f = dmin * m0 as f32;

            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;

            for i in 0..32 {
                let byte = q[q_offset + i];

                output[out + i] = d0 * (byte & 0x0F) as f32 - m0f;

                output[out + 32 + i] = d1 * (byte >> 4) as f32 - m1f;
            }

            out += 64;
            q_offset += 32;
            group += 2;
        }

        Ok(output)
    }

    pub fn half_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1f) as u32;
        let frac = (bits & 0x03ff) as u32;

        let bits32 = if exp == 0 {
            if frac == 0 {
                sign << 31
            } else {
                let mut f = frac;
                let mut e = -14i32;

                while (f & 0x0400) == 0 {
                    f <<= 1;
                    e -= 1;
                }

                f &= 0x03ff;

                let exp32 = (e + 127) as u32;

                (sign << 31) | (exp32 << 23) | (f << 13)
            }
        } else if exp == 31 {
            (sign << 31) | (255 << 23) | (frac << 13)
        } else {
            let exp32 = exp + (127 - 15);

            (sign << 31) | (exp32 << 23) | (frac << 13)
        };

        f32::from_bits(bits32)
    }
}
