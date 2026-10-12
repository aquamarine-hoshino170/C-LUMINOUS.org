use memmap2::Mmap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, Box<dyn std::error::Error>> {
    let mut buf8 = [0u8; 8];
    cursor.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8) as usize;
    let pos = cursor.position() as usize;
    let slice = cursor.get_ref();
    if pos + len > slice.len() {
        return Err("EOF".into());
    }
    let s = String::from_utf8_lossy(&slice[pos..pos + len]).to_string();
    cursor.seek(SeekFrom::Current(len as i64))?;
    Ok(s)
}

fn skip_val(cursor: &mut Cursor<&[u8]>, val_type: u32) -> Result<(), Box<dyn std::error::Error>> {
    match val_type {
        0..=2 => {
            cursor.seek(SeekFrom::Current(1))?;
        }
        3..=4 => {
            cursor.seek(SeekFrom::Current(2))?;
        }
        5..=7 => {
            cursor.seek(SeekFrom::Current(4))?;
        }
        8..=10 => {
            cursor.seek(SeekFrom::Current(8))?;
        }
        11 => {
            let mut buf8 = [0u8; 8];
            cursor.read_exact(&mut buf8)?;
            let len = u64::from_le_bytes(buf8);
            cursor.seek(SeekFrom::Current(len as i64))?;
        }
        12 => {
            let mut buf4 = [0u8; 4];
            let mut buf8 = [0u8; 8];
            cursor.read_exact(&mut buf4)?;
            cursor.read_exact(&mut buf8)?;
            let arr_type = u32::from_le_bytes(buf4);
            let arr_len = u64::from_le_bytes(buf8);
            match arr_type {
                0..=2 => {
                    cursor.seek(SeekFrom::Current(arr_len as i64))?;
                }
                3..=4 => {
                    cursor.seek(SeekFrom::Current((arr_len * 2) as i64))?;
                }
                5..=7 => {
                    cursor.seek(SeekFrom::Current((arr_len * 4) as i64))?;
                }
                8..=10 => {
                    cursor.seek(SeekFrom::Current((arr_len * 8) as i64))?;
                }
                11 => {
                    for _ in 0..arr_len {
                        let mut slen_b = [0u8; 8];
                        cursor.read_exact(&mut slen_b)?;
                        let slen = u64::from_le_bytes(slen_b);
                        cursor.seek(SeekFrom::Current(slen as i64))?;
                    }
                }
                _ => return Err("Unsupported nested array".into()),
            }
        }
        _ => return Err(format!("Unsupported type: {}", val_type).into()),
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/data/data/com.termux/files/home/luminous-ecosystem/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    let n_kv = u64::from_le_bytes(mmap[16..24].try_into()?);
    let mut cursor = Cursor::new(&mmap[24..]);

    println!("[+] Searching GGUF metadata for tokenizer vocab...");

    for _ in 0..n_kv {
        let key = read_string(&mut cursor)?;
        let mut vtype_b = [0u8; 4];
        cursor.read_exact(&mut vtype_b)?;
        let val_type = u32::from_le_bytes(vtype_b);

        if key == "tokenizer.ggml.tokens" {
            let mut atype_b = [0u8; 4];
            let mut alen_b = [0u8; 8];
            cursor.read_exact(&mut atype_b)?;
            cursor.read_exact(&mut alen_b)?;
            let count = u64::from_le_bytes(alen_b);
            println!("[+] Found tokenizer vocabulary with {} tokens!\n", count);

            for i in 0..count {
                let token_str = read_string(&mut cursor)?;
                if [0, 625, 9707, 36952].contains(&i) {
                    println!("  -> Token ID {:>5} : {:?}", i, token_str);
                }
            }
            return Ok(());
        } else {
            skip_val(&mut cursor, val_type)?;
        }
    }

    Ok(())
}
