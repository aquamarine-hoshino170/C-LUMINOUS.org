use crate::runtime::Value;
use std::fs::File;
use std::io::{Read, Write};

pub type Reg = u8;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum OpCode {
    LoadConst {
        dst: Reg,
        const_idx: usize,
    },
    Mov {
        dst: Reg,
        src: Reg,
    },
    Add {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Sub {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Mul {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Div {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Equal {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Less {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Greater {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    LessEqual {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    GreaterEqual {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Print {
        src: Reg,
    },
    Jmp {
        offset: usize,
    },
    JmpIfFalse {
        cond: Reg,
        offset: usize,
    },
    SpawnActor {
        entry_offset: usize,
        end_offset: usize,
    },
    SendMsg {
        src: Reg,
    },
    RecvMsg {
        dst: Reg,
    },
    Call {
        func_ip: usize,
        ret_dst: Reg,
        arg_start: Reg,
        arg_count: u8,
    },
    Ret {
        src: Reg,
    },
    Intensity {
        dst: Reg,
        src: Reg,
    },
    Fdtd2D {
        dst: Reg,
        steps: Reg,
        size: Reg,
    },
    Propagate2D {
        dst: Reg,
        field: Reg,
        z: Reg,
        lambda_val: Reg,
        dx: Reg,
    },
    Dft {
        dst: Reg,
        src: Reg,
        inverse: bool,
    },
    CanvasDef {
        width: Reg,
        height: Reg,
    },
    Render {
        src: Reg,
        path_idx: usize,
        cmap_idx: Option<usize>,
    },
    Halt,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<OpCode>,
    pub constants: Vec<Value>,
    pub strings: Vec<String>,
}

impl Chunk {
    pub fn new() -> Self {
        Chunk {
            code: Vec::new(),
            constants: Vec::new(),
            strings: Vec::new(),
        }
    }

    pub fn add_constant(&mut self, val: Value) -> usize {
        self.constants.push(val);
        self.constants.len() - 1
    }

    pub fn add_string(&mut self, s: String) -> usize {
        self.strings.push(s);
        self.strings.len() - 1
    }

    pub fn emit(&mut self, op: OpCode) -> usize {
        self.code.push(op);
        self.code.len() - 1
    }

    pub fn disassemble(&self, name: &str) {
        println!("==================================================");
        println!("       ❖ LUMINOUS BYTECODE DISASSEMBLER: {} ❖    ", name);
        println!("==================================================");
        println!("CONSTANTS ({} entries):", self.constants.len());
        for (i, c) in self.constants.iter().enumerate() {
            println!("  [{:04}] {:?}", i, c);
        }
        println!("STRINGS ({} entries):", self.strings.len());
        for (i, s) in self.strings.iter().enumerate() {
            println!("  [{:04}] \"{}\"", i, s);
        }
        println!("--------------------------------------------------");
        println!("OPCODE STREAM ({} instructions):", self.code.len());

        for (ip, op) in self.code.iter().enumerate() {
            print!("  {:04} | ", ip);
            match op {
                OpCode::LoadConst { dst, const_idx } => {
                    println!("LOAD_CONST      r{:<3} const[{}]", dst, const_idx);
                }
                OpCode::Mov { dst, src } => {
                    println!("MOV             r{:<3} r{}", dst, src);
                }
                OpCode::Add { dst, lhs, rhs } => {
                    println!("ADD             r{:<3} r{}, r{}", dst, lhs, rhs);
                }
                OpCode::Sub { dst, lhs, rhs } => {
                    println!("SUB             r{:<3} r{}, r{}", dst, lhs, rhs);
                }
                OpCode::Mul { dst, lhs, rhs } => {
                    println!("MUL             r{:<3} r{}, r{}", dst, lhs, rhs);
                }
                OpCode::Div { dst, lhs, rhs } => {
                    println!("DIV             r{:<3} r{}, r{}", dst, lhs, rhs);
                }
                OpCode::Equal { dst, lhs, rhs } => {
                    println!("EQUAL           r{:<3} r{}, r{}", dst, lhs, rhs);
                }
                OpCode::Less { dst, lhs, rhs } => {
                    println!("LESS_THAN       r{:<3} r{}, r{}", dst, lhs, rhs);
                }
                OpCode::Greater { dst, lhs, rhs } => {
                    println!("GREATER_THAN    r{:<3} r{}, r{}", dst, lhs, rhs);
                }
                OpCode::LessEqual { dst, lhs, rhs } => {
                    println!("LESS_EQUAL      r{:<3} r{}, r{}", dst, lhs, rhs);
                }
                OpCode::GreaterEqual { dst, lhs, rhs } => {
                    println!("GREATER_EQUAL   r{:<3} r{}, r{}", dst, lhs, rhs);
                }
                OpCode::Print { src } => {
                    println!("PRINT           r{}", src);
                }
                OpCode::Jmp { offset } => {
                    println!("JMP             --> [{:04}]", offset);
                }
                OpCode::JmpIfFalse { cond, offset } => {
                    println!("JMP_IF_FALSE    r{:<3} --> [{:04}]", cond, offset);
                }
                OpCode::SpawnActor {
                    entry_offset,
                    end_offset,
                } => {
                    println!(
                        "SPAWN_ACTOR     start:[{:04}] end:[{:04}]",
                        entry_offset, end_offset
                    );
                }
                OpCode::SendMsg { src } => {
                    println!("SEND_MSG        r{}", src);
                }
                OpCode::RecvMsg { dst } => {
                    println!("RECV_MSG        r{:<3} (blocking)", dst);
                }
                OpCode::Call {
                    func_ip,
                    ret_dst,
                    arg_start,
                    arg_count,
                } => {
                    println!(
                        "CALL            addr:[{:04}] ret:r{} args:r{}..r{}",
                        func_ip,
                        ret_dst,
                        arg_start,
                        arg_start + arg_count.saturating_sub(1)
                    );
                }
                OpCode::Ret { src } => {
                    println!("RET             r{}", src);
                }
                OpCode::Intensity { dst, src } => {
                    println!("ALU_INTENSITY   r{:<3} r{}", dst, src);
                }
                OpCode::Fdtd2D { dst, steps, size } => {
                    println!(
                        "ALU_FDTD2D      r{:<3} steps:r{}, size:r{}",
                        dst, steps, size
                    );
                }
                OpCode::Propagate2D {
                    dst,
                    field,
                    z,
                    lambda_val,
                    dx,
                } => {
                    println!(
                        "ALU_ASM_PROP    r{:<3} f:r{}, z:r{}, λ:r{}, dx:r{}",
                        dst, field, z, lambda_val, dx
                    );
                }
                OpCode::Dft { dst, src, inverse } => {
                    println!("ALU_DFT_ND      r{:<3} r{} (inv: {})", dst, src, inverse);
                }
                OpCode::CanvasDef { width, height } => {
                    println!("CANVAS_DEF      w:r{}, h:r{}", width, height);
                }
                OpCode::Render {
                    src,
                    path_idx,
                    cmap_idx,
                } => {
                    println!(
                        "RENDER_PIPELINE r{:<3} path:str[{}], cmap:{:?}",
                        src, path_idx, cmap_idx
                    );
                }
                OpCode::Halt => {
                    println!("HALT");
                }
            }
        }
        println!("==================================================");
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let mut file =
            File::create(path).map_err(|e| format!("LUM-LBC: Cannot create file: {}", e))?;
        file.write_all(b"LUMB ").unwrap();

        let str_count = self.strings.len() as u32;
        file.write_all(&str_count.to_le_bytes()).unwrap();
        for s in &self.strings {
            let bytes = s.as_bytes();
            file.write_all(&(bytes.len() as u32).to_le_bytes()).unwrap();
            file.write_all(bytes).unwrap();
        }

        let const_count = self.constants.len() as u32;
        file.write_all(&const_count.to_le_bytes()).unwrap();
        for c in &self.constants {
            match c {
                Value::Int(n) => {
                    file.write_all(&[1]).unwrap();
                    file.write_all(&n.to_le_bytes()).unwrap();
                }
                Value::Float(f) => {
                    file.write_all(&[2]).unwrap();
                    file.write_all(&f.to_le_bytes()).unwrap();
                }
                Value::Str(s) => {
                    file.write_all(&[3]).unwrap();
                    let b = s.as_bytes();
                    file.write_all(&(b.len() as u32).to_le_bytes()).unwrap();
                    file.write_all(b).unwrap();
                }
                Value::Bool(b) => {
                    file.write_all(&[4]).unwrap();
                    file.write_all(&[if *b { 1 } else { 0 }]).unwrap();
                }
                _ => {
                    file.write_all(&[0]).unwrap();
                }
            }
        }

        let code_len = self.code.len() as u32;
        file.write_all(&code_len.to_le_bytes()).unwrap();
        for op in &self.code {
            match op {
                OpCode::LoadConst { dst, const_idx } => {
                    file.write_all(&[10, *dst]).unwrap();
                    file.write_all(&(*const_idx as u32).to_le_bytes()).unwrap();
                }
                OpCode::Mov { dst, src } => {
                    file.write_all(&[11, *dst, *src]).unwrap();
                }
                OpCode::Add { dst, lhs, rhs } => {
                    file.write_all(&[12, *dst, *lhs, *rhs]).unwrap();
                }
                OpCode::Sub { dst, lhs, rhs } => {
                    file.write_all(&[13, *dst, *lhs, *rhs]).unwrap();
                }
                OpCode::Mul { dst, lhs, rhs } => {
                    file.write_all(&[14, *dst, *lhs, *rhs]).unwrap();
                }
                OpCode::Div { dst, lhs, rhs } => {
                    file.write_all(&[15, *dst, *lhs, *rhs]).unwrap();
                }
                OpCode::Equal { dst, lhs, rhs } => {
                    file.write_all(&[16, *dst, *lhs, *rhs]).unwrap();
                }
                OpCode::Less { dst, lhs, rhs } => {
                    file.write_all(&[17, *dst, *lhs, *rhs]).unwrap();
                }
                OpCode::Greater { dst, lhs, rhs } => {
                    file.write_all(&[18, *dst, *lhs, *rhs]).unwrap();
                }
                OpCode::LessEqual { dst, lhs, rhs } => {
                    file.write_all(&[19, *dst, *lhs, *rhs]).unwrap();
                }
                OpCode::GreaterEqual { dst, lhs, rhs } => {
                    file.write_all(&[20, *dst, *lhs, *rhs]).unwrap();
                }
                OpCode::Print { src } => {
                    file.write_all(&[30, *src]).unwrap();
                }
                OpCode::Jmp { offset } => {
                    file.write_all(&[40]).unwrap();
                    file.write_all(&(*offset as u32).to_le_bytes()).unwrap();
                }
                OpCode::JmpIfFalse { cond, offset } => {
                    file.write_all(&[41, *cond]).unwrap();
                    file.write_all(&(*offset as u32).to_le_bytes()).unwrap();
                }
                OpCode::SpawnActor {
                    entry_offset,
                    end_offset,
                } => {
                    file.write_all(&[70]).unwrap();
                    file.write_all(&(*entry_offset as u32).to_le_bytes())
                        .unwrap();
                    file.write_all(&(*end_offset as u32).to_le_bytes()).unwrap();
                }
                OpCode::SendMsg { src } => {
                    file.write_all(&[71, *src]).unwrap();
                }
                OpCode::RecvMsg { dst } => {
                    file.write_all(&[72, *dst]).unwrap();
                }
                OpCode::Call {
                    func_ip,
                    ret_dst,
                    arg_start,
                    arg_count,
                } => {
                    file.write_all(&[80, *ret_dst, *arg_start, *arg_count])
                        .unwrap();
                    file.write_all(&(*func_ip as u32).to_le_bytes()).unwrap();
                }
                OpCode::Ret { src } => {
                    file.write_all(&[81, *src]).unwrap();
                }
                OpCode::Intensity { dst, src } => {
                    file.write_all(&[50, *dst, *src]).unwrap();
                }
                OpCode::Fdtd2D { dst, steps, size } => {
                    file.write_all(&[51, *dst, *steps, *size]).unwrap();
                }
                OpCode::Propagate2D {
                    dst,
                    field,
                    z,
                    lambda_val,
                    dx,
                } => {
                    file.write_all(&[52, *dst, *field, *z, *lambda_val, *dx])
                        .unwrap();
                }
                OpCode::Dft { dst, src, inverse } => {
                    file.write_all(&[53, *dst, *src, if *inverse { 1 } else { 0 }])
                        .unwrap();
                }
                OpCode::CanvasDef { width, height } => {
                    file.write_all(&[60, *width, *height]).unwrap();
                }
                OpCode::Render {
                    src,
                    path_idx,
                    cmap_idx,
                } => {
                    file.write_all(&[61, *src]).unwrap();
                    file.write_all(&(*path_idx as u32).to_le_bytes()).unwrap();
                    let (has_c, c_val) = match cmap_idx {
                        Some(idx) => (1u8, *idx as u32),
                        None => (0u8, 0u32),
                    };
                    file.write_all(&[has_c]).unwrap();
                    file.write_all(&c_val.to_le_bytes()).unwrap();
                }
                OpCode::Halt => {
                    file.write_all(&[255]).unwrap();
                }
            }
        }
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let mut file =
            File::open(path).map_err(|e| format!("LUM-LBC: Cannot open {}: {}", path, e))?;
        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)
            .map_err(|e| format!("LUM-LBC: Invalid file format: {}", e))?;
        if &magic[0..4] != b"LUMB" {
            return Err("LUM-LBC: Corrupted file signature, expected 'LUMB'".into());
        }

        let mut str_cnt_b = [0u8; 4];
        file.read_exact(&mut str_cnt_b).unwrap();
        let str_count = u32::from_le_bytes(str_cnt_b) as usize;
        let mut strings = Vec::new();
        for _ in 0..str_count {
            let mut len_b = [0u8; 4];
            file.read_exact(&mut len_b).unwrap();
            let len = u32::from_le_bytes(len_b) as usize;
            let mut buf = vec![0u8; len];
            file.read_exact(&mut buf).unwrap();
            strings.push(String::from_utf8_lossy(&buf).to_string());
        }

        let mut const_cnt_b = [0u8; 4];
        file.read_exact(&mut const_cnt_b).unwrap();
        let const_count = u32::from_le_bytes(const_cnt_b) as usize;
        let mut constants = Vec::new();
        for _ in 0..const_count {
            let mut tag = [0u8; 1];
            file.read_exact(&mut tag).unwrap();
            match tag[0] {
                1 => {
                    let mut b = [0u8; 8];
                    file.read_exact(&mut b).unwrap();
                    constants.push(Value::Int(i64::from_le_bytes(b)));
                }
                2 => {
                    let mut b = [0u8; 8];
                    file.read_exact(&mut b).unwrap();
                    constants.push(Value::Float(f64::from_le_bytes(b)));
                }
                3 => {
                    let mut len_b = [0u8; 4];
                    file.read_exact(&mut len_b).unwrap();
                    let len = u32::from_le_bytes(len_b) as usize;
                    let mut buf = vec![0u8; len];
                    file.read_exact(&mut buf).unwrap();
                    constants.push(Value::Str(String::from_utf8_lossy(&buf).to_string()));
                }
                4 => {
                    let mut b = [0u8; 1];
                    file.read_exact(&mut b).unwrap();
                    constants.push(Value::Bool(b[0] == 1));
                }
                _ => constants.push(Value::Int(0)),
            }
        }

        let mut code_cnt_b = [0u8; 4];
        file.read_exact(&mut code_cnt_b).unwrap();
        let code_count = u32::from_le_bytes(code_cnt_b) as usize;
        let mut code = Vec::new();

        for _ in 0..code_count {
            let mut op_tag = [0u8; 1];
            file.read_exact(&mut op_tag).unwrap();
            match op_tag[0] {
                10 => {
                    let mut reg = [0u8; 1];
                    file.read_exact(&mut reg).unwrap();
                    let mut idx_b = [0u8; 4];
                    file.read_exact(&mut idx_b).unwrap();
                    code.push(OpCode::LoadConst {
                        dst: reg[0],
                        const_idx: u32::from_le_bytes(idx_b) as usize,
                    });
                }
                11 => {
                    let mut b = [0u8; 2];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Mov {
                        dst: b[0],
                        src: b[1],
                    });
                }
                12 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Add {
                        dst: b[0],
                        lhs: b[1],
                        rhs: b[2],
                    });
                }
                13 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Sub {
                        dst: b[0],
                        lhs: b[1],
                        rhs: b[2],
                    });
                }
                14 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Mul {
                        dst: b[0],
                        lhs: b[1],
                        rhs: b[2],
                    });
                }
                15 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Div {
                        dst: b[0],
                        lhs: b[1],
                        rhs: b[2],
                    });
                }
                16 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Equal {
                        dst: b[0],
                        lhs: b[1],
                        rhs: b[2],
                    });
                }
                17 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Less {
                        dst: b[0],
                        lhs: b[1],
                        rhs: b[2],
                    });
                }
                18 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Greater {
                        dst: b[0],
                        lhs: b[1],
                        rhs: b[2],
                    });
                }
                19 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::LessEqual {
                        dst: b[0],
                        lhs: b[1],
                        rhs: b[2],
                    });
                }
                20 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::GreaterEqual {
                        dst: b[0],
                        lhs: b[1],
                        rhs: b[2],
                    });
                }
                30 => {
                    let mut reg = [0u8; 1];
                    file.read_exact(&mut reg).unwrap();
                    code.push(OpCode::Print { src: reg[0] });
                }
                40 => {
                    let mut off_b = [0u8; 4];
                    file.read_exact(&mut off_b).unwrap();
                    code.push(OpCode::Jmp {
                        offset: u32::from_le_bytes(off_b) as usize,
                    });
                }
                41 => {
                    let mut reg = [0u8; 1];
                    file.read_exact(&mut reg).unwrap();
                    let mut off_b = [0u8; 4];
                    file.read_exact(&mut off_b).unwrap();
                    code.push(OpCode::JmpIfFalse {
                        cond: reg[0],
                        offset: u32::from_le_bytes(off_b) as usize,
                    });
                }
                50 => {
                    let mut b = [0u8; 2];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Intensity {
                        dst: b[0],
                        src: b[1],
                    });
                }
                51 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Fdtd2D {
                        dst: b[0],
                        steps: b[1],
                        size: b[2],
                    });
                }
                52 => {
                    let mut b = [0u8; 5];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Propagate2D {
                        dst: b[0],
                        field: b[1],
                        z: b[2],
                        lambda_val: b[3],
                        dx: b[4],
                    });
                }
                53 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Dft {
                        dst: b[0],
                        src: b[1],
                        inverse: b[2] == 1,
                    });
                }
                60 => {
                    let mut b = [0u8; 2];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::CanvasDef {
                        width: b[0],
                        height: b[1],
                    });
                }
                61 => {
                    let mut reg = [0u8; 1];
                    file.read_exact(&mut reg).unwrap();
                    let mut p_b = [0u8; 4];
                    file.read_exact(&mut p_b).unwrap();
                    let mut has_c = [0u8; 1];
                    file.read_exact(&mut has_c).unwrap();
                    let mut c_b = [0u8; 4];
                    file.read_exact(&mut c_b).unwrap();
                    let cmap_idx = if has_c[0] == 1 {
                        Some(u32::from_le_bytes(c_b) as usize)
                    } else {
                        None
                    };
                    code.push(OpCode::Render {
                        src: reg[0],
                        path_idx: u32::from_le_bytes(p_b) as usize,
                        cmap_idx,
                    });
                }
                70 => {
                    let mut b1 = [0u8; 4];
                    let mut b2 = [0u8; 4];
                    file.read_exact(&mut b1).unwrap();
                    file.read_exact(&mut b2).unwrap();
                    code.push(OpCode::SpawnActor {
                        entry_offset: u32::from_le_bytes(b1) as usize,
                        end_offset: u32::from_le_bytes(b2) as usize,
                    });
                }
                71 => {
                    let mut b = [0u8; 1];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::SendMsg { src: b[0] });
                }
                72 => {
                    let mut b = [0u8; 1];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::RecvMsg { dst: b[0] });
                }
                80 => {
                    let mut b = [0u8; 3];
                    file.read_exact(&mut b).unwrap();
                    let mut ip_b = [0u8; 4];
                    file.read_exact(&mut ip_b).unwrap();
                    code.push(OpCode::Call {
                        func_ip: u32::from_le_bytes(ip_b) as usize,
                        ret_dst: b[0],
                        arg_start: b[1],
                        arg_count: b[2],
                    });
                }
                81 => {
                    let mut b = [0u8; 1];
                    file.read_exact(&mut b).unwrap();
                    code.push(OpCode::Ret { src: b[0] });
                }
                255 => code.push(OpCode::Halt),
                _ => {}
            }
        }

        Ok(Chunk {
            code,
            constants,
            strings,
        })
    }
}
