use crate::jit::ExecutableBuffer;

/// লুপ সংক্রান্ত মেটাডাটা ও রেজিস্টার ম্যাপিং স্পেসিফিকেশন
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopSpec {
    pub counter_reg: usize,
    pub limit_reg: usize,
    pub accumulator_reg: usize,
    pub step_val: i32,
}

impl LoopSpec {
    #[inline(always)]
    pub fn reg_offset(reg_idx: usize) -> u32 {
        (reg_idx * 8) as u32
    }
}

/// ARMv7 (A32 Mode, Little-Endian) ডায়নামিক ইনস্ট্রাকশন এমিটার
pub struct ArmEmitter {
    code: Vec<u8>,
}

impl ArmEmitter {
    pub fn new() -> Self {
        Self { code: Vec::with_capacity(64) }
    }

    #[inline(always)]
    pub fn emit32(&mut self, inst: u32) {
        self.code.extend_from_slice(&inst.to_le_bytes());
    }

    pub fn current_offset(&self) -> usize {
        self.code.len()
    }

    /// MOV Rd, #imm (Condition: AL 0xE, Op: 0x3A0)
    pub fn mov_imm(&mut self, rd: u32, imm: u8) {
        let inst = 0xE3A0_0000 | ((rd & 0xF) << 12) | (imm as u32);
        self.emit32(inst);
    }

    /// MOV Rd, Rm (Condition: AL 0xE, Op: 0x1A0)
    pub fn mov_reg(&mut self, rd: u32, rm: u32) {
        let inst = 0xE1A0_0000 | ((rd & 0xF) << 12) | (rm & 0xF);
        self.emit32(inst);
    }

    /// ADD Rd, Rn, Rm (Condition: AL 0xE, Op: 0x08)
    pub fn add_reg(&mut self, rd: u32, rn: u32, rm: u32) {
        let inst = 0xE080_0000 | ((rn & 0xF) << 16) | ((rd & 0xF) << 12) | (rm & 0xF);
        self.emit32(inst);
    }

    /// ADD Rd, Rn, #imm (Condition: AL 0xE, Op: 0x28)
    pub fn add_imm(&mut self, rd: u32, rn: u32, imm: u8) {
        let inst = 0xE280_0000 | ((rn & 0xF) << 16) | ((rd & 0xF) << 12) | (imm as u32);
        self.emit32(inst);
    }

    /// CMP Rn, Rm (Condition: AL 0xE, Op: 0x15, Sets flags)
    pub fn cmp_reg(&mut self, rn: u32, rm: u32) {
        let inst = 0xE150_0000 | ((rn & 0xF) << 16) | (rm & 0xF);
        self.emit32(inst);
    }

    /// BGT (Branch if Greater Than, Signed, Condition: GT 0xC)
    /// offset_words: signed 24-bit relative PC offset (PC = current + 8 bytes)
    pub fn bgt(&mut self, offset_words: i32) {
        let inst = 0xCA00_0000 | ((offset_words as u32) & 0x00FF_FFFF);
        self.emit32(inst);
    }

    /// B (Unconditional Branch, Condition: AL 0xE)
    pub fn b(&mut self, offset_words: i32) {
        let inst = 0xEA00_0000 | ((offset_words as u32) & 0x00FF_FFFF);
        self.emit32(inst);
    }

    /// STR Rt, [Rn, #offset] (Store 32-bit register to memory)
    pub fn str_offset(&mut self, rt: u32, rn: u32, offset: u32) {
        assert!(offset <= 4095, "Offset exceeds ARM immediate limit (12-bit)");
        let inst = 0xE580_0000 | ((rn & 0xF) << 16) | ((rt & 0xF) << 12) | (offset & 0xFFF);
        self.emit32(inst);
    }

    /// LDR Rt, [Rn, #offset] (Load 32-bit value into register)
    pub fn ldr_offset(&mut self, rt: u32, rn: u32, offset: u32) {
        assert!(offset <= 4095, "Offset exceeds ARM immediate limit (12-bit)");
        let inst = 0xE590_0000 | ((rn & 0xF) << 16) | ((rt & 0xF) << 12) | (offset & 0xFFF);
        self.emit32(inst);
    }

    /// BX LR (Return from Subroutine: 0xE12FFF1E)
    pub fn bx_lr(&mut self) {
        self.emit32(0xE12F_FF1E);
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.code
    }
}

pub struct DynamicJit;

impl DynamicJit {
    /// LoopSpec অনুযায়ী ডায়নামিক ARMv7 মেশিন কোড কম্পাইল করা
    /// ABI Contract:
    ///   r0 = pointer to SVM registers (*mut i64)
    ///   r1 = loop limit value
    /// Return:
    ///   r0 = final accumulator value
    pub fn compile_loop(spec: &LoopSpec) -> Result<ExecutableBuffer, String> {
        let mut emitter = ArmEmitter::new();

        // 1. Initial State Setup
        // r2 = accumulator (0 দিয়ে শুরু)
        emitter.mov_imm(2, 0);

        // r3 = loop counter (SVM মেমোরি থেকে ডায়নামিক অফসেটে লোড করা)
        let counter_offset = LoopSpec::reg_offset(spec.counter_reg);
        emitter.ldr_offset(3, 0, counter_offset);

        // 2. Loop Header
        let _loop_start_offset = emitter.current_offset();

        // cmp counter(r3), limit(r1)
        emitter.cmp_reg(3, 1);

        // bgt exit (+2 words / +8 bytes ahead from PC)
        emitter.bgt(2);

        // 3. Loop Body
        // accum (r2) += counter (r3)
        emitter.add_reg(2, 2, 3);

        // counter (r3) += step
        emitter.add_imm(3, 3, spec.step_val as u8);

        // b loop_start (relative to PC+8: 24 bytes backward = -6 words)
        emitter.b(-6);

        // 4. Loop Exit
        // ডায়নামিক রেজিস্টার সিঙ্ক: মেমোরিতে নির্দিষ্ট রেজিস্টারগুলো আপডেট করা
        let accum_offset = LoopSpec::reg_offset(spec.accumulator_reg);
        emitter.str_offset(2, 0, accum_offset);
        emitter.str_offset(3, 0, counter_offset);

        // রিসিভারকে রিটার্ন ভ্যালু দেওয়া: r0 = r2
        emitter.mov_reg(0, 2);

        // রিটার্ন: bx lr
        emitter.bx_lr();

        let machine_code = emitter.into_bytes();
        ExecutableBuffer::new(&machine_code)
    }
}

use crate::bytecode::{OpCode, Chunk};

pub struct LoopInspector;

impl LoopInspector {
    /// লুপের বাইটকোড উইন্ডো স্ক্যান করে সাপোর্টেড প্যাটার্ন ডিটেক্ট করা
    pub fn inspect(chunk: &Chunk, loop_start: usize, loop_end: usize) -> Option<LoopSpec> {
        if loop_start >= loop_end || loop_end >= chunk.code.len() {
            return None;
        }

        let slice = &chunk.code[loop_start..=loop_end];
        if slice.len() < 3 {
            return None;
        }

        let mut counter_reg: Option<usize> = None;
        let mut limit_reg: Option<usize> = None;
        let mut accumulator_reg: Option<usize> = None;
        let mut cond_reg: Option<usize> = None;
        let mut step_val: i32 = 1;

        // ১. হেডার অ্যানালাইসিস: কম্প্যারিসন এবং জাম্প
        for op in slice {
            match op {
                OpCode::Less { dst, lhs, rhs } | OpCode::LessEqual { dst, lhs, rhs } => {
                    cond_reg = Some(*dst as usize);
                    counter_reg = Some(*lhs as usize);
                    limit_reg = Some(*rhs as usize);
                }
                OpCode::JmpIfFalse { cond, offset: _ } => {
                    if cond_reg != Some(*cond as usize) {
                        return None; // কন্ডিশন ভ্যারিয়েবল মিসম্যাচ
                    }
                }
                OpCode::Add { dst, lhs, rhs } => {
                    let d = *dst as usize;
                    let l = *lhs as usize;
                    let r = *rhs as usize;

                    // কাউন্টার ইনক্রিমেন্ট: counter = counter + 1
                    if Some(d) == counter_reg && (Some(l) == counter_reg || Some(r) == counter_reg) {
                        step_val = 1;
                    } 
                    // অ্যাকুমুলেটর আপডেট: accum = accum + counter
                    else if Some(l) == counter_reg || Some(r) == counter_reg {
                        accumulator_reg = Some(d);
                    }
                }
                OpCode::Jmp { .. } => {
                    // লুপ ক্লোজিং জাম্প
                }
                // সাপোর্টেড প্যাটার্নের বাইরে কোনো অপকোড পেলে ফলব্যাক
                _ => return None,
            }
        }

        if let (Some(c), Some(l), Some(a)) = (counter_reg, limit_reg, accumulator_reg) {
            Some(LoopSpec {
                counter_reg: c,
                limit_reg: l,
                accumulator_reg: a,
                step_val,
            })
        } else {
            None
        }
    }
}
