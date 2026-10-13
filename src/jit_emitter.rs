use crate::bytecode::{Chunk, OpCode};
use crate::runtime::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepSpec {
    Constant(i64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSpec {
    Constant(i64),
    Register(u8),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopSpec {
    pub loop_start_ip: usize,
    pub loop_end_ip: usize,
    pub counter_reg: u8,
    pub accum_reg: u8,
    pub limit: LimitSpec,
    pub step: StepSpec,
}

pub struct LoopInspector;

impl LoopInspector {
    /// vm.rs-এর সাথে সামঞ্জস্য রেখে inspect ফাংশন (loop_start এবং loop_end_jmp সহ)
    pub fn inspect(chunk: &Chunk, loop_start: usize, loop_end_jmp: usize) -> Option<LoopSpec> {
        let code = &chunk.code;

        if loop_start >= code.len() || loop_end_jmp >= code.len() || loop_start > loop_end_jmp {
            return None;
        }

        let loop_slice = &code[loop_start..=loop_end_jmp];

        // ১. Header আলাদা করে validate করা হবে।
        // Body-তে emitter-supported opcode ছাড়া কিছুই native JIT-এ যাবে না.
        //
        // NOTE:
        // LessEqual/JmpIfFalse header-এর বৈধ অংশ, তাই পুরো loop_slice-এ
        // strict whitelist প্রয়োগ করা যাবে না।
        //
        // Body whitelist:
        //   Mov
        //   LoadConst
        //   Add
        //
        // এর বাইরে যেকোনো opcode => interpreter fallback.

        // ২. হেডার স্ট্রাকচার ভেরিফিকেশন (MOV, LOAD_CONST, LESS_EQUAL, JMP_IF_FALSE)
        if loop_slice.len() < 5 {
            return None;
        }

        let (counter_reg, limit_spec, loop_end_ip) = match (
            &loop_slice[0],
            &loop_slice[1],
            &loop_slice[2],
            &loop_slice[3],
        ) {
            (
                OpCode::Mov {
                    dst: r_temp,
                    src: r_counter,
                },
                OpCode::LoadConst {
                    dst: r_lim,
                    const_idx,
                },
                OpCode::LessEqual {
                    dst: r_cond,
                    lhs,
                    rhs,
                },
                OpCode::JmpIfFalse { cond, offset },
            ) => {
                if r_cond != cond || lhs != r_temp || rhs != r_lim {
                    return None;
                }
                let limit_val = match chunk.constants.get(*const_idx) {
                    Some(Value::Int(v)) => *v,
                    _ => return None,
                };
                (*r_counter, LimitSpec::Constant(limit_val), *offset)
            }
            _ => return None,
        };

        // ৩. বডি অ্যানালাইসিস: accumulator ও counter আপডেট
        let mut recognized_accum = None;
        let mut recognized_step = None;

        let body_slice = &loop_slice[4..loop_slice.len() - 1];

        // Strict body opcode whitelist.
        // Unknown/future opcodes must NEVER be silently skipped.
        for op in body_slice {
            match op {
                OpCode::Mov { .. } | OpCode::LoadConst { .. } | OpCode::Add { .. } => {}
                _ => return None,
            }
        }

        let mut i = 0;
        while i < body_slice.len() {
            // accum = accum + counter
            if i + 3 < body_slice.len() {
                if let (
                    OpCode::Mov {
                        dst: ra,
                        src: s_accum,
                    },
                    OpCode::Mov {
                        dst: rc,
                        src: s_cnt,
                    },
                    OpCode::Add {
                        dst: r_res,
                        lhs,
                        rhs,
                    },
                    OpCode::Mov {
                        dst: final_accum,
                        src: res_src,
                    },
                ) = (
                    &body_slice[i],
                    &body_slice[i + 1],
                    &body_slice[i + 2],
                    &body_slice[i + 3],
                ) {
                    if lhs == ra
                        && rhs == rc
                        && res_src == r_res
                        && final_accum == s_accum
                        && *s_cnt == counter_reg
                    {
                        recognized_accum = Some(*final_accum);
                        i += 4;
                        continue;
                    }
                }
            }

            // counter = counter + step
            if i + 3 < body_slice.len() {
                if let (
                    OpCode::Mov {
                        dst: rc,
                        src: s_cnt,
                    },
                    OpCode::LoadConst {
                        dst: r_step,
                        const_idx,
                    },
                    OpCode::Add {
                        dst: r_res,
                        lhs,
                        rhs,
                    },
                    OpCode::Mov {
                        dst: final_cnt,
                        src: res_src,
                    },
                ) = (
                    &body_slice[i],
                    &body_slice[i + 1],
                    &body_slice[i + 2],
                    &body_slice[i + 3],
                ) {
                    if lhs == rc
                        && rhs == r_step
                        && res_src == r_res
                        && final_cnt == s_cnt
                        && *s_cnt == counter_reg
                    {
                        let step_val = match chunk.constants.get(*const_idx) {
                            Some(Value::Int(v)) => *v,
                            _ => return None,
                        };
                        recognized_step = Some(StepSpec::Constant(step_val));
                        i += 4;
                        continue;
                    }
                }
            }

            // An allowed opcode that is not consumed by a recognized
            // accumulator/counter pattern must invalidate native JIT.
            return None;
        }

        let accum_reg = recognized_accum?;
        let step = recognized_step?;

        Some(LoopSpec {
            loop_start_ip: loop_start,
            loop_end_ip,
            counter_reg,
            accum_reg,
            limit: limit_spec,
            step,
        })
    }

    /// পুরো চাঙ্ক স্ক্যান করে স্বয়ংক্রিয়ভাবে ব্যাকওয়ার্ড জাম্প ডিটেক্ট করে inspect করা (টেস্টের সুবিধার জন্য)
    pub fn inspect_chunk(chunk: &Chunk) -> Option<LoopSpec> {
        for (idx, op) in chunk.code.iter().enumerate() {
            if let OpCode::Jmp { offset } = op {
                if *offset < idx {
                    return Self::inspect(chunk, *offset, idx);
                }
            }
        }
        None
    }
}

pub struct DynamicJit;

impl DynamicJit {
    fn encode_movw(rd: u8, imm16: u16) -> u32 {
        let imm4 = ((imm16 >> 12) & 0x0F) as u32;
        let imm12 = (imm16 & 0x0FFF) as u32;

        0xE300_0000 | (imm4 << 16) | ((rd as u32) << 12) | imm12
    }

    fn encode_movt(rd: u8, imm16: u16) -> u32 {
        let imm4 = ((imm16 >> 12) & 0x0F) as u32;
        let imm12 = (imm16 & 0x0FFF) as u32;

        0xE340_0000 | (imm4 << 16) | ((rd as u32) << 12) | imm12
    }

    pub fn compile_loop(spec: &LoopSpec) -> Result<crate::jit::ExecutableBuffer, String> {
        let limit_val = match spec.limit {
            LimitSpec::Constant(v) => v,
            LimitSpec::Register(_) => {
                return Err("Register limits not supported yet".into());
            }
        };

        let StepSpec::Constant(step_val) = spec.step;

        // Current native loop supports positive unit/integer steps.
        if step_val <= 0 {
            return Err("Native ARMv7 loop currently requires step > 0".into());
        }

        if !(0..=0x7FFF_FFFF).contains(&limit_val) {
            return Err("Limit exceeds supported positive 31-bit range".into());
        }

        if step_val > 0x7FFF_FFFF {
            return Err("Step exceeds supported positive 31-bit range".into());
        }

        let mut code: Vec<u32> = Vec::new();

        // ------------------------------------------------------------
        // Prologue
        // push {r4, r5, r6, r7, r8, lr}
        // ------------------------------------------------------------
        code.push(0xE92D_41F0);

        let cnt_off = (spec.counter_reg as u32) * 8;
        let acc_off = (spec.accum_reg as u32) * 8;

        if cnt_off > 2040 || acc_off > 2040 || acc_off + 4 > 2044 {
            return Err("VM register offset exceeds ARMv7 LDR/STR range".into());
        }

        // counter low 32-bit
        code.push(0xE590_4000 | cnt_off);

        // accumulator low 32-bit
        code.push(0xE590_5000 | acc_off);

        // accumulator high 32-bit
        code.push(0xE590_7000 | (acc_off + 4));

        // ------------------------------------------------------------
        // Load limit into r6
        // ------------------------------------------------------------
        let limit_lo = (limit_val & 0xFFFF) as u16;
        let limit_hi = ((limit_val >> 16) & 0xFFFF) as u16;

        code.push(Self::encode_movw(6, limit_lo));

        if limit_hi != 0 {
            code.push(Self::encode_movt(6, limit_hi));
        }

        // ------------------------------------------------------------
        // Load step into r8
        // ------------------------------------------------------------
        let step_lo = (step_val & 0xFFFF) as u16;
        let step_hi = ((step_val >> 16) & 0xFFFF) as u16;

        code.push(Self::encode_movw(8, step_lo));

        if step_hi != 0 {
            code.push(Self::encode_movt(8, step_hi));
        }

        // ------------------------------------------------------------
        // Loop
        // ------------------------------------------------------------
        let loop_head_idx = code.len();

        // cmp r4, r6
        code.push(0xE154_0006);

        // bgt exit
        let bgt_idx = code.len();
        code.push(0xCA00_0000);

        // ------------------------------------------------------------
        // 64-bit accumulator:
        //
        // adds r5, r5, r4
        // adc  r7, r7, #0
        // ------------------------------------------------------------
        code.push(0xE095_5004);
        code.push(0xE2A7_7000);

        // counter += step
        code.push(0xE084_4008);

        // branch back to loop header
        let backward_offset = (loop_head_idx as i32) - (code.len() as i32) - 2;

        if !(-0x0080_0000..=0x007F_FFFF).contains(&backward_offset) {
            return Err("Backward ARM branch offset out of range".into());
        }

        code.push(0xEA00_0000 | ((backward_offset as u32) & 0x00FF_FFFF));

        // ------------------------------------------------------------
        // Exit
        // ------------------------------------------------------------
        let exit_idx = code.len();

        let forward_offset = (exit_idx as i32) - (bgt_idx as i32) - 2;

        if !(0..=0x007F_FFFF).contains(&forward_offset) {
            return Err("Forward ARM branch offset out of range".into());
        }

        code[bgt_idx] = 0xCA00_0000 | ((forward_offset as u32) & 0x00FF_FFFF);

        // ------------------------------------------------------------
        // Authoritative VM memory write-back
        // ------------------------------------------------------------

        // counter low
        code.push(0xE580_4000 | cnt_off);

        // accumulator low
        code.push(0xE580_5000 | acc_off);

        // accumulator high
        code.push(0xE580_7000 | (acc_off + 4));

        // ------------------------------------------------------------
        // Epilogue
        // pop {r4, r5, r6, r7, r8, pc}
        // ------------------------------------------------------------
        code.push(0xE8BD_81F0);

        let mut byte_code = Vec::with_capacity(code.len() * 4);

        for instruction in code {
            byte_code.extend_from_slice(&instruction.to_le_bytes());
        }

        crate::jit::ExecutableBuffer::new(&byte_code)
    }
}
