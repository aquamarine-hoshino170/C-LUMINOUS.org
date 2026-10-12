use lum_rust::jit_engine::{Armv7JitEmitter, ExecutableBuffer};
use std::time::Instant;

fn interpreter_sum(limit: u32) -> u64 {
    let mut acc: u64 = 0;
    let mut i: u32 = 1;
    while i <= limit {
        acc += i as u64;
        i += 1;
    }
    acc
}

#[test]
fn test_live_jit_vs_interpreter_speed() {
    let limit: u32 = 10_000_000;

    println!("\n========================================================");
    println!("       LUMINOUS JIT: INTERPRETER vs BARE-METAL ARMv7");
    println!("========================================================");

    // 1. Run Interpreter
    let t0 = Instant::now();
    let interp_res = interpreter_sum(limit);
    let interp_duration = t0.elapsed();
    println!(" [Interpreter] Result : {}", interp_res);
    println!(" [Interpreter] Time   : {:.2?}", interp_duration);

    // 2. Generate Assembly Opcode & Compile into Executable Page
    let t_jit_compile = Instant::now();
    let raw_arm_code = Armv7JitEmitter::emit_sum_loop();
    let exec_buffer = ExecutableBuffer::new(&raw_arm_code);
    let compile_time = t_jit_compile.elapsed();

    // 3. Direct Hardware Execution via AAPCS
    let t_exec = Instant::now();
    let jit_res = unsafe { exec_buffer.execute_loop(limit) };
    let jit_duration = t_exec.elapsed();

    println!(" -------------------------------------------------------");
    println!(" [JIT Emitter] Opcode Size : {} bytes", raw_arm_code.len());
    println!(" [JIT Emitter] Compile Time: {:.2?}", compile_time);
    println!(" [ARMv7 Native] Result     : {}", jit_res);
    println!(" [ARMv7 Native] Exec Time  : {:.2?}", jit_duration);
    println!(" ========================================================");

    assert_eq!(interp_res, jit_res);
    println!(" [✓] ZERO DATA LOSS — 64-bit Precision Verified!");
}
