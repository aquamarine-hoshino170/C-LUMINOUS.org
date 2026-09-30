use lum_rust::jit_emitter::{LoopSpec, DynamicJit};

#[test]
fn test_dynamic_register_isolation() {
    // Dynamic Case: Counter in reg[7], Accumulator in reg[15]
    let spec = LoopSpec {
        counter_reg: 7,
        limit_reg: 12,
        accumulator_reg: 15,
        step_val: 1,
    };

    let exec_buf = DynamicJit::compile_loop(&spec).expect("Compilation must succeed");
    
    let mut fake_registers: [i64; 32] = [0; 32];
    fake_registers[7] = 1; // Counter starts at 1
    let limit: i32 = 10;   // 1 + 2 + ... + 10 = 55

    unsafe {
        type NativeFn = extern "C" fn(*mut i64, i32) -> i32;
        let func: NativeFn = exec_buf.as_fn();
        let ret = func(fake_registers.as_mut_ptr(), limit);

        assert_eq!(ret, 55, "Native return must be 55");
        assert_eq!(fake_registers[15], 55, "Dynamic accumulator (reg 15) must receive 55");
        assert_eq!(fake_registers[7], 11, "Dynamic counter (reg 7) must be incremented to 11");
        
        // Ensure other registers remained untouched (r3, r4 untouched)
        assert_eq!(fake_registers[3], 0, "Reg 3 must remain untouched");
        assert_eq!(fake_registers[4], 0, "Reg 4 must remain untouched");
    }
}
