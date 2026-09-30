use crate::bytecode::{Chunk, OpCode, Reg};
use crate::jit::ExecutableBuffer;
use crate::runtime::{Complex64, Value};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

pub struct TensorArena {
    pub storage: Vec<f64>,
}

impl TensorArena {
    pub fn new(capacity: usize) -> Self {
        TensorArena {
            storage: Vec::with_capacity(capacity),
        }
    }

    pub fn allocate_grid(&mut self, width: usize, height: usize, fill: f64) -> usize {
        let offset = self.storage.len();
        self.storage.resize(offset + (width * height), fill);
        offset
    }
}

pub struct ExecutionProfiler {
    loop_hit_counts: HashMap<usize, usize>,
    hot_threshold: usize,
    hot_loops_compiled: HashMap<usize, bool>,
}

impl ExecutionProfiler {
    pub fn new(hot_threshold: usize) -> Self {
        ExecutionProfiler {
            loop_hit_counts: HashMap::new(),
            hot_threshold,
            hot_loops_compiled: HashMap::new(),
        }
    }

    pub fn record_jump(&mut self, target_ip: usize) -> bool {
        let count = self.loop_hit_counts.entry(target_ip).or_insert(0);
        *count += 1;
        if *count >= self.hot_threshold && !self.hot_loops_compiled.contains_key(&target_ip) {
            self.hot_loops_compiled.insert(target_ip, true);
            return true;
        }
        false
    }
}

pub struct CallFrame {
    pub return_ip: usize,
    pub ret_dst: Reg,
    pub saved_registers: [Value; 256],
}

pub struct SVM {
    pub registers: [Value; 256],
    pub ip: usize,
    pub canvas_w: usize,
    pub canvas_h: usize,
    pub arena: TensorArena,
    pub profiler: ExecutionProfiler,
    pub mailbox_tx: Sender<Value>,
    pub mailbox_rx: Arc<Mutex<Receiver<Value>>>,
    pub call_stack: Vec<CallFrame>,
    #[allow(dead_code)]
    pub compiled_jits: HashMap<usize, ExecutableBuffer>,
}

impl SVM {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        SVM {
            registers: std::array::from_fn(|_| Value::Int(0)),
            ip: 0,
            canvas_w: 800,
            canvas_h: 800,
            arena: TensorArena::new(1024 * 1024),
            profiler: ExecutionProfiler::new(5),
            mailbox_tx: tx,
            mailbox_rx: Arc::new(Mutex::new(rx)),
            call_stack: Vec::new(),
            compiled_jits: HashMap::new(),
        }
    }

    pub fn execute(&mut self, chunk: &Chunk) -> Result<(), String> {
        let code_len = chunk.code.len();
        let start_time = Instant::now();

        println!("[SVM Core] Initialized with 256 Flat Registers & 1MB Tensor Arena.");
        println!("[SVM Engine] JIT Hot-Spot Profiler Active (Threshold: 5 backward jumps).");

        while self.ip < code_len {
            let op = &chunk.code[self.ip];
            let current_ip = self.ip;
            self.ip += 1;

            let step_result = self.execute_opcode(op, chunk, current_ip);
            if let Err(err_msg) = step_result {
                eprintln!("[SVM Actor Fault-Isolation] Trapped Exception at IP {:04}: {}", current_ip, err_msg);
                eprintln!("[SVM Actor Sandbox] Trapped execution boundary; isolated process safely terminated.");
                return Err(err_msg);
            }
        }

        let elapsed = start_time.elapsed();
        println!("[SVM Pipeline] Total Execution Finished in {:.3} ms.", elapsed.as_secs_f64() * 1000.0);
        Ok(())
    }

    pub fn execute_slice(&mut self, chunk: &Chunk) -> Result<(), String> {
        let code_len = chunk.code.len();
        while self.ip < code_len {
            let op = &chunk.code[self.ip];
            let current_ip = self.ip;
            self.ip += 1;
            if let OpCode::Halt = op { break; }
            self.execute_opcode(op, chunk, current_ip)?;
        }
        Ok(())
    }

    #[inline(always)]
    fn execute_opcode(&mut self, op: &OpCode, chunk: &Chunk, current_ip: usize) -> Result<(), String> {
        match op {
            OpCode::LoadConst { dst, const_idx } => {
                self.registers[*dst as usize] = chunk.constants[*const_idx].clone();
            }
            OpCode::Mov { dst, src } => {
                self.registers[*dst as usize] = self.registers[*src as usize].clone();
            }
            OpCode::Add { dst, lhs, rhs } => {
                let v_lhs = &self.registers[*lhs as usize];
                let v_rhs = &self.registers[*rhs as usize];
                self.registers[*dst as usize] = vm_add(v_lhs, v_rhs)?;
            }
            OpCode::Sub { dst, lhs, rhs } => {
                let v_lhs = &self.registers[*lhs as usize];
                let v_rhs = &self.registers[*rhs as usize];
                self.registers[*dst as usize] = vm_sub(v_lhs, v_rhs)?;
            }
            OpCode::Mul { dst, lhs, rhs } => {
                let v_lhs = &self.registers[*lhs as usize];
                let v_rhs = &self.registers[*rhs as usize];
                self.registers[*dst as usize] = vm_mul(v_lhs, v_rhs)?;
            }
            OpCode::Div { dst, lhs, rhs } => {
                let v_lhs = &self.registers[*lhs as usize];
                let v_rhs = &self.registers[*rhs as usize];
                self.registers[*dst as usize] = vm_div(v_lhs, v_rhs)?;
            }
            OpCode::Equal { dst, lhs, rhs } => {
                let eq = self.registers[*lhs as usize] == self.registers[*rhs as usize];
                self.registers[*dst as usize] = Value::Bool(eq);
            }
            OpCode::Less { dst, lhs, rhs } => {
                let l = vm_to_float(&self.registers[*lhs as usize])?;
                let r = vm_to_float(&self.registers[*rhs as usize])?;
                self.registers[*dst as usize] = Value::Bool(l < r);
            }
            OpCode::Greater { dst, lhs, rhs } => {
                let l = vm_to_float(&self.registers[*lhs as usize])?;
                let r = vm_to_float(&self.registers[*rhs as usize])?;
                self.registers[*dst as usize] = Value::Bool(l > r);
            }
            OpCode::LessEqual { dst, lhs, rhs } => {
                let l = vm_to_float(&self.registers[*lhs as usize])?;
                let r = vm_to_float(&self.registers[*rhs as usize])?;
                self.registers[*dst as usize] = Value::Bool(l <= r);
            }
            OpCode::GreaterEqual { dst, lhs, rhs } => {
                let l = vm_to_float(&self.registers[*lhs as usize])?;
                let r = vm_to_float(&self.registers[*rhs as usize])?;
                self.registers[*dst as usize] = Value::Bool(l >= r);
            }
            OpCode::Jmp { offset } => {
                // Backward Jump Detection for Native JIT
                if *offset <= current_ip {
                    if self.profiler.record_jump(*offset) {
                        println!("[SVM Hot-Spot Profiler] Hot Loop Detected at target IP {:04}!", *offset);
                        if let Some(spec) = crate::jit_emitter::LoopInspector::inspect(chunk, *offset, current_ip) {
                            println!("  ├── [Loop Inspector] Pattern Matched: {:?}", spec);
                            println!("  ├── [JIT Compiler] Compiling Dynamic ARMv7 Machine Code...");
                            match crate::jit_emitter::DynamicJit::compile_loop(&spec) {
                                Ok(exec_buf) => {
                                    println!("  ├── [Executable Memory] Allocated at {:p} (PROT_READ | PROT_EXEC)", exec_buf.ptr);
                                    let mut raw_regs: [i64; 32] = [0; 32];
                                    for i in 0..32 {
                                        if let Value::Int(n) = self.registers[i] {
                                            raw_regs[i] = n;
                                        }
                                    }
                                    let limit_val = if let Value::Int(lim) = self.registers[spec.limit_reg] { lim as i32 } else { 0 };

                                    unsafe {
                                        type NativeFn = extern "C" fn(*mut i64, i32) -> i32;
                                        let jitted_fn: NativeFn = exec_buf.as_fn();
                                        let ret_val = jitted_fn(raw_regs.as_mut_ptr(), limit_val);
                                        println!("  └── [Native ARM CPU] Executed loop to completion. Native return: {}", ret_val);
                                        // Commit explicitly to target accumulator and counter
                                        self.registers[spec.accumulator_reg] = Value::Int(raw_regs[spec.accumulator_reg]);
                                        self.registers[spec.counter_reg] = Value::Int(raw_regs[spec.counter_reg]);
                                    }
                                    // Loop finished natively, skip past the loop back-edge
                                    self.ip = current_ip + 1;
                                    return Ok(());
                                }
                                Err(e) => eprintln!("  └── [JIT Fallback] Compilation failed: {}. Continuing interpreter.", e),
                            }
                        } else {
                            println!("  └── [JIT Fallback] Loop pattern unsupported or non-trivial. Running interpreter.");
                        }
                    }
                }
                self.ip = *offset;
            }
            OpCode::JmpIfFalse { cond, offset } => {
                let is_truthy = match &self.registers[*cond as usize] {
                    Value::Bool(b) => *b,
                    Value::Int(n) => *n != 0,
                    Value::Float(f) => *f != 0.0,
                    _ => false,
                };
                if !is_truthy {
                    self.ip = *offset;
                }
            }
            OpCode::Print { src } => {
                println!("{}", self.registers[*src as usize]);
            }
            OpCode::SpawnActor { entry_offset, end_offset: _ } => {
                let entry = *entry_offset;
                let chunk_clone = chunk.clone();
                let initial_regs = self.registers.clone();
                let worker_tx = self.mailbox_tx.clone();
                let worker_rx = Arc::clone(&self.mailbox_rx);

                println!("[SVM Actor Runtime] Spawning Concurrent Worker Process at IP {:04}...", entry);

                thread::spawn(move || {
                    let mut actor_vm = SVM::new();
                    actor_vm.registers = initial_regs;
                    actor_vm.ip = entry;
                    actor_vm.mailbox_tx = worker_tx;
                    actor_vm.mailbox_rx = worker_rx;
                    let _ = actor_vm.execute_slice(&chunk_clone);
                });
            }
            OpCode::SendMsg { src } => {
                let val = self.registers[*src as usize].clone();
                let _ = self.mailbox_tx.send(val);
            }
            OpCode::RecvMsg { dst } => {
                let rx_lock = self.mailbox_rx.lock().map_err(|e| e.to_string())?;
                let val = rx_lock.recv().map_err(|_| "SVM-Actor: Channel closed during receive".to_string())?;
                self.registers[*dst as usize] = val;
            }
            OpCode::Call { func_ip, ret_dst, arg_start, arg_count } => {
                let frame = CallFrame {
                    return_ip: self.ip,
                    ret_dst: *ret_dst,
                    saved_registers: self.registers.clone(),
                };
                self.call_stack.push(frame);

                let mut new_regs: [Value; 256] = std::array::from_fn(|_| Value::Int(0));
                for i in 0..(*arg_count as usize) {
                    new_regs[i] = self.registers[*arg_start as usize + i].clone();
                }
                self.registers = new_regs;
                self.ip = *func_ip;
            }
            OpCode::Ret { src } => {
                let ret_val = self.registers[*src as usize].clone();
                if let Some(frame) = self.call_stack.pop() {
                    self.registers = frame.saved_registers;
                    self.registers[frame.ret_dst as usize] = ret_val;
                    self.ip = frame.return_ip;
                } else {
                    self.ip = usize::MAX;
                }
            }
            OpCode::Intensity { dst, src } => {
                let val = &self.registers[*src as usize];
                self.registers[*dst as usize] = vm_intensity(val)?;
            }
            OpCode::Fdtd2D { dst, steps, size } => {
                let s = vm_to_int(&self.registers[*steps as usize])? as usize;
                let sz = vm_to_int(&self.registers[*size as usize])? as usize;
                self.registers[*dst as usize] = self.vm_fdtd2d_arena(s, sz)?;
            }
            OpCode::Propagate2D { dst, field, z, lambda_val, dx } => {
                let f = self.registers[*field as usize].clone();
                let z_v = vm_to_float(&self.registers[*z as usize])?;
                let l_v = vm_to_float(&self.registers[*lambda_val as usize])?;
                let dx_v = vm_to_float(&self.registers[*dx as usize])?;
                self.registers[*dst as usize] = vm_asm_propagation(f, z_v, l_v, dx_v)?;
            }
            OpCode::Dft { dst, src, inverse } => {
                let val = self.registers[*src as usize].clone();
                self.registers[*dst as usize] = vm_dft_nd(val, *inverse)?;
            }
            OpCode::CanvasDef { width, height } => {
                self.canvas_w = vm_to_int(&self.registers[*width as usize])? as usize;
                self.canvas_h = vm_to_int(&self.registers[*height as usize])? as usize;
            }
            OpCode::Render { src, path_idx, cmap_idx } => {
                let val = &self.registers[*src as usize];
                let path = &chunk.strings[*path_idx];
                let cmap = cmap_idx.map(|idx| chunk.strings[idx].as_str()).unwrap_or("viridis");
                vm_render_surface(val, cmap, path, self.canvas_w, self.canvas_h)?;
                println!("[✓ SVM] Bytecode Pipeline Rendered Surface: '{}' (Colormap: {})", path, cmap);
            }
            OpCode::Halt => {
                self.ip = usize::MAX;
            }
        }
        Ok(())
    }

    fn vm_fdtd2d_arena(&mut self, steps: usize, size: usize) -> Result<Value, String> {
        if size < 3 { return Err("SVM-ALU: FDTD grid dimension must be >= 3".into()); }
        let ez_offset = self.arena.allocate_grid(size, size, 0.0);
        let hx_offset = self.arena.allocate_grid(size, size, 0.0);
        let hy_offset = self.arena.allocate_grid(size, size, 0.0);

        let sc = 0.7;
        let center = size / 2;

        for t in 0..steps {
            for i in 0..size - 1 {
                for j in 0..size - 1 {
                    let ez_curr = self.arena.storage[ez_offset + i * size + j];
                    let ez_j1 = self.arena.storage[ez_offset + i * size + (j + 1)];
                    let ez_i1 = self.arena.storage[ez_offset + (i + 1) * size + j];

                    self.arena.storage[hx_offset + i * size + j] -= sc * (ez_j1 - ez_curr);
                    self.arena.storage[hy_offset + i * size + j] += sc * (ez_i1 - ez_curr);
                }
            }
            for i in 1..size {
                for j in 1..size {
                    let hy_curr = self.arena.storage[hy_offset + i * size + j];
                    let hy_prev = self.arena.storage[hy_offset + (i - 1) * size + j];
                    let hx_curr = self.arena.storage[hx_offset + i * size + j];
                    let hx_prev = self.arena.storage[hx_offset + i * size + (j - 1)];

                    self.arena.storage[ez_offset + i * size + j] += sc * ((hy_curr - hy_prev) - (hx_curr - hx_prev));
                }
            }
            let pulse = (-(((t as f64 - 10.0) / 4.0).powi(2))).exp();
            self.arena.storage[ez_offset + center * size + center] += pulse;
        }

        let mut result = Vec::with_capacity(size);
        for i in 0..size {
            let mut row = Vec::with_capacity(size);
            for j in 0..size {
                let v = self.arena.storage[ez_offset + i * size + j];
                row.push(Value::Complex(Complex64::new(v, 0.0)));
            }
            result.push(Value::Array(row));
        }

        Ok(Value::Array(result))
    }
}

fn vm_to_int(v: &Value) -> Result<i64, String> {
    match v {
        Value::Int(n) => Ok(*n),
        Value::Float(f) => Ok(*f as i64),
        _ => Err("SVM: Expected integer register operand".into()),
    }
}

fn vm_to_float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        _ => Err("SVM: Expected numeric float operand".into()),
    }
}

fn vm_add(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x + y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x + y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 + y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x + *y as f64)),
        _ => Err("SVM: Add operand type mismatch".into()),
    }
}

fn vm_sub(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x - y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x - y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 - y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x - *y as f64)),
        _ => Err("SVM: Sub operand type mismatch".into()),
    }
}

fn vm_mul(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x * y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x * y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 * y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x * *y as f64)),
        _ => Err("SVM: Mul operand type mismatch".into()),
    }
}

fn vm_div(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            if *y == 0 { Err("SVM: Division by zero".into()) } else { Ok(Value::Int(x / y)) }
        }
        (Value::Float(x), Value::Float(y)) => {
            if *y == 0.0 { Err("SVM: Division by zero".into()) } else { Ok(Value::Float(x / y)) }
        }
        (Value::Int(x), Value::Float(y)) => {
            if *y == 0.0 { Err("SVM: Division by zero".into()) } else { Ok(Value::Float((*x as f64) / y)) }
        }
        (Value::Float(x), Value::Int(y)) => {
            if *y == 0 { Err("SVM: Division by zero".into()) } else { Ok(Value::Float(x / (*y as f64))) }
        }
        _ => Err("SVM: Div operand type mismatch".into()),
    }
}

fn vm_intensity(val: &Value) -> Result<Value, String> {
    match val {
        Value::Complex(c) => Ok(Value::Float(c.re * c.re + c.im * c.im)),
        Value::Float(f) => Ok(Value::Float(f * f)),
        Value::Int(n) => Ok(Value::Float((n * n) as f64)),
        Value::Array(arr) => {
            let mut res = Vec::with_capacity(arr.len());
            for it in arr {
                res.push(vm_intensity(it)?);
            }
            Ok(Value::Array(res))
        }
        _ => Err("SVM: Intensity operation invalid for data type".into()),
    }
}

fn dft_1d(inputs: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = inputs.len();
    let mut output = Vec::with_capacity(n);
    let sign = if inverse { 1.0 } else { -1.0 };
    let scale = if inverse { n as f64 } else { 1.0 };

    for k in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for (j, &xj) in inputs.iter().enumerate() {
            let angle = sign * 2.0 * PI * (k as f64) * (j as f64) / (n as f64);
            let exp = Complex64::new(angle.cos(), angle.sin());
            sum = sum.add(xj.mul(exp));
        }
        output.push(Complex64::new(sum.re / scale, sum.im / scale));
    }
    output
}

fn vm_dft_nd(val: Value, inverse: bool) -> Result<Value, String> {
    match val {
        Value::Array(arr) => {
            if arr.is_empty() { return Ok(Value::Array(Vec::new())); }
            let rows = arr.len();
            let mut mat: Vec<Vec<Complex64>> = Vec::with_capacity(rows);
            for r in arr {
                match r {
                    Value::Array(cols) => {
                        let mut row_c = Vec::with_capacity(cols.len());
                        for item in cols {
                            let c = match item {
                                Value::Complex(c) => c,
                                Value::Float(f) => Complex64::new(f, 0.0),
                                Value::Int(n) => Complex64::new(n as f64, 0.0),
                                _ => return Err("SVM-DFT: Matrix cells must be numeric".into()),
                            };
                            row_c.push(c);
                        }
                        mat.push(row_c);
                    }
                    _ => return Err("SVM-DFT: Input tensor must be 2D array".into()),
                }
            }
            let cols = mat[0].len();
            for i in 0..rows { mat[i] = dft_1d(&mat[i], inverse); }
            for j in 0..cols {
                let mut col = Vec::with_capacity(rows);
                for i in 0..rows { col.push(mat[i][j]); }
                let trans = dft_1d(&col, inverse);
                for i in 0..rows { mat[i][j] = trans[i]; }
            }
            Ok(Value::Array(mat.into_iter().map(|r| Value::Array(r.into_iter().map(Value::Complex).collect())).collect()))
        }
        _ => Err("SVM-DFT: Input required as tensor array".into()),
    }
}

fn vm_asm_propagation(field_val: Value, z: f64, wavelength: f64, dx: f64) -> Result<Value, String> {
    let freq_field = vm_dft_nd(field_val, false)?;
    let k0 = 2.0 * PI / wavelength;
    let mat = match freq_field {
        Value::Array(rows) => rows,
        _ => return Err("SVM-ASM: Invalid angular spectrum grid".into()),
    };
    let n_rows = mat.len();
    let n_cols = match &mat[0] {
        Value::Array(c) => c.len(),
        _ => return Err("SVM-ASM: Angular spectrum tensor must be 2D".into()),
    };

    let mut modified = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let mut row_mod = Vec::with_capacity(n_cols);
        let kx = 2.0 * PI * (i as f64 - n_rows as f64 / 2.0) / (n_rows as f64 * dx);
        if let Value::Array(cols) = &mat[i] {
            for j in 0..n_cols {
                let ky = 2.0 * PI * (j as f64 - n_cols as f64 / 2.0) / (n_cols as f64 * dx);
                let val = match &cols[j] {
                    Value::Complex(c) => *c,
                    _ => Complex64::new(0.0, 0.0),
                };
                let k_sq = kx * kx + ky * ky;
                let h = if k0 * k0 >= k_sq {
                    let kz = (k0 * k0 - k_sq).sqrt();
                    Complex64::new((kz * z).cos(), (kz * z).sin())
                } else {
                    let kz = (k_sq - k0 * k0).sqrt();
                    Complex64::new((-kz * z).exp(), 0.0)
                };
                row_mod.push(Value::Complex(val.mul(h)));
            }
        }
        modified.push(Value::Array(row_mod));
    }
    vm_dft_nd(Value::Array(modified), true)
}

fn vm_render_surface(val: &Value, cmap: &str, filename: &str, _cw: usize, _ch: usize) -> Result<(), String> {
    let rows = match val {
        Value::Array(r) => r,
        _ => return Err("SVM-Render: Render target must be 2D array".into()),
    };
    if rows.is_empty() { return Ok(()); }
    let h = rows.len();
    let mut num_grid: Vec<Vec<f64>> = Vec::with_capacity(h);
    let mut max_val: f64 = 1e-9;
    let mut min_val: f64 = f64::MAX;

    for r in rows {
        if let Value::Array(cols) = r {
            let mut row_n = Vec::with_capacity(cols.len());
            for c in cols {
                let n: f64 = match c {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    Value::Complex(comp) => (comp.re * comp.re + comp.im * comp.im).sqrt(),
                    _ => 0.0,
                };
                if n > max_val { max_val = n; }
                if n < min_val { min_val = n; }
                row_n.push(n);
            }
            num_grid.push(row_n);
        }
    }
    let w = num_grid[0].len();
    let range: f64 = (max_val - min_val).max(1e-9);

    let mut file = File::create(filename).map_err(|e| format!("SVM-IO: {}", e))?;
    writeln!(file, "P3").unwrap();
    writeln!(file, "{} {}", w, h).unwrap();
    writeln!(file, "255").unwrap();

    for y in 0..h {
        let mut line = String::new();
        for x in 0..w {
            let t: f64 = ((num_grid[y][x] - min_val) / range).clamp(0.0, 1.0);
            let (r, g, b) = match cmap {
                "viridis" => {
                    let r = (0.28 + 0.70 * t).clamp(0.0, 1.0);
                    let g = (0.01 + 1.25 * t - 0.35 * t * t).clamp(0.0, 1.0);
                    let b = (0.33 + 0.85 * (1.0 - t).powi(2)).clamp(0.0, 1.0);
                    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
                }
                _ => {
                    let r = (1.5 - (t * 4.0 - 3.0).abs()).clamp(0.0, 1.0);
                    let g = (1.5 - (t * 4.0 - 2.0).abs()).clamp(0.0, 1.0);
                    let b = (1.5 - (t * 4.0 - 1.0).abs()).clamp(0.0, 1.0);
                    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
                }
            };
            line.push_str(&format!("{} {} {} ", r, g, b));
        }
        writeln!(file, "{}", line).unwrap();
    }
    Ok(())
}
