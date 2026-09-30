use crate::ast::{BinaryOp, Expr, Stmt};
use crate::bytecode::{Chunk, OpCode, Reg};
use crate::runtime::Value;
use std::collections::HashMap;

pub struct BytecodeCompiler {
    chunk: Chunk,
    var_map: HashMap<String, Reg>,
    func_map: HashMap<String, (usize, Vec<String>)>,
    next_reg: Reg,
}

impl BytecodeCompiler {
    pub fn new() -> Self {
        BytecodeCompiler {
            chunk: Chunk::new(),
            var_map: HashMap::new(),
            func_map: HashMap::new(),
            next_reg: 0,
        }
    }

    fn alloc_reg(&mut self) -> Result<Reg, String> {
        if self.next_reg == 255 {
            return Err("LUM-VM01: Register allocation overflow (>255 virtual registers)".into());
        }
        let r = self.next_reg;
        self.next_reg += 1;
        Ok(r)
    }

    pub fn compile(mut self, stmts: Vec<Stmt>) -> Result<Chunk, String> {
        for stmt in stmts {
            self.compile_stmt(stmt)?;
        }
        self.chunk.emit(OpCode::Halt);
        Ok(self.chunk)
    }

    fn compile_stmt(&mut self, stmt: Stmt) -> Result<(), String> {
        match stmt {
            Stmt::FnDef { name, params, body } => {
                let jmp_over_idx = self.chunk.emit(OpCode::Jmp { offset: 0 });
                let func_entry = self.chunk.code.len();
                self.func_map.insert(name.clone(), (func_entry, params.clone()));

                // Save outer variables scope
                let outer_vars = self.var_map.clone();
                let outer_next_reg = self.next_reg;

                // Parameter registers
                self.next_reg = 0;
                for p in &params {
                    let r = self.alloc_reg()?;
                    self.var_map.insert(p.clone(), r);
                }

                self.compile_stmt(*body)?;

                // Default return if not explicit
                let dummy_zero = self.alloc_reg()?;
                let c_idx = self.chunk.add_constant(Value::Int(0));
                self.chunk.emit(OpCode::LoadConst { dst: dummy_zero, const_idx: c_idx });
                self.chunk.emit(OpCode::Ret { src: dummy_zero });

                let func_end = self.chunk.code.len();
                self.patch_jump(jmp_over_idx, func_end);

                // Restore outer scope
                self.var_map = outer_vars;
                self.next_reg = outer_next_reg;
            }
            Stmt::Return(opt_expr) => {
                let r = if let Some(expr) = opt_expr {
                    self.compile_expr(expr)?
                } else {
                    let r_zero = self.alloc_reg()?;
                    let c_idx = self.chunk.add_constant(Value::Int(0));
                    self.chunk.emit(OpCode::LoadConst { dst: r_zero, const_idx: c_idx });
                    r_zero
                };
                self.chunk.emit(OpCode::Ret { src: r });
            }
            Stmt::Let(name, expr) => {
                let r = self.compile_expr(expr)?;
                self.var_map.insert(name, r);
            }
            Stmt::Assign(name, expr) => {
                let r_val = self.compile_expr(expr)?;
                let &dst = self.var_map.get(&name)
                    .ok_or_else(|| format!("LUM-VM06: Cannot assign to undefined variable '{}'", name))?;
                self.chunk.emit(OpCode::Mov { dst, src: r_val });
            }
            Stmt::Print(expr) => {
                let r = self.compile_expr(expr)?;
                self.chunk.emit(OpCode::Print { src: r });
            }
            Stmt::Block(stmts) => {
                for s in stmts {
                    self.compile_stmt(s)?;
                }
            }
            Stmt::If { cond, then_branch, else_branch } => {
                let r_cond = self.compile_expr(cond)?;
                // Emit placeholder jump if false
                let jmp_false_idx = self.chunk.emit(OpCode::JmpIfFalse { cond: r_cond, offset: 0 });
                
                self.compile_stmt(*then_branch)?;

                if let Some(else_b) = else_branch {
                    let jmp_end_idx = self.chunk.emit(OpCode::Jmp { offset: 0 });
                    // Patch jump if false to start of else
                    let else_start = self.chunk.code.len();
                    self.patch_jump(jmp_false_idx, else_start);

                    self.compile_stmt(*else_b)?;
                    let post_else = self.chunk.code.len();
                    self.patch_jump(jmp_end_idx, post_else);
                } else {
                    let post_then = self.chunk.code.len();
                    self.patch_jump(jmp_false_idx, post_then);
                }
            }
            Stmt::Send(expr) => {
                let r = self.compile_expr(expr)?;
                self.chunk.emit(OpCode::SendMsg { src: r });
            }
            Stmt::Spawn(body) => {
                let spawn_idx = self.chunk.emit(OpCode::SpawnActor { entry_offset: 0, end_offset: 0 });
                let jmp_over_idx = self.chunk.emit(OpCode::Jmp { offset: 0 });
                
                let actor_start = self.chunk.code.len();
                self.compile_stmt(*body)?;
                self.chunk.emit(OpCode::Halt);
                let actor_end = self.chunk.code.len();

                // Patch offsets
                if let OpCode::SpawnActor { entry_offset, end_offset } = &mut self.chunk.code[spawn_idx] {
                    *entry_offset = actor_start;
                    *end_offset = actor_end;
                }
                self.patch_jump(jmp_over_idx, actor_end);
            }
            Stmt::While { cond, body } => {
                let loop_start = self.chunk.code.len();
                let r_cond = self.compile_expr(cond)?;
                let jmp_false_idx = self.chunk.emit(OpCode::JmpIfFalse { cond: r_cond, offset: 0 });

                self.compile_stmt(*body)?;
                // Loop back to start
                self.chunk.emit(OpCode::Jmp { offset: loop_start });

                let loop_end = self.chunk.code.len();
                self.patch_jump(jmp_false_idx, loop_end);
            }
            Stmt::CanvasDef { width, height, .. } => {
                let rw = self.compile_expr(width)?;
                let rh = self.compile_expr(height)?;
                self.chunk.emit(OpCode::CanvasDef { width: rw, height: rh });
            }
            Stmt::RenderPipeline { source, colormap, target_path } => {
                let r_src = self.compile_expr(source)?;
                let path_str = match target_path {
                    Expr::Str(s) => s,
                    _ => return Err("LUM-VM02: Render path must be string constant".into()),
                };
                let path_idx = self.chunk.add_string(path_str);
                let cmap_idx = colormap.map(|c| self.chunk.add_string(c));
                self.chunk.emit(OpCode::Render { src: r_src, path_idx, cmap_idx });
            }
            Stmt::Expr(expr) => {
                let _ = self.compile_expr(expr)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn patch_jump(&mut self, jump_idx: usize, target: usize) {
        match &mut self.chunk.code[jump_idx] {
            OpCode::Jmp { offset } => *offset = target,
            OpCode::JmpIfFalse { offset, .. } => *offset = target,
            _ => panic!("Compiler internal error: Trying to patch non-jump opcode"),
        }
    }

    fn compile_expr(&mut self, expr: Expr) -> Result<Reg, String> {
        match expr {
            Expr::Receive => {
                let dst = self.alloc_reg()?;
                self.chunk.emit(OpCode::RecvMsg { dst });
                Ok(dst)
            }
            Expr::Call { name, args } => {
                let &(func_ip, ref params) = self.func_map.get(&name)
                    .ok_or_else(|| format!("LUM-FN06: Call to undefined function '{}'", name))?;
                if args.len() != params.len() {
                    return Err(format!("LUM-FN07: Function '{}' expects {} arguments, but got {}", name, params.len(), args.len()));
                }
                
                let arg_count = args.len() as u8;
                let arg_start = self.next_reg;
                for arg in args {
                    let _ = self.compile_expr(arg)?;
                }
                let ret_dst = self.alloc_reg()?;
                self.chunk.emit(OpCode::Call { func_ip, ret_dst, arg_start, arg_count });
                Ok(ret_dst)
            }
            Expr::Int(n) => {
                let dst = self.alloc_reg()?;
                let c_idx = self.chunk.add_constant(Value::Int(n));
                self.chunk.emit(OpCode::LoadConst { dst, const_idx: c_idx });
                Ok(dst)
            }
            Expr::Float(f) => {
                let dst = self.alloc_reg()?;
                let c_idx = self.chunk.add_constant(Value::Float(f));
                self.chunk.emit(OpCode::LoadConst { dst, const_idx: c_idx });
                Ok(dst)
            }
            Expr::Str(s) => {
                let dst = self.alloc_reg()?;
                let c_idx = self.chunk.add_constant(Value::Str(s));
                self.chunk.emit(OpCode::LoadConst { dst, const_idx: c_idx });
                Ok(dst)
            }
            Expr::Variable(name) => {
                let &src = self.var_map.get(&name)
                    .ok_or_else(|| format!("LUM-VM03: Undefined variable '{}'", name))?;
                let dst = self.alloc_reg()?;
                self.chunk.emit(OpCode::Mov { dst, src });
                Ok(dst)
            }
            Expr::Binary { left, op, right } => {
                let lhs = self.compile_expr(*left)?;
                let rhs = self.compile_expr(*right)?;
                let dst = self.alloc_reg()?;
                match op {
                    BinaryOp::Add => self.chunk.emit(OpCode::Add { dst, lhs, rhs }),
                    BinaryOp::Sub => self.chunk.emit(OpCode::Sub { dst, lhs, rhs }),
                    BinaryOp::Mul => self.chunk.emit(OpCode::Mul { dst, lhs, rhs }),
                    BinaryOp::Div => self.chunk.emit(OpCode::Div { dst, lhs, rhs }),
                    BinaryOp::Equal => self.chunk.emit(OpCode::Equal { dst, lhs, rhs }),
                    BinaryOp::Less => self.chunk.emit(OpCode::Less { dst, lhs, rhs }),
                    BinaryOp::Greater => self.chunk.emit(OpCode::Greater { dst, lhs, rhs }),
                    BinaryOp::LessEqual => self.chunk.emit(OpCode::LessEqual { dst, lhs, rhs }),
                    BinaryOp::GreaterEqual => self.chunk.emit(OpCode::GreaterEqual { dst, lhs, rhs }),
                    _ => return Err("LUM-VM04: Operator not yet in bytecode ALU".into()),
                };
                Ok(dst)
            }
            Expr::Intensity(inner) => {
                let src = self.compile_expr(*inner)?;
                let dst = self.alloc_reg()?;
                self.chunk.emit(OpCode::Intensity { dst, src });
                Ok(dst)
            }
            Expr::Fdtd2D { steps, size } => {
                let r_steps = self.compile_expr(*steps)?;
                let r_size = self.compile_expr(*size)?;
                let dst = self.alloc_reg()?;
                self.chunk.emit(OpCode::Fdtd2D { dst, steps: r_steps, size: r_size });
                Ok(dst)
            }
            Expr::Propagate2D { field, z, wavelength, dx } => {
                let r_field = self.compile_expr(*field)?;
                let r_z = self.compile_expr(*z)?;
                let r_lambda = self.compile_expr(*wavelength)?;
                let r_dx = self.compile_expr(*dx)?;
                let dst = self.alloc_reg()?;
                self.chunk.emit(OpCode::Propagate2D {
                    dst,
                    field: r_field,
                    z: r_z,
                    lambda_val: r_lambda,
                    dx: r_dx,
                });
                Ok(dst)
            }
            Expr::Dft(inner) => {
                let src = self.compile_expr(*inner)?;
                let dst = self.alloc_reg()?;
                self.chunk.emit(OpCode::Dft { dst, src, inverse: false });
                Ok(dst)
            }
            Expr::Idft(inner) => {
                let src = self.compile_expr(*inner)?;
                let dst = self.alloc_reg()?;
                self.chunk.emit(OpCode::Dft { dst, src, inverse: true });
                Ok(dst)
            }
            _ => Err("LUM-VM05: Expression not yet supported in Bytecode compiler".into()),
        }
    }
}
