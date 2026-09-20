use crate::ast::{BinaryOp, Expr, Stmt};
use crate::lexer;
use crate::parser::Parser;
use crate::runtime::{Complex64, Environment, HornClause, KnowledgeBase, Value};
use std::collections::HashSet;
use std::f64::consts::PI;
use std::fs;

pub fn execute_program(stmts: Vec<Stmt>, env: &mut Environment) -> Result<(), String> {
    for stmt in stmts {
        execute_stmt(&stmt, env)?;
    }
    Ok(())
}

fn execute_stmt(stmt: &Stmt, env: &mut Environment) -> Result<(), String> {
    match stmt {
        Stmt::Let(name, expr) => {
            let val = eval_expr(expr, env)?;
            env.set(name.clone(), val);
        }
        Stmt::Print(expr) => {
            let val = eval_expr(expr, env)?;
            println!("{}", val);
        }
        Stmt::Import(module_path) => {
            let safe_path = env.resolve_module_path(module_path)?;
            let src = fs::read_to_string(&safe_path).map_err(|e| {
                format!("LUM-IO01: Failed to read module {}: {}", safe_path.display(), e)
            })?;
            let tokens = lexer::tokenize(&src);
            let mut parser = Parser::new(tokens);
            let ast = parser.parse()?;
            execute_program(ast, env)?;
        }
        Stmt::Expr(expr) => {
            eval_expr(expr, env)?;
        }
    }
    Ok(())
}

fn eval_expr(expr: &Expr, env: &mut Environment) -> Result<Value, String> {
    match expr {
        Expr::Int(n) => Ok(Value::Int(*n)),
        Expr::Float(f) => Ok(Value::Float(*f)),
        Expr::Imaginary(im) => Ok(Value::Complex(Complex64::new(0.0, *im))),
        Expr::Str(s) => Ok(Value::Str(s.clone())),
        Expr::Variable(name) => env
            .get(name)
            .ok_or_else(|| format!("LUM-R4001: Undefined variable '{}'", name)),
        Expr::Array(elements) => {
            let mut vals = Vec::new();
            for el in elements {
                vals.push(eval_expr(el, env)?);
            }
            Ok(Value::Array(vals))
        }
        Expr::Conj(inner) => {
            let val = eval_expr(inner, env)?;
            eval_conj(val)
        }
        Expr::Intensity(inner) => {
            let val = eval_expr(inner, env)?;
            eval_intensity(val)
        }
        Expr::Dft(inner) => {
            let val = eval_expr(inner, env)?;
            eval_dft_nd(val, false)
        }
        Expr::Idft(inner) => {
            let val = eval_expr(inner, env)?;
            eval_dft_nd(val, true)
        }
        Expr::Propagate2D { field, z, wavelength, dx } => {
            let f_val = eval_expr(field, env)?;
            let z_val = eval_float(&eval_expr(z, env)?)?;
            let lambda_val = eval_float(&eval_expr(wavelength, env)?)?;
            let dx_val = eval_float(&eval_expr(dx, env)?)?;
            eval_asm_propagation(f_val, z_val, lambda_val, dx_val)
        }
        Expr::Fdtd2D { steps, size } => {
            let s_val = eval_int(&eval_expr(steps, env)?)? as usize;
            let sz_val = eval_int(&eval_expr(size, env)?)? as usize;
            eval_fdtd_2d(s_val, sz_val)
        }
        Expr::QState(inner) => {
            let val = eval_expr(inner, env)?;
            eval_qstate(val)
        }
        Expr::BeamSplitter(inner) => {
            let val = eval_expr(inner, env)?;
            eval_beamsplitter(val)
        }
        Expr::KbInit => {
            let id = env.kbs.len();
            env.kbs.push(KnowledgeBase {
                facts: HashSet::new(),
                rules: Vec::new(),
            });
            Ok(Value::KbHandle(id))
        }
        Expr::KbFact { kb, fact } => {
            let kb_val = eval_expr(kb, env)?;
            let fact_val = eval_expr(fact, env)?;
            let id = match kb_val {
                Value::KbHandle(i) => i,
                _ => return Err("LUM-KB01: First argument to kb_fact must be a KnowledgeBase handle".into()),
            };
            let fact_str = match fact_val {
                Value::Str(s) => s.trim().to_string(),
                _ => return Err("LUM-KB02: Fact must be a string".into()),
            };
            if id < env.kbs.len() {
                env.kbs[id].facts.insert(fact_str);
                Ok(Value::Bool(true))
            } else {
                Err("LUM-KB03: Invalid KB handle".into())
            }
        }
        Expr::KbRule { kb, head, body } => {
            let kb_val = eval_expr(kb, env)?;
            let head_val = eval_expr(head, env)?;
            let body_val = eval_expr(body, env)?;
            let id = match kb_val {
                Value::KbHandle(i) => i,
                _ => return Err("LUM-KB04: First argument to kb_rule must be a KnowledgeBase handle".into()),
            };
            let head_str = match head_val {
                Value::Str(s) => s.trim().to_string(),
                _ => return Err("LUM-KB05: Rule head must be a string".into()),
            };
            let body_str = match body_val {
                Value::Str(s) => s,
                _ => return Err("LUM-KB06: Rule body must be a comma-separated string".into()),
            };
            let premises: Vec<String> = body_str
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect();

            if id < env.kbs.len() {
                env.kbs[id].rules.push(HornClause {
                    head: head_str,
                    body: premises,
                });
                Ok(Value::Bool(true))
            } else {
                Err("LUM-KB07: Invalid KB handle".into())
            }
        }
        Expr::KbProve { kb, query } => {
            let kb_val = eval_expr(kb, env)?;
            let query_val = eval_expr(query, env)?;
            let id = match kb_val {
                Value::KbHandle(i) => i,
                _ => return Err("LUM-KB08: First argument to kb_prove must be a KnowledgeBase handle".into()),
            };
            let query_str = match query_val {
                Value::Str(s) => s.trim().to_string(),
                _ => return Err("LUM-KB09: Query must be a string".into()),
            };
            if id < env.kbs.len() {
                let proven = forward_chain_prove(&env.kbs[id], &query_str);
                Ok(Value::Bool(proven))
            } else {
                Err("LUM-KB10: Invalid KB handle".into())
            }
        }
        Expr::Binary { left, op, right } => {
            let l = eval_expr(left, env)?;
            let r = eval_expr(right, env)?;
            eval_binary(l, op, r)
        }
    }
}

fn forward_chain_prove(kb: &KnowledgeBase, query: &str) -> bool {
    let mut known_facts = kb.facts.clone();
    if known_facts.contains(query) {
        return true;
    }

    loop {
        let mut new_inferred = false;
        for rule in &kb.rules {
            if known_facts.contains(&rule.head) {
                continue;
            }
            let satisfied = rule.body.iter().all(|premise| known_facts.contains(premise));
            if satisfied {
                known_facts.insert(rule.head.clone());
                new_inferred = true;
                if rule.head == query {
                    return true;
                }
            }
        }
        if !new_inferred {
            break;
        }
    }
    known_facts.contains(query)
}

fn eval_float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        _ => Err("LUM-E3001: Expected scalar number".into()),
    }
}

fn eval_int(v: &Value) -> Result<i64, String> {
    match v {
        Value::Int(n) => Ok(*n),
        Value::Float(f) => Ok(*f as i64),
        _ => Err("LUM-E3002: Expected integer".into()),
    }
}

fn eval_conj(val: Value) -> Result<Value, String> {
    match val {
        Value::Complex(c) => Ok(Value::Complex(Complex64::new(c.re, -c.im))),
        Value::Int(n) => Ok(Value::Int(n)),
        Value::Float(f) => Ok(Value::Float(f)),
        Value::Array(arr) => {
            let mut res = Vec::new();
            for item in arr {
                res.push(eval_conj(item)?);
            }
            Ok(Value::Array(res))
        }
        _ => Err("LUM-P3001: Cannot compute conjugate of non-numeric type".into()),
    }
}

fn eval_intensity(val: Value) -> Result<Value, String> {
    match val {
        Value::Complex(c) => Ok(Value::Float(((c.re * c.re + c.im * c.im) * 1e6).round() / 1e6)),
        Value::Int(n) => Ok(Value::Float((n * n) as f64)),
        Value::Float(f) => Ok(Value::Float(f * f)),
        Value::Array(arr) => {
            let mut res = Vec::new();
            for item in arr {
                res.push(eval_intensity(item)?);
            }
            Ok(Value::Array(res))
        }
        _ => Err("LUM-P3002: Cannot compute optical intensity of non-numeric type".into()),
    }
}

fn eval_qstate(val: Value) -> Result<Value, String> {
    match val {
        Value::Array(arr) => {
            let mut complex_vec = Vec::new();
            let mut norm_sq = 0.0;
            for item in arr {
                let c = to_complex(item).ok_or("LUM-Q01: Quantum state entries must be complex/numeric")?;
                norm_sq += c.re * c.re + c.im * c.im;
                complex_vec.push(c);
            }
            if norm_sq == 0.0 {
                return Err("LUM-Q02: Cannot normalize zero quantum state vector".into());
            }
            let norm = norm_sq.sqrt();
            let normalized: Vec<Value> = complex_vec
                .into_iter()
                .map(|c| {
                    let re = (c.re / norm * 1e6).round() / 1e6;
                    let im = (c.im / norm * 1e6).round() / 1e6;
                    Value::Complex(Complex64::new(re, im))
                })
                .collect();
            Ok(Value::Array(normalized))
        }
        _ => Err("LUM-Q03: qstate requires a vector input".into()),
    }
}

fn eval_beamsplitter(val: Value) -> Result<Value, String> {
    let state = eval_qstate(val)?;
    match state {
        Value::Array(arr) => {
            if arr.len() != 2 {
                return Err("LUM-Q04: 50:50 Beam splitter acts on a 2-mode quantum state [a, b]".into());
            }
            let a = match arr[0] { Value::Complex(c) => c, _ => unreachable!() };
            let b = match arr[1] { Value::Complex(c) => c, _ => unreachable!() };

            let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
            let ib = Complex64::new(-b.im, b.re);
            let ia = Complex64::new(-a.im, a.re);

            let out_a = a.add(ib);
            let out_b = ia.add(b);

            let res_a = Complex64::new(
                (out_a.re * inv_sqrt2 * 1e6).round() / 1e6,
                (out_a.im * inv_sqrt2 * 1e6).round() / 1e6,
            );
            let res_b = Complex64::new(
                (out_b.re * inv_sqrt2 * 1e6).round() / 1e6,
                (out_b.im * inv_sqrt2 * 1e6).round() / 1e6,
            );

            Ok(Value::Array(vec![Value::Complex(res_a), Value::Complex(res_b)]))
        }
        _ => unreachable!(),
    }
}

fn dft_1d(inputs: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = inputs.len();
    let mut output = Vec::new();
    let sign = if inverse { 1.0 } else { -1.0 };
    let scale = if inverse { n as f64 } else { 1.0 };

    for k in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for (j, &xj) in inputs.iter().enumerate() {
            let angle = sign * 2.0 * PI * (k as f64) * (j as f64) / (n as f64);
            let exp = Complex64::new(angle.cos(), angle.sin());
            sum = sum.add(xj.mul(exp));
        }
        let re = (sum.re / scale * 1e6).round() / 1e6;
        let im = (sum.im / scale * 1e6).round() / 1e6;
        output.push(Complex64::new(re, im));
    }
    output
}

fn eval_dft_nd(val: Value, inverse: bool) -> Result<Value, String> {
    match val {
        Value::Array(arr) => {
            if arr.is_empty() {
                return Ok(Value::Array(Vec::new()));
            }
            if matches!(&arr[0], Value::Array(_)) {
                let rows = arr.len();
                let mut mat: Vec<Vec<Complex64>> = Vec::new();
                for row in arr {
                    match row {
                        Value::Array(elements) => {
                            let mut row_c = Vec::new();
                            for el in elements {
                                row_c.push(to_complex(el).ok_or("LUM-DFT: Elements must be complex/numeric")?);
                            }
                            mat.push(row_c);
                        }
                        _ => return Err("LUM-DFT: Inconsistent dimensions".into()),
                    }
                }
                let cols = mat[0].len();
                for i in 0..rows {
                    mat[i] = dft_1d(&mat[i], inverse);
                }
                for j in 0..cols {
                    let mut col = Vec::new();
                    for i in 0..rows {
                        col.push(mat[i][j]);
                    }
                    let transformed = dft_1d(&col, inverse);
                    for i in 0..rows {
                        mat[i][j] = transformed[i];
                    }
                }
                let res = mat.into_iter().map(|r| Value::Array(r.into_iter().map(Value::Complex).collect())).collect();
                Ok(Value::Array(res))
            } else {
                let mut complex_inputs = Vec::new();
                for item in arr {
                    complex_inputs.push(to_complex(item).ok_or("LUM-DFT: Elements must be complex/numeric")?);
                }
                let out = dft_1d(&complex_inputs, inverse);
                Ok(Value::Array(out.into_iter().map(Value::Complex).collect()))
            }
        }
        _ => Err("LUM-DFT: Target must be array".into()),
    }
}

fn eval_asm_propagation(field_val: Value, z: f64, wavelength: f64, dx: f64) -> Result<Value, String> {
    let freq_field = eval_dft_nd(field_val, false)?;
    let k0 = 2.0 * PI / wavelength;
    let mat = match freq_field {
        Value::Array(rows) => rows,
        _ => return Err("LUM-ASM: Invalid 2D field".into()),
    };

    let n_rows = mat.len();
    let n_cols = match &mat[0] {
        Value::Array(c) => c.len(),
        _ => return Err("LUM-ASM: Must be 2D array".into()),
    };

    let mut modified = Vec::new();
    for i in 0..n_rows {
        let mut row_mod = Vec::new();
        let kx = 2.0 * PI * (i as f64 - n_rows as f64 / 2.0) / (n_rows as f64 * dx);
        match &mat[i] {
            Value::Array(cols) => {
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
            _ => unreachable!(),
        }
        modified.push(Value::Array(row_mod));
    }

    eval_dft_nd(Value::Array(modified), true)
}

fn eval_fdtd_2d(steps: usize, size: usize) -> Result<Value, String> {
    if size < 3 {
        return Err("LUM-FDTD: Grid size must be at least 3x3".into());
    }

    let mut ez = vec![vec![0.0; size]; size];
    let mut hx = vec![vec![0.0; size]; size];
    let mut hy = vec![vec![0.0; size]; size];

    let sc = 0.7;
    let center = size / 2;

    for t in 0..steps {
        for i in 0..size - 1 {
            for j in 0..size - 1 {
                hx[i][j] -= sc * (ez[i][j + 1] - ez[i][j]);
                hy[i][j] += sc * (ez[i + 1][j] - ez[i][j]);
            }
        }

        for i in 1..size {
            for j in 1..size {
                ez[i][j] += sc * ((hy[i][j] - hy[i - 1][j]) - (hx[i][j] - hx[i][j - 1]));
            }
        }

        let pulse = (-(((t as f64 - 10.0) / 4.0).powi(2))).exp();
        ez[center][center] += pulse;
    }

    let mut result = Vec::new();
    for row in ez {
        let row_vals = row.into_iter().map(|val| {
            let rounded = (val * 1e4).round() / 1e4;
            Value::Complex(Complex64::new(rounded, 0.0))
        }).collect();
        result.push(Value::Array(row_vals));
    }

    Ok(Value::Array(result))
}

fn to_complex(v: Value) -> Option<Complex64> {
    match v {
        Value::Int(n) => Some(Complex64::new(n as f64, 0.0)),
        Value::Float(f) => Some(Complex64::new(f, 0.0)),
        Value::Complex(c) => Some(c),
        _ => None,
    }
}

fn matrix_multiply(a: &[Value], b: &[Value]) -> Result<Value, String> {
    let rows_a = a.len();
    let mat_a: Vec<&Vec<Value>> = a.iter().map(|row| match row {
        Value::Array(r) => Ok(r),
        _ => Err("LUM-M7001: LHS in @ must be a 2D matrix".to_string()),
    }).collect::<Result<_, _>>()?;

    let cols_a = if rows_a > 0 { mat_a[0].len() } else { 0 };

    let rows_b = b.len();
    let mat_b: Vec<&Vec<Value>> = b.iter().map(|row| match row {
        Value::Array(r) => Ok(r),
        _ => Err("LUM-M7002: RHS in @ must be a 2D matrix".to_string()),
    }).collect::<Result<_, _>>()?;

    let cols_b = if rows_b > 0 { mat_b[0].len() } else { 0 };

    if cols_a != rows_b {
        return Err(format!("LUM-M7003: Incompatible dimensions for matrix multiplication: {}x{} vs {}x{}", rows_a, cols_a, rows_b, cols_b));
    }

    let mut result = Vec::new();
    for i in 0..rows_a {
        let mut row_res = Vec::new();
        for j in 0..cols_b {
            let mut sum = Value::Int(0);
            for k in 0..cols_a {
                let prod = eval_binary(mat_a[i][k].clone(), &BinaryOp::Mul, mat_b[k][j].clone())?;
                sum = eval_binary(sum, &BinaryOp::Add, prod)?;
            }
            row_res.push(sum);
        }
        result.push(Value::Array(row_res));
    }

    Ok(Value::Array(result))
}

fn eval_binary(left: Value, op: &BinaryOp, right: Value) -> Result<Value, String> {
    if let BinaryOp::MatMul = op {
        if let (Value::Array(a), Value::Array(b)) = (&left, &right) {
            return matrix_multiply(a, b);
        }
        return Err("LUM-M7004: @ operator requires array matrices".into());
    }

    if let (Value::Array(a), Value::Array(b)) = (&left, &right) {
        if a.len() != b.len() {
            return Err("LUM-T6001: Tensor dimension mismatch for vector addition".into());
        }
        let mut result = Vec::new();
        for (x, y) in a.iter().zip(b.iter()) {
            result.push(eval_binary(x.clone(), op, y.clone())?);
        }
        return Ok(Value::Array(result));
    }

    if matches!(left, Value::Complex(_)) || matches!(right, Value::Complex(_)) {
        let c1 = to_complex(left).ok_or("LUM-M5001: Invalid complex conversion")?;
        let c2 = to_complex(right).ok_or("LUM-M5001: Invalid complex conversion")?;
        return match op {
            BinaryOp::Add => Ok(Value::Complex(c1.add(c2))),
            BinaryOp::Sub => Ok(Value::Complex(c1.sub(c2))),
            BinaryOp::Mul => Ok(Value::Complex(c1.mul(c2))),
            _ => Err("LUM-M5002: Operator not yet implemented for Complex numbers".into()),
        };
    }

    match (left, right) {
        (Value::Int(a), Value::Int(b)) => match op {
            BinaryOp::Add => Ok(Value::Int(a + b)),
            BinaryOp::Sub => Ok(Value::Int(a - b)),
            BinaryOp::Mul => Ok(Value::Int(a * b)),
            BinaryOp::Div => {
                if b == 0 { Err("LUM-R4002: Division by zero".into()) } else { Ok(Value::Int(a / b)) }
            }
            BinaryOp::Equal => Ok(Value::Bool(a == b)),
            _ => Err("LUM-R4005: Unsupported operator".into()),
        },
        (Value::Float(a), Value::Float(b)) => match op {
            BinaryOp::Add => Ok(Value::Float(a + b)),
            BinaryOp::Sub => Ok(Value::Float(a - b)),
            BinaryOp::Mul => Ok(Value::Float(a * b)),
            BinaryOp::Div => {
                if b == 0.0 { Err("LUM-R4002: Division by zero".into()) } else { Ok(Value::Float(a / b)) }
            }
            BinaryOp::Equal => Ok(Value::Bool(a == b)),
            _ => Err("LUM-R4005: Unsupported operator".into()),
        },
        (Value::Str(a), Value::Str(b)) => match op {
            BinaryOp::Add => Ok(Value::Str(format!("{}{}", a, b))),
            BinaryOp::Equal => Ok(Value::Bool(a == b)),
            _ => Err("LUM-R4003: Unsupported operator for strings".into()),
        },
        _ => Err("LUM-R4004: Type mismatch in binary expression".into()),
    }
}
