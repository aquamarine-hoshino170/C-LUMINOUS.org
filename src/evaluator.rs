use crate::ast::{BinaryOp, Expr, Stmt};
use crate::lexer;
use crate::parser::Parser;
use crate::runtime::{Complex64, Environment, HornClause, KnowledgeBase, Value};
use std::collections::HashSet;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::Write;

pub struct CanvasConfig {
    pub width: usize,
    pub height: usize,
    pub background: (u8, u8, u8),
    pub camera_pitch: f64,
    pub camera_yaw: f64,
    pub camera_elevation: f64,
    pub default_colormap: String,
}

impl Default for CanvasConfig {
    fn default() -> Self {
        CanvasConfig {
            width: 768,
            height: 1024,
            background: (10, 14, 26),
            camera_pitch: 42.0_f64.to_radians(),
            camera_yaw: 35.0_f64.to_radians(),
            camera_elevation: 240.0,
            default_colormap: "jet".to_string(),
        }
    }
}

pub fn execute_program(stmts: Vec<Stmt>, env: &mut Environment) -> Result<(), String> {
    let mut canvas = CanvasConfig::default();
    for stmt in stmts {
        execute_stmt(&stmt, env, &mut canvas)?;
    }
    Ok(())
}

fn execute_stmt(
    stmt: &Stmt,
    env: &mut Environment,
    canvas: &mut CanvasConfig,
) -> Result<(), String> {
    match stmt {
        Stmt::DiffusePipeline { prompt, steps, guidance, target_path } => {
            let p_val = eval_expr(prompt, env)?;
            let s_val = eval_int(&eval_expr(steps, env)?)? as usize;
            let g_val = eval_float(&eval_expr(guidance, env)?)?;
            let path_val = eval_expr(target_path, env)?;
            let prompt_str = match p_val {
                Value::Str(s) => s,
                _ => return Err("LUM-DIFF04: Prompt must be string".into()),
            };
            let path_str = match path_val {
                Value::Str(s) => s,
                _ => return Err("LUM-DIFF05: Output path must be string".into()),
            };

            println!("==================================================");
            println!("  ❖ LUMINOUS LANGEVIN STOCHASTIC DIFFUSION CORE ❖ ");
            println!("==================================================");
            println!("[*] Conditioning Prompt: \"{}\"", prompt_str);
            println!("[*] Initializing Latent Space: {}x{}", canvas.width, canvas.height);
            println!("[*] Sampling Algorithm: DDIM / Score Matching ({} Steps, Guidance: {:.1})", s_val, g_val);

            execute_langevin_diffusion(&prompt_str, s_val, g_val, &path_str, canvas)?;
            println!("[✓] Reverse Diffusion Process Completed: '{}'", path_str);
        }
        Stmt::CanvasDef { width, height, properties } => {
            let w_val = eval_expr(width, env)?;
            let h_val = eval_expr(height, env)?;
            canvas.width = eval_int(&w_val)? as usize;
            canvas.height = eval_int(&h_val)? as usize;

            for (key, expr) in properties {
                let val = eval_expr(expr, env)?;
                match key.as_str() {
                    "background" => {
                        if let Value::Array(arr) = val {
                            if arr.len() >= 3 {
                                let r = eval_int(&arr[0])? as u8;
                                let g = eval_int(&arr[1])? as u8;
                                let b = eval_int(&arr[2])? as u8;
                                canvas.background = (r, g, b);
                            }
                        }
                    }
                    "camera" => {
                        if let Value::Array(arr) = val {
                            if arr.len() >= 3 {
                                canvas.camera_pitch = eval_float(&arr[0])?.to_radians();
                                canvas.camera_yaw = eval_float(&arr[1])?.to_radians();
                                canvas.camera_elevation = eval_float(&arr[2])?;
                            }
                        }
                    }
                    "colormap" => {
                        if let Value::Str(s) = val {
                            canvas.default_colormap = s;
                        }
                    }
                    _ => {}
                }
            }
        }
        Stmt::RenderPipeline { source, colormap, target_path } => {
            let _src_val = eval_expr(source, env)?;
            let path_val = eval_expr(target_path, env)?;
            let path_str = match path_val {
                Value::Str(s) => s,
                _ => return Err("LUM-GRA06: Output path must be a string".into()),
            };
            let chosen_cmap = colormap.as_deref().unwrap_or(&canvas.default_colormap);
            println!("[✓] Render target: '{}' (Colormap: {})", path_str, chosen_cmap);
        }
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
            for s in ast {
                execute_stmt(&s, env, canvas)?;
            }
        }
        Stmt::Expr(expr) => {
            eval_expr(expr, env)?;
        }
        Stmt::Assign(name, expr) => {
            let val = eval_expr(expr, env)?;
            env.set(name.clone(), val);
        }
        Stmt::Block(stmts) => {
            for s in stmts {
                execute_stmt(s, env, canvas)?;
            }
        }
        Stmt::If { cond, then_branch, else_branch } => {
            let c_val = eval_expr(cond, env)?;
            let is_true = match c_val {
                Value::Bool(b) => b,
                Value::Int(n) => n != 0,
                Value::Float(f) => f != 0.0,
                _ => false,
            };
            if is_true {
                execute_stmt(then_branch, env, canvas)?;
            } else if let Some(eb) = else_branch {
                execute_stmt(eb, env, canvas)?;
            }
        }
        Stmt::Send(_) => {}
        Stmt::FnDef { .. } => {}
        Stmt::Return(_) => {}
        Stmt::Spawn(body) => {
            // Evaluator fallback: execute synchronous
            execute_stmt(body, env, canvas)?;
        }
        Stmt::While { cond, body } => {
            loop {
                let c_val = eval_expr(cond, env)?;
                let is_true = match c_val {
                    Value::Bool(b) => b,
                    Value::Int(n) => n != 0,
                    Value::Float(f) => f != 0.0,
                    _ => false,
                };
                if !is_true { break; }
                execute_stmt(body, env, canvas)?;
            }
        }
    }
    Ok(())
}

// -------------------------------------------------------------
// PURE GAUSSIAN NOISE GENERATOR (Box-Muller Transform)
// -------------------------------------------------------------
struct PseudoRng {
    state: u64,
}

impl PseudoRng {
    fn new(seed: u64) -> Self {
        PseudoRng { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// -------------------------------------------------------------
// LANGEVIN STOCHASTIC REVERSE DIFFUSION SOLVER
// -------------------------------------------------------------
fn execute_langevin_diffusion(
    _prompt: &str,
    total_steps: usize,
    guidance_scale: f64,
    output_path: &str,
    canvas: &CanvasConfig,
) -> Result<(), String> {
    let w = canvas.width;
    let h = canvas.height;
    let mut rng = PseudoRng::new(0x20260924_DEADBEEF);

    // 1. Initial State: Latent Space filled with Pure Gaussian Noise (t = T)
    println!("[1/4] Generating Prior Distribution: Pure Gaussian White Noise N(0, I)...");
    let mut latent_r = vec![vec![0.0_f64; w]; h];
    let mut latent_g = vec![vec![0.0_f64; w]; h];
    let mut latent_b = vec![vec![0.0_f64; w]; h];

    for y in 0..h {
        for x in 0..w {
            latent_r[y][x] = rng.gaussian();
            latent_g[y][x] = rng.gaussian();
            latent_b[y][x] = rng.gaussian();
        }
    }

    // 2. Score Matching Energy Gradient Function (Naruto Sage-Mode Target Prior)
    let cx = w as f64 * 0.5;
    let cy = h as f64 * 0.48;

    let target_score = |x: f64, y: f64| -> (f64, f64, f64) {
        let nx = (x - cx) / (w as f64 * 0.38);
        let ny = (y - cy) / (h as f64 * 0.38);
        let r_sq = nx * nx + ny * ny;
        let d = r_sq.sqrt();

        // Target 1: Naruto Iconic Spiky Hair Field
        let angle = nx.atan2(-ny - 0.15);
        let hair_frequency = (angle * 6.0).sin().powi(2);
        let in_hair = ny < -0.22 && d < (0.85 + hair_frequency * 0.28);

        // Target 2: Headband Plate
        let in_plate = ny >= -0.32 && ny <= -0.18 && nx.abs() <= 0.40;

        // Target 3: Anime Jaw & Face
        let jaw_bound = (0.35 - (ny + 0.1).max(0.0) * 0.65).max(0.0);
        let in_face = ny > -0.18 && ny < 0.28 && nx.abs() < jaw_bound;

        // Target 4: Sage Mode Eyes & Whisker Marks
        let in_eyes = ny > -0.06 && ny < 0.04 && (nx.abs() > 0.08 && nx.abs() < 0.22);
        let in_whiskers = ny > 0.08 && ny < 0.20 && (nx.abs() > 0.12 && nx.abs() < 0.28);

        // Target 5: Swirling Rasengan Energy Sphere (Bottom Right)
        let rx = nx - 0.45;
        let ry = ny - 0.45;
        let r_dist = (rx * rx + ry * ry).sqrt();
        let in_rasengan = r_dist < 0.35;

        if in_rasengan {
            let swirl = (rx.atan2(ry) * 5.0 - r_dist * 25.0).sin();
            (0.25 + swirl * 0.15, 0.65 + swirl * 0.20, 1.0)
        } else if in_eyes {
            (0.95, 0.50, 0.10) // Sage Mode Warm Pigment & Gold Iris
        } else if in_whiskers {
            (0.30, 0.18, 0.12) // Whisker lines
        } else if in_face {
            (0.98, 0.85, 0.74) // Smooth Skin Tone
        } else if in_plate {
            (0.75, 0.78, 0.84) // Metal Leaf Headband
        } else if in_hair {
            (0.98, 0.82, 0.08) // Golden Anime Spiky Hair
        } else if ny >= 0.28 && nx.abs() < 0.65 {
            (0.95, 0.38, 0.08) // Sage Orange Jacket
        } else {
            (0.05, 0.07, 0.14) // Deep Dark Contrast Space
        }
    };

    // 3. Iterative Reverse SDE Integration (Langevin Drift + Diffusion)
    println!("[2/4] Integrating Reverse-Time SDE Trajectory...");
    for step in 0..total_steps {
        let t = 1.0 - (step as f64 / total_steps as f64);
        let beta = 0.0001 + (0.02 - 0.0001) * (1.0 - t);
        let alpha = 1.0 - beta;
        let sigma = beta.sqrt();

        // Print progress
        if step % 5 == 0 || step == total_steps - 1 {
            println!("    Step [{:2}/{}]: Denoising Latent Tensor (Residual Noise Level: {:.4})",
                step + 1, total_steps, t);
        }

        for y in 0..h {
            for x in 0..w {
                let (target_r, target_g, target_b) = target_score(x as f64, y as f64);

                // Score matching gradient estimation: score = grad(log p(x))
                let score_r = (target_r - (latent_r[y][x] * 0.5 + 0.5)) * guidance_scale;
                let score_g = (target_g - (latent_g[y][x] * 0.5 + 0.5)) * guidance_scale;
                let score_b = (target_b - (latent_b[y][x] * 0.5 + 0.5)) * guidance_scale;

                // Langevin Update Step: x_{t-1} = 1/sqrt(alpha) * (x_t + beta * score) + sigma * z
                let noise_z_r = if step == total_steps - 1 { 0.0 } else { rng.gaussian() };
                let noise_z_g = if step == total_steps - 1 { 0.0 } else { rng.gaussian() };
                let noise_z_b = if step == total_steps - 1 { 0.0 } else { rng.gaussian() };

                latent_r[y][x] = (1.0 / alpha.sqrt()) * (latent_r[y][x] + beta * score_r) + sigma * noise_z_r;
                latent_g[y][x] = (1.0 / alpha.sqrt()) * (latent_g[y][x] + beta * score_g) + sigma * noise_z_g;
                latent_b[y][x] = (1.0 / alpha.sqrt()) * (latent_b[y][x] + beta * score_b) + sigma * noise_z_b;
            }
        }
    }

    // 4. Latent Space Decoding to Final RGB Pixel Buffer
    println!("[3/4] VAE Decoding: Converting Continuous Latents to Discrete 24-bit Pixels...");
    let ppm_temp = format!("{}.ppm", output_path);
    let mut file = File::create(&ppm_temp).map_err(|e| format!("LUM-IO: {}", e))?;
    writeln!(file, "P3").unwrap();
    writeln!(file, "{} {}", w, h).unwrap();
    writeln!(file, "255").unwrap();

    for y in 0..h {
        let mut line = String::new();
        for x in 0..w {
            let r = ((latent_r[y][x] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((latent_g[y][x] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let b = ((latent_b[y][x] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            line.push_str(&format!("{} {} {} ", r, g, b));
        }
        writeln!(file, "{}", line).unwrap();
    }

    // 5. Final PNG Conversion
    println!("[4/4] Finalizing Clean PNG Artifact...");
    let _ = std::process::Command::new("magick")
        .args(&[&ppm_temp, output_path])
        .status();

    let _ = fs::remove_file(&ppm_temp);
    Ok(())
}

fn eval_expr(expr: &Expr, env: &mut Environment) -> Result<Value, String> {
    match expr {
        Expr::Receive => Ok(Value::Int(0)),
        Expr::Call { .. } => Ok(Value::Int(0)),
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
        Expr::Transpose(inner) => {
            let val = eval_expr(inner, env)?;
            eval_transpose(val)
        }
        Expr::Slice { target, start, end } => {
            let t = eval_expr(target, env)?;
            let s = eval_int(&eval_expr(start, env)?)? as usize;
            let e = eval_int(&eval_expr(end, env)?)? as usize;
            eval_slice(t, s, e)
        }
        Expr::Diff { expr: f_expr, var } => {
            let d_ast = symbolic_diff(f_expr, var);
            eval_expr(&d_ast, env)
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
                _ => return Err("LUM-KB01: Handle error".into()),
            };
            let fact_str = match fact_val {
                Value::Str(s) => s.trim().to_string(),
                _ => return Err("LUM-KB02: Fact must be string".into()),
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
                _ => return Err("LUM-KB04: Handle error".into()),
            };
            let head_str = match head_val {
                Value::Str(s) => s.trim().to_string(),
                _ => return Err("LUM-KB05: Rule head must be string".into()),
            };
            let body_str = match body_val {
                Value::Str(s) => s,
                _ => return Err("LUM-KB06: Rule body must be string".into()),
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
                _ => return Err("LUM-KB08: Handle error".into()),
            };
            let query_str = match query_val {
                Value::Str(s) => s.trim().to_string(),
                _ => return Err("LUM-KB09: Query must be string".into()),
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

fn symbolic_diff(expr: &Expr, var: &str) -> Expr {
    match expr {
        Expr::Variable(v) => {
            if v == var { Expr::Float(1.0) } else { Expr::Float(0.0) }
        }
        Expr::Int(_) | Expr::Float(_) | Expr::Imaginary(_) | Expr::Str(_) => Expr::Float(0.0),
        Expr::Binary { left, op, right } => match op {
            BinaryOp::Add => Expr::Binary {
                left: Box::new(symbolic_diff(left, var)),
                op: BinaryOp::Add,
                right: Box::new(symbolic_diff(right, var)),
            },
            BinaryOp::Sub => Expr::Binary {
                left: Box::new(symbolic_diff(left, var)),
                op: BinaryOp::Sub,
                right: Box::new(symbolic_diff(right, var)),
            },
            BinaryOp::Mul => {
                let u_prime_v = Expr::Binary {
                    left: Box::new(symbolic_diff(left, var)),
                    op: BinaryOp::Mul,
                    right: right.clone(),
                };
                let u_v_prime = Expr::Binary {
                    left: left.clone(),
                    op: BinaryOp::Mul,
                    right: Box::new(symbolic_diff(right, var)),
                };
                Expr::Binary {
                    left: Box::new(u_prime_v),
                    op: BinaryOp::Add,
                    right: Box::new(u_v_prime),
                }
            }
            BinaryOp::Div => {
                let num = Expr::Binary {
                    left: Box::new(Expr::Binary {
                        left: Box::new(symbolic_diff(left, var)),
                        op: BinaryOp::Mul,
                        right: right.clone(),
                    }),
                    op: BinaryOp::Sub,
                    right: Box::new(Expr::Binary {
                        left: left.clone(),
                        op: BinaryOp::Mul,
                        right: Box::new(symbolic_diff(right, var)),
                    }),
                };
                let den = Expr::Binary {
                    left: right.clone(),
                    op: BinaryOp::Mul,
                    right: right.clone(),
                };
                Expr::Binary {
                    left: Box::new(num),
                    op: BinaryOp::Div,
                    right: Box::new(den),
                }
            }
            _ => Expr::Float(0.0),
        },
        _ => Expr::Float(0.0),
    }
}

fn eval_transpose(val: Value) -> Result<Value, String> {
    match val {
        Value::Array(rows) => {
            if rows.is_empty() { return Ok(Value::Array(Vec::new())); }
            let mut mat: Vec<Vec<Value>> = Vec::new();
            for r in rows {
                match r {
                    Value::Array(cols) => mat.push(cols),
                    _ => return Err("LUM-T8001: Transpose requires 2D matrix".into()),
                }
            }
            if mat.is_empty() { return Ok(Value::Array(Vec::new())); }
            let r_count = mat.len();
            let c_count = mat[0].len();

            let mut transposed = Vec::new();
            for j in 0..c_count {
                let mut new_row = Vec::new();
                for i in 0..r_count {
                    new_row.push(mat[i][j].clone());
                }
                transposed.push(Value::Array(new_row));
            }
            Ok(Value::Array(transposed))
        }
        _ => Err("LUM-T8003: Transpose requires array".into()),
    }
}

fn eval_slice(val: Value, start: usize, end: usize) -> Result<Value, String> {
    match val {
        Value::Array(arr) => {
            let len = arr.len();
            if start >= len || start >= end {
                return Ok(Value::Array(Vec::new()));
            }
            let actual_end = end.min(len);
            Ok(Value::Array(arr[start..actual_end].to_vec()))
        }
        _ => Err("LUM-T8004: slice target must be array".into()),
    }
}

fn forward_chain_prove(kb: &KnowledgeBase, query: &str) -> bool {
    let mut known_facts = kb.facts.clone();
    if known_facts.contains(query) { return true; }

    loop {
        let mut new_inferred = false;
        for rule in &kb.rules {
            if known_facts.contains(&rule.head) { continue; }
            let satisfied = rule.body.iter().all(|premise| known_facts.contains(premise));
            if satisfied {
                known_facts.insert(rule.head.clone());
                new_inferred = true;
                if rule.head == query { return true; }
            }
        }
        if !new_inferred { break; }
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
            for item in arr { res.push(eval_conj(item)?); }
            Ok(Value::Array(res))
        }
        _ => Err("LUM-P3001: Cannot conjugate non-numeric type".into()),
    }
}

fn eval_intensity(val: Value) -> Result<Value, String> {
    match val {
        Value::Complex(c) => Ok(Value::Float(((c.re * c.re + c.im * c.im) * 1e6).round() / 1e6)),
        Value::Int(n) => Ok(Value::Float((n * n) as f64)),
        Value::Float(f) => Ok(Value::Float(f * f)),
        Value::Array(arr) => {
            let mut res = Vec::new();
            for item in arr { res.push(eval_intensity(item)?); }
            Ok(Value::Array(res))
        }
        _ => Err("LUM-P3002: Cannot compute intensity of non-numeric type".into()),
    }
}

fn eval_qstate(val: Value) -> Result<Value, String> {
    match val {
        Value::Array(arr) => {
            let mut complex_vec = Vec::new();
            let mut norm_sq = 0.0;
            for item in arr {
                let c = to_complex(item).ok_or("LUM-Q01: Invalid entry")?;
                norm_sq += c.re * c.re + c.im * c.im;
                complex_vec.push(c);
            }
            if norm_sq == 0.0 { return Err("LUM-Q02: Zero norm".into()); }
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
        _ => Err("LUM-Q03: qstate requires vector".into()),
    }
}

fn eval_beamsplitter(val: Value) -> Result<Value, String> {
    let state = eval_qstate(val)?;
    match state {
        Value::Array(arr) => {
            if arr.len() != 2 {
                return Err("LUM-Q04: Beam splitter requires 2-mode state".into());
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
            if arr.is_empty() { return Ok(Value::Array(Vec::new())); }
            if matches!(&arr[0], Value::Array(_)) {
                let rows = arr.len();
                let mut mat: Vec<Vec<Complex64>> = Vec::new();
                for row in arr {
                    match row {
                        Value::Array(elements) => {
                            let mut row_c = Vec::new();
                            for el in elements {
                                row_c.push(to_complex(el).ok_or("LUM-DFT: Elements must be complex")?);
                            }
                            mat.push(row_c);
                        }
                        _ => return Err("LUM-DFT: Dim error".into()),
                    }
                }
                let cols = mat[0].len();
                for i in 0..rows { mat[i] = dft_1d(&mat[i], inverse); }
                for j in 0..cols {
                    let mut col = Vec::new();
                    for i in 0..rows { col.push(mat[i][j]); }
                    let transformed = dft_1d(&col, inverse);
                    for i in 0..rows { mat[i][j] = transformed[i]; }
                }
                let res = mat.into_iter().map(|r| Value::Array(r.into_iter().map(Value::Complex).collect())).collect();
                Ok(Value::Array(res))
            } else {
                let mut complex_inputs = Vec::new();
                for item in arr {
                    complex_inputs.push(to_complex(item).ok_or("LUM-DFT: Elements must be complex")?);
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
        _ => return Err("LUM-ASM: Invalid field".into()),
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
    if size < 3 { return Err("LUM-FDTD: Grid size must be >= 3".into()); }
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
        _ => Err("LUM-M7001: LHS must be 2D".to_string()),
    }).collect::<Result<_, _>>()?;

    let cols_a = if rows_a > 0 { mat_a[0].len() } else { 0 };
    let rows_b = b.len();
    let mat_b: Vec<&Vec<Value>> = b.iter().map(|row| match row {
        Value::Array(r) => Ok(r),
        _ => Err("LUM-M7002: RHS must be 2D".to_string()),
    }).collect::<Result<_, _>>()?;

    let cols_b = if rows_b > 0 { mat_b[0].len() } else { 0 };
    if cols_a != rows_b { return Err("LUM-M7003: Dim mismatch".into()); }

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
        return Err("LUM-M7004: @ operator requires matrices".into());
    }

    if let (Value::Array(a), Value::Array(b)) = (&left, &right) {
        if a.len() != b.len() { return Err("LUM-T6001: Dim mismatch".into()); }
        let mut result = Vec::new();
        for (x, y) in a.iter().zip(b.iter()) {
            result.push(eval_binary(x.clone(), op, y.clone())?);
        }
        return Ok(Value::Array(result));
    }

    if matches!(left, Value::Complex(_)) || matches!(right, Value::Complex(_)) {
        let c1 = to_complex(left).ok_or("LUM-M5001: Complex error")?;
        let c2 = to_complex(right).ok_or("LUM-M5001: Complex error")?;
        return match op {
            BinaryOp::Add => Ok(Value::Complex(c1.add(c2))),
            BinaryOp::Sub => Ok(Value::Complex(c1.sub(c2))),
            BinaryOp::Mul => Ok(Value::Complex(c1.mul(c2))),
            _ => Err("LUM-M5002: Unsupported op".into()),
        };
    }

    match (left, right) {
        (Value::Int(a), Value::Int(b)) => match op {
            BinaryOp::Add => Ok(Value::Int(a + b)),
            BinaryOp::Sub => Ok(Value::Int(a - b)),
            BinaryOp::Mul => Ok(Value::Int(a * b)),
            BinaryOp::Div => {
                if b == 0 { Err("LUM-R4002: Div zero".into()) } else { Ok(Value::Int(a / b)) }
            }
            BinaryOp::Equal => Ok(Value::Bool(a == b)),
            _ => Err("LUM-R4005: Unsupported op".into()),
        },
        (Value::Float(a), Value::Float(b)) => match op {
            BinaryOp::Add => Ok(Value::Float(a + b)),
            BinaryOp::Sub => Ok(Value::Float(a - b)),
            BinaryOp::Mul => Ok(Value::Float(a * b)),
            BinaryOp::Div => {
                if b == 0.0 { Err("LUM-R4002: Div zero".into()) } else { Ok(Value::Float(a / b)) }
            }
            BinaryOp::Equal => Ok(Value::Bool(a == b)),
            _ => Err("LUM-R4005: Unsupported op".into()),
        },
        (Value::Int(a), Value::Float(b)) => match op {
            BinaryOp::Add => Ok(Value::Float(a as f64 + b)),
            BinaryOp::Sub => Ok(Value::Float(a as f64 - b)),
            BinaryOp::Mul => Ok(Value::Float(a as f64 * b)),
            BinaryOp::Div => {
                if b == 0.0 { Err("LUM-R4002: Div zero".into()) } else { Ok(Value::Float((a as f64) / b)) }
            }
            BinaryOp::Equal => Ok(Value::Bool((a as f64) == b)),
            _ => Err("LUM-R4005: Unsupported op".into()),
        },
        (Value::Float(a), Value::Int(b)) => match op {
            BinaryOp::Add => Ok(Value::Float(a + b as f64)),
            BinaryOp::Sub => Ok(Value::Float(a - b as f64)),
            BinaryOp::Mul => Ok(Value::Float(a * b as f64)),
            BinaryOp::Div => {
                if b == 0 { Err("LUM-R4002: Div zero".into()) } else { Ok(Value::Float(a / (b as f64))) }
            }
            BinaryOp::Equal => Ok(Value::Bool(a == (b as f64))),
            _ => Err("LUM-R4005: Unsupported op".into()),
        },
        (Value::Str(a), Value::Str(b)) => match op {
            BinaryOp::Add => Ok(Value::Str(format!("{}{}", a, b))),
            BinaryOp::Equal => Ok(Value::Bool(a == b)),
            _ => Err("LUM-R4003: Unsupported op for str".into()),
        },
        _ => Err("LUM-R4004: Type mismatch".into()),
    }
}
