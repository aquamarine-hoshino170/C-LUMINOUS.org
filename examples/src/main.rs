use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};
use std::fs;
use std::env;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Nil,
    Number(f64),
    String(String),
    Bool(bool),
    List(Vec<Value>),
    Map(HashMap<String, Value>),
    Channel(Arc<Mutex<mpsc::Sender<Value>>>),
}

impl Value {
    pub fn to_string_repr(&self) -> String {
        match self {
            Value::String(s) => s.clone(),
            Value::Number(n) => {
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            Value::Bool(b) => b.to_string(),
            Value::Nil => "nil".to_string(),
            Value::List(l) => {
                let items: Vec<String> = l.iter().map(|v| v.to_string_repr()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Map(m) => {
                let items: Vec<String> = m.iter().map(|(k, v)| format!("{}: {}", k, v.to_string_repr())).collect();
                format!("{{{}}}", items.join(", "))
            }
            Value::Channel(_) => "<channel>".to_string(),
        }
    }
}

fn eval_atomic(expr: &str, env: &HashMap<String, Value>) -> Value {
    let expr = expr.trim();
    if expr.starts_with('"') && expr.ends_with('"') && expr.len() >= 2 {
        let mut inner = expr[1..expr.len() - 1].to_string();
        for (k, v) in env {
            let placeholder = format!("${{{}}}", k);
            inner = inner.replace(&placeholder, &v.to_string_repr());
        }
        Value::String(inner)
    } else if let Ok(n) = expr.parse::<f64>() {
        Value::Number(n)
    } else if expr == "true" {
        Value::Bool(true)
    } else if expr == "false" {
        Value::Bool(false)
    } else if let Some(existing) = env.get(expr) {
        existing.clone()
    } else {
        Value::Nil
    }
}

fn eval_expression(expr: &str, env: &HashMap<String, Value>) -> Value {
    let expr = expr.trim();
    if expr.contains('+') {
        let parts: Vec<&str> = expr.split('+').collect();
        let mut combined_str = String::new();
        let mut numeric_sum = 0.0;
        let mut is_pure_number = true;

        for part in &parts {
            let val = eval_atomic(part, env);
            match val {
                Value::Number(n) if is_pure_number => {
                    numeric_sum += n;
                }
                _ => {
                    is_pure_number = false;
                }
            }
        }

        if is_pure_number && !parts.is_empty() {
            Value::Number(numeric_sum)
        } else {
            for part in parts {
                let val = eval_atomic(part, env);
                combined_str.push_str(&val.to_string_repr());
            }
            Value::String(combined_str)
        }
    } else {
        eval_atomic(expr, env)
    }
}

fn execute_block(lines: &[String], env: &mut HashMap<String, Value>) -> Result<(), ()> {
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }

        if line == "break" {
            return Err(());
        }

        if line.starts_with("tell me ") {
            let expr = line["tell me ".len()..].trim();
            let result = eval_expression(expr, env);
            println!("{}", result.to_string_repr());
            i += 1;
        } else if line.starts_with("today ") {
            let rest = line["today ".len()..].trim();
            if let Some((var, val_expr)) = rest.split_once(" is ") {
                let var = var.trim();
                let val = eval_expression(val_expr.trim(), env);
                env.insert(var.to_string(), val);
            }
            i += 1;
        } else if line.starts_with("for ") && line.ends_with('{') {
            let header = line[4..line.len() - 1].trim();
            let mut block_lines = Vec::new();
            i += 1;
            while i < lines.len() && lines[i].trim() != "}" {
                block_lines.push(lines[i].clone());
                i += 1;
            }
            i += 1; // skip '}'

            if let Some((var_part, range_part)) = header.split_once(" from ") {
                let loop_var = var_part.trim();
                let step = if range_part.contains(" step ") {
                    let parts: Vec<&str> = range_part.split(" step ").collect();
                    parts[1].trim().parse::<i64>().unwrap_or(1)
                } else {
                    1
                };

                let range_clean = if range_part.contains(" step ") {
                    range_part.split(" step ").next().unwrap()
                } else {
                    range_part
                };

                if let Some((start_s, end_s)) = range_clean.split_once(" to ") {
                    let start = eval_expression(start_s.trim(), env).to_string_repr().parse::<i64>().unwrap_or(0);
                    let end = eval_expression(end_s.trim(), env).to_string_repr().parse::<i64>().unwrap_or(0);

                    let mut cur = start;
                    while cur <= end {
                        env.insert(loop_var.to_string(), Value::Number(cur as f64));
                        if execute_block(&block_lines, env).is_err() {
                            break;
                        }
                        cur += step;
                    }
                }
            }
        } else if line.starts_with("switch ") && line.ends_with('{') {
            let target_expr = line[7..line.len() - 1].trim();
            let target_val = eval_expression(target_expr, env);
            let mut matched = false;
            i += 1;

            while i < lines.len() && lines[i].trim() != "}" {
                let current_line = lines[i].trim();
                if current_line.starts_with("case ") && current_line.ends_with('{') {
                    let case_expr = current_line[5..current_line.len() - 1].trim();
                    let case_val = eval_expression(case_expr, env);
                    let mut case_block = Vec::new();
                    i += 1;
                    while i < lines.len() && lines[i].trim() != "}" {
                        case_block.push(lines[i].clone());
                        i += 1;
                    }
                    i += 1; // skip '}'

                    if !matched && case_val == target_val {
                        let _ = execute_block(&case_block, env);
                        matched = true;
                    }
                } else if current_line.starts_with("default {") {
                    let mut def_block = Vec::new();
                    i += 1;
                    while i < lines.len() && lines[i].trim() != "}" {
                        def_block.push(lines[i].clone());
                        i += 1;
                    }
                    i += 1; // skip '}'

                    if !matched {
                        let _ = execute_block(&def_block, env);
                        matched = true;
                    }
                } else {
                    i += 1;
                }
            }
            i += 1; // skip switch '}'
        } else {
            i += 1;
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Luminous Engine v2.0 (Phase 1 Full)");
        println!("Usage: luminous <script.lum>");
        return;
    }

    let filename = &args[1];
    let content = fs::read_to_string(filename).unwrap_or_else(|_| {
        eprintln!("Error: Could not read file {}", filename);
        std::process::exit(1);
    });

    let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let mut env: HashMap<String, Value> = HashMap::new();
    let _ = execute_block(&lines, &mut env);
}
