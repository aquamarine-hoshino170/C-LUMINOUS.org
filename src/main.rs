use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Instance {
        struct_name: String,
        fields: HashMap<String, Value>,
    },
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Str(s) => write!(f, "{}", s),
            Value::List(l) => {
                let items: Vec<String> = l.iter().map(|v| format!("{}", v)).collect();
                write!(f, "[{}]", items.join(", "))
            }
            Value::Instance { struct_name, fields } => {
                let mut field_strs = Vec::new();
                for (k, v) in fields {
                    field_strs.push(format!("{}: {}", k, v));
                }
                write!(f, "{} {{ {} }}", struct_name, field_strs.join(", "))
            }
        }
    }
}

impl Value {
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Str(s) => s.trim().parse::<f64>().ok(),
            _ => None,
        }
    }

    pub fn to_json_string(&self) -> String {
        match self {
            Value::Nil => "null".to_string(),
            Value::Bool(b) => format!("{}", b),
            Value::Int(i) => format!("{}", i),
            Value::Float(fl) => format!("{}", fl),
            Value::Str(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            Value::List(l) => {
                let items: Vec<String> = l.iter().map(|v| v.to_json_string()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Instance { fields, .. } => {
                let mut entries = Vec::new();
                for (k, v) in fields {
                    entries.push(format!("\"{}\": {}", k, v.to_json_string()));
                }
                format!("{{{}}}", entries.join(", "))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Flow {
    Next,
    Break,
    Continue,
    Return(Value),
}

#[derive(Debug, Clone)]
pub struct Function {
    pub params: Vec<String>,
    pub body: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub methods: HashMap<String, Function>,
}

#[derive(Debug, Clone)]
pub struct DiagnosticError {
    pub err_type: String,
    pub message: String,
    pub line_num: usize,
    pub col_num: usize,
    pub line_content: String,
    pub suggestion: Option<String>,
}

pub struct Runtime {
    pub globals: HashMap<String, Value>,
    pub structs: HashMap<String, StructDef>,
    pub functions: HashMap<String, Function>,
    pub call_stack: Vec<HashMap<String, Value>>,
    pub imported_files: HashSet<String>,
    pub last_diag: Option<DiagnosticError>,
    pub current_filename: String,
}

impl Runtime {
    pub fn new(filename: &str) -> Self {
        Runtime {
            globals: HashMap::new(),
            structs: HashMap::new(),
            functions: HashMap::new(),
            call_stack: Vec::new(),
            imported_files: HashSet::new(),
            last_diag: None,
            current_filename: filename.to_string(),
        }
    }

    pub fn report_error(&mut self, err_type: &str, msg: &str, line_num: usize, col: usize, line_content: &str, suggestion: Option<&str>) {
        self.last_diag = Some(DiagnosticError {
            err_type: err_type.to_string(),
            message: msg.to_string(),
            line_num,
            col_num: col,
            line_content: line_content.to_string(),
            suggestion: suggestion.map(|s| s.to_string()),
        });
        self.globals.insert("__lum_last_err__".to_string(), Value::Str(msg.to_string()));
    }

    pub fn print_diagnostics(&self) {
        if let Some(ref d) = self.last_diag {
            eprintln!("\x1b[1;31merror[{}]\x1b[0m: \x1b[1m{}\x1b[0m", d.err_type, d.message);
            eprintln!("  \x1b[1;34m-->\x1b[0m {}:{}:{}", self.current_filename, d.line_num, d.col_num);
            eprintln!("   \x1b[1;34m|\x1b[0m\n\x1b[1;34m{:>3} |\x1b[0m {}", d.line_num, d.line_content);
            let spaces = " ".repeat(d.col_num.saturating_sub(1));
            eprintln!("   \x1b[1;34m|\x1b[0m {}\x1b[1;31m^\x1b[0m \x1b[1;31m{}\x1b[0m", spaces, d.message);
            if let Some(ref sugg) = d.suggestion {
                eprintln!("   \x1b[1;34m|\x1b[0m\n   \x1b[1;32m= help:\x1b[0m {}", sugg);
            }
            eprintln!();
        }
    }

    pub fn get_var(&self, name: &str) -> Value {
        for scope in self.call_stack.iter().rev() {
            if let Some(val) = scope.get(name) {
                return val.clone();
            }
        }
        if let Some(val) = self.globals.get(name) {
            return val.clone();
        }
        Value::Nil
    }

    pub fn set_var(&mut self, name: &str, val: Value) {
        for scope in self.call_stack.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), val);
                return;
            }
        }
        if let Some(scope) = self.call_stack.last_mut() {
            scope.insert(name.to_string(), val);
        } else {
            self.globals.insert(name.to_string(), val);
        }
    }

    pub fn is_truthy(&self, val: &Value) -> bool {
        match val {
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::Nil => false,
            _ => true,
        }
    }

    pub fn execute_block(&mut self, lines: &[String]) -> Flow {
        let mut i = 0;
        while i < lines.len() {
            if self.globals.contains_key("__lum_last_err__") {
                return Flow::Next;
            }

            let raw_line = &lines[i];
            let line = raw_line.trim();

            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                i += 1;
                continue;
            }

            // 1. Module Import
            if line.starts_with("import ") {
                let mut path_str = line[7..].trim();
                if (path_str.starts_with('"') && path_str.ends_with('"')) || (path_str.starts_with('\'') && path_str.ends_with('\'')) {
                    path_str = &path_str[1..path_str.len() - 1];
                }
                let target_path = if Path::new(&self.current_filename).parent().is_some() {
                    Path::new(&self.current_filename).parent().unwrap().join(path_str).to_string_lossy().to_string()
                } else {
                    path_str.to_string()
                };

                if !self.imported_files.contains(&target_path) {
                    self.imported_files.insert(target_path.clone());
                    match fs::read_to_string(&target_path) {
                        Ok(content) => {
                            let old_fn = self.current_filename.clone();
                            self.current_filename = target_path;
                            let import_lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
                            let flow = self.execute_block(&import_lines);
                            self.current_filename = old_fn;
                            if let Flow::Return(v) = flow { return Flow::Return(v); }
                        }
                        Err(e) => {
                            self.report_error("ImportError", &format!("cannot import module '{}': {}", path_str, e), i + 1, 1, raw_line, Some("Verify target file path exists"));
                            return Flow::Next;
                        }
                    }
                }
                i += 1;
                continue;
            }

            // 2. Loop Controls
            if line == "break" { return Flow::Break; }
            if line == "continue" { return Flow::Continue; }

            // 3. Return Statement
            if line.starts_with("return ") || line == "return" {
                let expr_part = if line == "return" { "" } else { line[7..].trim() };
                let val = if expr_part.is_empty() { Value::Nil } else { self.eval_expr(expr_part, i + 1, raw_line) };
                return Flow::Return(val);
            }

            // 4. Print Statement
            if line.starts_with("print(") && line.ends_with(')') {
                let inner = line[6..line.len() - 1].trim();
                let val = self.eval_expr(inner, i + 1, raw_line);
                if !self.globals.contains_key("__lum_last_err__") { println!("{}", val); }
                i += 1;
                continue;
            }

            // 5. Global Function Definition
            if line.starts_with("fn ") && line.ends_with('{') {
                let paren_open = line.find('(').unwrap();
                let paren_close = line.find(')').unwrap();
                let fn_name = line[3..paren_open].trim().to_string();
                let params_str = &line[paren_open + 1..paren_close];
                let params: Vec<String> = params_str.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();

                let mut fn_body = Vec::new();
                i += 1;
                let mut fdepth = 1;
                while i < lines.len() && fdepth > 0 {
                    let fl = lines[i].trim();
                    if fl == "}" {
                        fdepth -= 1;
                        if fdepth == 0 { i += 1; break; }
                    } else if fl.ends_with('{') {
                        fdepth += 1;
                    }
                    fn_body.push(lines[i].clone());
                    i += 1;
                }
                self.functions.insert(fn_name, Function { params, body: fn_body });
                continue;
            }

            // 6. If Statement Handler (Proper Block Skip on True / False)
            if line.starts_with("if ") && line.ends_with('{') {
                let cond_str = line[3..line.len() - 1].trim();
                let cond_val = self.eval_expr(cond_str, i + 1, raw_line);

                let mut if_body = Vec::new();
                i += 1;
                let mut depth = 1;
                while i < lines.len() && depth > 0 {
                    let l = lines[i].trim();
                    if l == "}" {
                        depth -= 1;
                        if depth == 0 { i += 1; break; }
                    } else if l.ends_with('{') {
                        depth += 1;
                    }
                    if_body.push(lines[i].clone());
                    i += 1;
                }

                if self.is_truthy(&cond_val) {
                    let flow = self.execute_block(&if_body);
                    if flow != Flow::Next { return flow; }
                }
                continue;
            }

            // 7. Assignment Statement
            let parsed_assignment = if line.starts_with("today ") && line.contains(" is ") {
                let rest = line[6..].trim();
                let is_idx = rest.find(" is ").unwrap();
                let var_name = rest[..is_idx].trim();
                let expr_part = rest[is_idx + 4..].trim();
                Some((var_name, expr_part))
            } else if let Some(eq_idx) = line.find('=') {
                if !line[..eq_idx].ends_with('!') && !line[..eq_idx].ends_with('=') && !line[..eq_idx].ends_with('<') && !line[..eq_idx].ends_with('>') && line.chars().nth(eq_idx + 1) != Some('=') {
                    Some((line[..eq_idx].trim(), line[eq_idx + 1..].trim()))
                } else {
                    None
                }
            } else {
                None
            };

            if let Some((var_name, expr_part)) = parsed_assignment {
                let val = self.eval_expr(expr_part, i + 1, raw_line);
                if self.globals.contains_key("__lum_last_err__") { return Flow::Next; }
                self.set_var(var_name, val);
                i += 1;
                continue;
            }

            // 8. Standalone Expression
            let res = self.eval_expr(line, i + 1, raw_line);
            if self.globals.contains_key("__lum_last_err__") { return Flow::Next; }
            if self.globals.contains_key("__lum_repl_mode__") && res != Value::Nil { println!("{}", res); }
            i += 1;
        }
        Flow::Next
    }

    pub fn eval_expr(&mut self, expr: &str, line_num: usize, raw_line: &str) -> Value {
        let mut parser = ExprParser::new(expr, line_num, raw_line);
        parser.parse_expr(self)
    }
}

pub struct ExprParser<'a> {
    pub chars: Vec<char>,
    pub pos: usize,
    pub line_num: usize,
    pub raw_line: String,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> ExprParser<'a> {
    pub fn new(input: &str, line_num: usize, raw_line: &str) -> Self {
        ExprParser { chars: input.chars().collect(), pos: 0, line_num, raw_line: raw_line.to_string(), _marker: std::marker::PhantomData }
    }
    fn skip_whitespace(&mut self) { while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() { self.pos += 1; } }
    pub fn parse_expr(&mut self, rt: &mut Runtime) -> Value { self.parse_equality(rt) }

    fn parse_equality(&mut self, rt: &mut Runtime) -> Value {
        let mut left = self.parse_additive(rt);
        self.skip_whitespace();
        while self.pos + 1 < self.chars.len() {
            if self.chars[self.pos] == '=' && self.chars[self.pos + 1] == '=' {
                self.pos += 2;
                let right = self.parse_additive(rt);
                left = Value::Bool(match (&left, &right) {
                    (Value::Str(a), Value::Str(b)) => a == b,
                    _ => match (left.to_f64(), right.to_f64()) {
                        (Some(a), Some(b)) => (a - b).abs() < f64::EPSILON,
                        _ => left == right,
                    },
                });
                self.skip_whitespace();
            } else if self.chars[self.pos] == '<' {
                self.pos += 1;
                let right = self.parse_additive(rt);
                left = Value::Bool(match (left.to_f64(), right.to_f64()) {
                    (Some(a), Some(b)) => a < b,
                    _ => false,
                });
                self.skip_whitespace();
            } else if self.chars[self.pos] == '>' {
                self.pos += 1;
                let right = self.parse_additive(rt);
                left = Value::Bool(match (left.to_f64(), right.to_f64()) {
                    (Some(a), Some(b)) => a > b,
                    _ => false,
                });
                self.skip_whitespace();
            } else {
                break;
            }
        }
        left
    }

    fn parse_additive(&mut self, rt: &mut Runtime) -> Value {
        let mut left = self.parse_multiplicative(rt);
        self.skip_whitespace();
        while self.pos < self.chars.len() {
            let op = self.chars[self.pos];
            if op == '+' || op == '-' {
                self.pos += 1;
                let right = self.parse_multiplicative(rt);
                left = match (&left, &right) {
                    (Value::Str(a), b) if op == '+' => Value::Str(format!("{}{}", a, b)),
                    (a, Value::Str(b)) if op == '+' => Value::Str(format!("{}{}", a, b)),
                    _ => {
                        if let (Some(na), Some(nb)) = (left.to_f64(), right.to_f64()) {
                            let res = if op == '+' { na + nb } else { na - nb };
                            if res.fract() == 0.0 { Value::Int(res as i64) } else { Value::Float(res) }
                        } else {
                            Value::Nil
                        }
                    }
                };
                self.skip_whitespace();
            } else {
                break;
            }
        }
        left
    }

    fn parse_multiplicative(&mut self, rt: &mut Runtime) -> Value {
        let mut left = self.parse_postfix(rt);
        self.skip_whitespace();
        while self.pos < self.chars.len() {
            let op = self.chars[self.pos];
            if op == '*' || op == '/' || op == '%' {
                self.pos += 1;
                let right = self.parse_postfix(rt);
                left = {
                    if let (Some(na), Some(nb)) = (left.to_f64(), right.to_f64()) {
                        if op == '*' {
                            let res = na * nb;
                            if res.fract() == 0.0 { Value::Int(res as i64) } else { Value::Float(res) }
                        } else if op == '/' {
                            if nb == 0.0 {
                                rt.report_error("DivisionByZero", "attempt to divide by zero", self.line_num, self.pos, &self.raw_line, Some("Check divisor value"));
                                Value::Nil
                            } else {
                                let res = na / nb;
                                if res.fract() == 0.0 { Value::Int(res as i64) } else { Value::Float(res) }
                            }
                        } else {
                            Value::Float(na % nb)
                        }
                    } else {
                        Value::Nil
                    }
                };
                self.skip_whitespace();
            } else {
                break;
            }
        }
        left
    }

    fn parse_postfix(&mut self, rt: &mut Runtime) -> Value {
        let mut left = self.parse_primary(rt);
        loop {
            if rt.globals.contains_key("__lum_last_err__") { break; }
            self.skip_whitespace();

            if self.pos < self.chars.len() && self.chars[self.pos] == '[' {
                self.pos += 1;
                let idx_val = self.parse_expr(rt);
                self.skip_whitespace();
                if self.pos < self.chars.len() && self.chars[self.pos] == ']' { self.pos += 1; }
                if let Value::List(ref lst) = left {
                    if let Some(idx) = idx_val.to_f64() {
                        let i = idx as usize;
                        if i < lst.len() {
                            left = lst[i].clone();
                            continue;
                        }
                    }
                }
                left = Value::Nil;
                continue;
            }
            break;
        }
        left
    }

    fn parse_primary(&mut self, rt: &mut Runtime) -> Value {
        self.skip_whitespace();
        if self.pos >= self.chars.len() { return Value::Nil; }

        if self.chars[self.pos] == '-' && self.pos + 1 < self.chars.len() && (self.chars[self.pos + 1].is_ascii_digit() || self.chars[self.pos + 1] == '.') {
            self.pos += 1;
            let mut num_str = String::from("-");
            let mut is_float = false;
            while self.pos < self.chars.len() && (self.chars[self.pos].is_ascii_digit() || self.chars[self.pos] == '.') {
                if self.chars[self.pos] == '.' { if is_float { break; } is_float = true; }
                num_str.push(self.chars[self.pos]);
                self.pos += 1;
            }
            return if is_float { Value::Float(num_str.parse::<f64>().unwrap_or(0.0)) } else { Value::Int(num_str.parse::<i64>().unwrap_or(0)) };
        }
        if self.chars[self.pos].is_ascii_digit() {
            let mut num_str = String::new();
            let mut is_float = false;
            while self.pos < self.chars.len() && (self.chars[self.pos].is_ascii_digit() || self.chars[self.pos] == '.') {
                if self.chars[self.pos] == '.' { if is_float { break; } is_float = true; }
                num_str.push(self.chars[self.pos]);
                self.pos += 1;
            }
            return if is_float { Value::Float(num_str.parse::<f64>().unwrap_or(0.0)) } else { Value::Int(num_str.parse::<i64>().unwrap_or(0)) };
        }
        if self.chars[self.pos] == '[' {
            self.pos += 1;
            let mut list = Vec::new();
            self.skip_whitespace();
            while self.pos < self.chars.len() && self.chars[self.pos] != ']' {
                list.push(self.parse_expr(rt));
                self.skip_whitespace();
                if self.pos < self.chars.len() && self.chars[self.pos] == ',' { self.pos += 1; self.skip_whitespace(); }
            }
            if self.pos < self.chars.len() && self.chars[self.pos] == ']' { self.pos += 1; }
            return Value::List(list);
        }
        if self.chars[self.pos] == '"' || self.chars[self.pos] == '\'' {
            let quote = self.chars[self.pos];
            self.pos += 1;
            let mut s = String::new();
            while self.pos < self.chars.len() && self.chars[self.pos] != quote {
                if self.chars[self.pos] == '\\' && self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == 'n' {
                    s.push('\n'); self.pos += 2;
                } else if self.chars[self.pos] == '$' && self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == '{' {
                    self.pos += 2;
                    let mut expr_buf = String::new();
                    let mut depth = 1;
                    while self.pos < self.chars.len() && depth > 0 {
                        if self.chars[self.pos] == '{' { depth += 1; } else if self.chars[self.pos] == '}' { depth -= 1; }
                        if depth == 0 { self.pos += 1; break; }
                        expr_buf.push(self.chars[self.pos]);
                        self.pos += 1;
                    }
                    let res = rt.eval_expr(expr_buf.trim(), self.line_num, &self.raw_line);
                    s.push_str(&format!("{}", res));
                } else {
                    s.push(self.chars[self.pos]);
                    self.pos += 1;
                }
            }
            if self.pos < self.chars.len() && self.chars[self.pos] == quote { self.pos += 1; }
            return Value::Str(s);
        }

        if self.chars[self.pos].is_alphabetic() || self.chars[self.pos] == '_' {
            let mut id = String::new();
            while self.pos < self.chars.len() && (self.chars[self.pos].is_alphanumeric() || self.chars[self.pos] == '_') {
                id.push(self.chars[self.pos]);
                self.pos += 1;
            }

            self.skip_whitespace();
            if self.pos < self.chars.len() && self.chars[self.pos] == '(' {
                self.pos += 1;
                let mut args = Vec::new();
                self.skip_whitespace();
                if self.pos < self.chars.len() && self.chars[self.pos] != ')' {
                    loop {
                        args.push(self.parse_expr(rt));
                        self.skip_whitespace();
                        if self.pos < self.chars.len() && self.chars[self.pos] == ',' { self.pos += 1; self.skip_whitespace(); } else { break; }
                    }
                }
                if self.pos < self.chars.len() && self.chars[self.pos] == ')' { self.pos += 1; }

                match id.as_str() {
                    "httpServe" => {
                        let port = match args.first().unwrap_or(&Value::Nil) {
                            Value::Int(p) => *p,
                            Value::Float(f) => *f as i64,
                            _ => 8008,
                        };
                        let handler_fn_name = match args.get(1) {
                            Some(Value::Str(s)) => s.clone(),
                            _ => "handleRequest".to_string(),
                        };

                        let addr = format!("127.0.0.1:{}", port);
                        println!("\x1b[1;32m🚀 [Luminous HTTP Engine] Server listening on http://{}\x1b[0m", addr);

                        match TcpListener::bind(&addr) {
                            Ok(listener) => {
                                for stream_res in listener.incoming() {
                                    if let Ok(mut stream) = stream_res {
                                        let mut buffer = [0; 2048];
                                        let bytes_read = stream.read(&mut buffer).unwrap_or(0);
                                        let request_text = String::from_utf8_lossy(&buffer[..bytes_read]);

                                        let mut req_path = "/".to_string();
                                        if let Some(first_line) = request_text.lines().next() {
                                            let parts: Vec<&str> = first_line.split_whitespace().collect();
                                            if parts.len() >= 2 {
                                                req_path = parts[1].to_string();
                                            }
                                        }

                                        let mut body_content = format!("<h1>Welcome to Luminous HTTP Server!</h1><p>Route: {}</p>", req_path);

                                        // Isolated Call-Stack Execution
                                        if let Some(func) = rt.functions.get(&handler_fn_name).cloned() {
                                            let mut local_scope = HashMap::new();
                                            if let Some(p) = func.params.first() {
                                                local_scope.insert(p.clone(), Value::Str(req_path.clone()));
                                            }
                                            rt.call_stack.push(local_scope);
                                            let flow = rt.execute_block(&func.body);
                                            rt.call_stack.pop();
                                            if let Flow::Return(v) = flow {
                                                body_content = format!("{}", v);
                                            }
                                        }

                                        let is_json = body_content.trim().starts_with('{') || body_content.trim().starts_with('[');
                                        let content_type = if is_json { "application/json" } else { "text/html" };

                                        let response = format!(
                                            "HTTP/1.1 200 OK\r\nContent-Type: {}; charset=UTF-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                            content_type, body_content.len(), body_content
                                        );

                                        let _ = stream.write_all(response.as_bytes());
                                        let _ = stream.flush();
                                    }
                                }
                            }
                            Err(e) => {
                                rt.report_error("NetworkError", &format!("Failed to bind port {}: {}", port, e), self.line_num, self.pos, &self.raw_line, Some("Try using a different port"));
                            }
                        }
                        return Value::Nil;
                    }
                    "jsonEncode" => return Value::Str(args.first().unwrap_or(&Value::Nil).to_json_string()),
                    _ => {}
                }

                if let Some(func) = rt.functions.get(&id).cloned() {
                    let mut local_scope = HashMap::new();
                    for (idx, p) in func.params.iter().enumerate() {
                        if idx < args.len() { local_scope.insert(p.clone(), args[idx].clone()); }
                    }
                    rt.call_stack.push(local_scope);
                    let flow = rt.execute_block(&func.body);
                    rt.call_stack.pop();
                    if let Flow::Return(v) = flow { return v; }
                    return Value::Nil;
                }
            }

            if id == "true" { return Value::Bool(true); }
            if id == "false" { return Value::Bool(false); }
            if id == "nil" { return Value::Nil; }

            return rt.get_var(&id);
        }
        Value::Nil
    }
}

fn main() {
    let raw_args: Vec<String> = env::args().collect();
    if raw_args.len() < 2 { return; }

    let filename = &raw_args[1];
    let content = match fs::read_to_string(filename) {
        Ok(c) => c,
        Err(e) => { eprintln!("Error: {}", e); return; }
    };

    let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let mut runtime = Runtime::new(filename);

    let cli_list: Vec<Value> = raw_args[1..].iter().map(|s| Value::Str(s.clone())).collect();
    runtime.globals.insert("args".to_string(), Value::List(cli_list));

    runtime.execute_block(&lines);
    if runtime.globals.contains_key("__lum_last_err__") { runtime.print_diagnostics(); }
}
