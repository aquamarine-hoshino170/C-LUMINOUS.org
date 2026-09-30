import re

with open("src/main.rs", "r") as f:
    code = f.read()

# 1. Mathematically Pure Raw String (r"") for Interpolation
str_impl = r"""// 2. String literal with rigorous ${expr} substitution
        if self.pos < self.chars.len() && (self.chars[self.pos] == '"' || self.chars[self.pos] == '\'') {
            let quote = self.chars[self.pos];
            self.pos += 1;
            let mut s = String::new();
            while self.pos < self.chars.len() && self.chars[self.pos] != quote {
                if self.chars[self.pos] == '\\' && self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == 'n' {
                    s.push('\n');
                    self.pos += 2;
                } else if self.chars[self.pos] == '$' && self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == '{' {
                    self.pos += 2;
                    let mut expr_buf = String::new();
                    let mut d = 1;
                    while self.pos < self.chars.len() && d > 0 {
                        if self.chars[self.pos] == '{' { d += 1; }
                        else if self.chars[self.pos] == '}' { d -= 1; }
                        if d == 0 { self.pos += 1; break; }
                        expr_buf.push(self.chars[self.pos]);
                        self.pos += 1;
                    }
                    let res = rt.eval_expr(expr_buf.trim());
                    s.push_str(&format!("{}", res));
                } else {
                    s.push(self.chars[self.pos]);
                    self.pos += 1;
                }
            }
            if self.pos < self.chars.len() && self.chars[self.pos] == quote {
                self.pos += 1;
            }
            return Value::Str(s);
        }"""

code = re.sub(r"// 2\. String literal[\s\S]+?return Value::Str\(s\);\s*\}", lambda _: str_impl, code)

# 2. Extract and strictly replace the Try-Catch state machine domain
start_idx = code.find("// 4.4 Throw")
if start_idx == -1:
    start_idx = code.find("// 4.5 Try-Catch")

if start_idx != -1:
    end_idx = code.find("continue;\n            }", start_idx)
    if end_idx != -1:
        end_idx += len("continue;\n            }")
        
        tc_impl = r"""// 4.4 Throw Statement
            if line.starts_with("throw ") {
                let err_expr = line[6..].trim();
                let evaluated_err = self.eval_expr(err_expr);
                self.globals.insert("__lum_last_err__".to_string(), Value::Str(format!("{}", evaluated_err)));
                return (Value::Nil, true);
            }

            // 4.5 Try-Catch Statement: try { ... } catch (err) { ... }
            if line.starts_with("try") && line.ends_with("{") {
                let mut try_body = Vec::new();
                i += 1;
                let mut depth = 1;
                while i < lines.len() && depth > 0 {
                    let l = lines[i].trim();
                    depth += l.matches("{").count();
                    depth -= l.matches("}").count();
                    if depth == 0 { break; }
                    try_body.push(lines[i].clone());
                    i += 1;
                }
                i += 1; // Move past try block closing brace

                let mut err_var = "err".to_string();
                let mut catch_body = Vec::new();

                while i < lines.len() && lines[i].trim().is_empty() {
                    i += 1;
                }

                if i < lines.len() && lines[i].trim().starts_with("catch") && lines[i].trim().ends_with("{") {
                    let next_l = lines[i].trim();
                    let header = next_l["catch".len()..next_l.len()-1].trim();
                    if header.starts_with("(") && header.ends_with(")") {
                        err_var = header[1..header.len()-1].trim().to_string();
                    } else if !header.is_empty() {
                        err_var = header.to_string();
                    }

                    let mut cdepth = 1;
                    i += 1;
                    while i < lines.len() && cdepth > 0 {
                        let cl = lines[i].trim();
                        cdepth += cl.matches("{").count();
                        cdepth -= cl.matches("}").count();
                        if cdepth == 0 { break; }
                        catch_body.push(lines[i].clone());
                        i += 1;
                    }
                    i += 1; // Move past catch block closing brace
                }

                // Execute block purely
                let (val, returned) = self.execute_block(&try_body);

                if let Some(err_val) = self.globals.remove("__lum_last_err__") {
                    let err_msg = format!("{}", err_val);
                    if !catch_body.is_empty() {
                        let mut catch_scope = std::collections::HashMap::new();
                        catch_scope.insert(err_var, Value::Str(err_msg));
                        self.call_stack.push(catch_scope);
                        let (cval, creturned) = self.execute_block(&catch_body);
                        self.call_stack.pop();
                        if creturned { return (cval, true); }
                    }
                } else if returned {
                    return (val, true);
                }
                continue;
            }"""
        
        # Slicing the exact indices geometrically
        code = code[:start_idx] + tc_impl + code[end_idx:]

with open("src/main.rs", "w") as f:
    f.write(code)

print("✅ Pure Mathematical Raw String Logic Applied!")
