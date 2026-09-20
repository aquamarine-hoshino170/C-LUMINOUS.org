use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub fn new(re: f64, im: f64) -> Self {
        Complex64 { re, im }
    }

    pub fn add(self, other: Self) -> Self {
        Complex64::new(self.re + other.re, self.im + other.im)
    }

    pub fn sub(self, other: Self) -> Self {
        Complex64::new(self.re - other.re, self.im - other.im)
    }

    pub fn mul(self, other: Self) -> Self {
        Complex64::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HornClause {
    pub head: String,
    pub body: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeBase {
    pub facts: HashSet<String>,
    pub rules: Vec<HornClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    #[allow(dead_code)]
    Nil,
    Int(i64),
    Float(f64),
    Complex(Complex64),
    Str(String),
    Bool(bool),
    Array(Vec<Value>),
    KbHandle(usize),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::Complex(c) => {
                if c.im >= 0.0 {
                    write!(f, "{}+{}i", c.re, c.im)
                } else {
                    write!(f, "{}{}i", c.re, c.im)
                }
            }
            Value::Str(s) => write!(f, "{}", s),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Array(items) => {
                write!(f, "[")?;
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::KbHandle(id) => write!(f, "<KnowledgeBase #{}>", id),
        }
    }
}

pub struct Environment {
    scopes: Vec<HashMap<String, Value>>,
    base_dir: PathBuf,
    pub kbs: Vec<KnowledgeBase>,
}

impl Environment {
    pub fn new(base_dir: PathBuf) -> Self {
        Environment {
            scopes: vec![HashMap::new()],
            base_dir,
            kbs: Vec::new(),
        }
    }

    pub fn set(&mut self, name: String, val: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, val);
        }
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(val.clone());
            }
        }
        None
    }

    pub fn resolve_module_path(&self, requested: &str) -> Result<PathBuf, String> {
        let candidate = self.base_dir.join(requested);
        let canonical = candidate.canonicalize().map_err(|e| {
            format!("LUM-M3001: Cannot resolve path '{}': {}", requested, e)
        })?;

        let canonical_base = self.base_dir.canonicalize().map_err(|e| {
            format!("LUM-M3002: Base directory error: {}", e)
        })?;

        if !canonical.starts_with(&canonical_base) {
            return Err(format!(
                "LUM-SEC01: Security violation! Path '{}' attempts to break out of base directory.",
                requested
            ));
        }

        Ok(canonical)
    }
}
