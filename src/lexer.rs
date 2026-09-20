#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Let,
    Fn,
    Print,
    Import,
    Return,
    Conj,
    Intensity,
    Dft,
    Idft,
    Propagate,
    Fdtd2D,
    QState,
    BeamSplitter,
    KbInit,
    KbFact,
    KbRule,
    KbProve,
    Identifier(String),
    Int(i64),
    Float(f64),
    Imaginary(f64),
    Str(String),
    Plus,
    Minus,
    Star,
    Slash,
    At,
    Equal,
    EqualEqual,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    EOF,
}

pub fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\r' | '\n' => {
                chars.next();
            }
            '/' => {
                chars.next(); // first slash
                if let Some(&'/') = chars.peek() {
                    chars.next(); // consume second slash
                    // single line comment: skip until newline
                    while let Some(&ch) = chars.peek() {
                        if ch == '\n' { break; }
                        chars.next();
                    }
                } else {
                    tokens.push(Token::Slash);
                }
            }
            '#' => {
                chars.next();
                while let Some(&ch) = chars.peek() {
                    if ch == '\n' { break; }
                    chars.next();
                }
            }
            '+' => { chars.next(); tokens.push(Token::Plus); }
            '-' => { chars.next(); tokens.push(Token::Minus); }
            '*' => { chars.next(); tokens.push(Token::Star); }
            '@' => { chars.next(); tokens.push(Token::At); }
            ';' => { chars.next(); tokens.push(Token::Semicolon); }
            ',' => { chars.next(); tokens.push(Token::Comma); }
            '(' => { chars.next(); tokens.push(Token::LParen); }
            ')' => { chars.next(); tokens.push(Token::RParen); }
            '{' => { chars.next(); tokens.push(Token::LBrace); }
            '}' => { chars.next(); tokens.push(Token::RBrace); }
            '[' => { chars.next(); tokens.push(Token::LBracket); }
            ']' => { chars.next(); tokens.push(Token::RBracket); }
            '=' => {
                chars.next();
                if let Some(&'=') = chars.peek() {
                    chars.next();
                    tokens.push(Token::EqualEqual);
                } else {
                    tokens.push(Token::Equal);
                }
            }
            '"' => {
                chars.next();
                let mut s = String::new();
                while let Some(&ch) = chars.peek() {
                    chars.next();
                    if ch == '"' { break; }
                    s.push(ch);
                }
                tokens.push(Token::Str(s));
            }
            '0'..='9' => {
                let mut num_str = String::new();
                let mut is_float = false;
                while let Some(&ch) = chars.peek() {
                    if ch.is_digit(10) {
                        num_str.push(ch);
                        chars.next();
                    } else if ch == '.' && !is_float {
                        is_float = true;
                        num_str.push(ch);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if let Some(&'i') = chars.peek() {
                    chars.next();
                    let val = num_str.parse::<f64>().unwrap_or(0.0);
                    tokens.push(Token::Imaginary(val));
                } else if is_float {
                    tokens.push(Token::Float(num_str.parse().unwrap_or(0.0)));
                } else {
                    tokens.push(Token::Int(num_str.parse().unwrap_or(0)));
                }
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut ident = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_alphanumeric() || ch == '_' {
                        ident.push(ch);
                        chars.next();
                    } else {
                        break;
                    }
                }
                match ident.as_str() {
                    "let" | "val" => tokens.push(Token::Let),
                    "fn" => tokens.push(Token::Fn),
                    "print" => tokens.push(Token::Print),
                    "import" => tokens.push(Token::Import),
                    "return" => tokens.push(Token::Return),
                    "conj" => tokens.push(Token::Conj),
                    "intensity" => tokens.push(Token::Intensity),
                    "dft" => tokens.push(Token::Dft),
                    "idft" => tokens.push(Token::Idft),
                    "propagate2d" => tokens.push(Token::Propagate),
                    "fdtd2d" => tokens.push(Token::Fdtd2D),
                    "qstate" => tokens.push(Token::QState),
                    "beamsplitter" => tokens.push(Token::BeamSplitter),
                    "kb_init" => tokens.push(Token::KbInit),
                    "kb_fact" => tokens.push(Token::KbFact),
                    "kb_rule" => tokens.push(Token::KbRule),
                    "kb_prove" => tokens.push(Token::KbProve),
                    "i" => tokens.push(Token::Imaginary(1.0)),
                    _ => tokens.push(Token::Identifier(ident)),
                }
            }
            _ => {
                chars.next();
            }
        }
    }
    tokens.push(Token::EOF);
    tokens
}
