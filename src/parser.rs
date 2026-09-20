use crate::ast::{BinaryOp, Expr, Stmt};
use crate::lexer::Token;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn current(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> Token {
        let t = self.current().clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn match_token(&mut self, token: Token) -> bool {
        if self.current() == &token {
            self.advance();
            true
        } else {
            false
        }
    }

    pub fn parse(&mut self) -> Result<Vec<Stmt>, String> {
        let mut statements = Vec::new();
        while self.current() != &Token::EOF {
            statements.push(self.parse_statement()?);
        }
        Ok(statements)
    }

    fn parse_statement(&mut self) -> Result<Stmt, String> {
        match self.current() {
            Token::Let => {
                self.advance();
                let name = match self.advance() {
                    Token::Identifier(id) => id,
                    other => return Err(format!("LUM-P2001: Expected identifier after let, found {:?}", other)),
                };
                if !self.match_token(Token::Equal) {
                    return Err("LUM-P2002: Expected '=' after identifier".into());
                }
                let expr = self.parse_expr()?;
                self.match_token(Token::Semicolon);
                Ok(Stmt::Let(name, expr))
            }
            Token::Print => {
                self.advance();
                let has_paren = self.match_token(Token::LParen);
                let expr = self.parse_expr()?;
                if has_paren {
                    if !self.match_token(Token::RParen) {
                        return Err("LUM-P2004: Expected ')' after print argument".into());
                    }
                }
                self.match_token(Token::Semicolon);
                Ok(Stmt::Print(expr))
            }
            Token::Import => {
                self.advance();
                let path = match self.advance() {
                    Token::Str(s) => s,
                    other => return Err(format!("LUM-P2003: Expected string path after import, found {:?}", other)),
                };
                self.match_token(Token::Semicolon);
                Ok(Stmt::Import(path))
            }
            _ => {
                let expr = self.parse_expr()?;
                self.match_token(Token::Semicolon);
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_equality()
    }

    fn parse_equality(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_addition()?;
        while self.match_token(Token::EqualEqual) {
            let right = self.parse_addition()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::Equal,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_addition(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_multiplication()?;
        while self.current() == &Token::Plus || self.current() == &Token::Minus {
            let op = if self.match_token(Token::Plus) {
                BinaryOp::Add
            } else {
                self.advance();
                BinaryOp::Sub
            };
            let right = self.parse_multiplication()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_multiplication(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_primary()?;
        while self.current() == &Token::Star || self.current() == &Token::Slash || self.current() == &Token::At {
            let op = if self.match_token(Token::Star) {
                BinaryOp::Mul
            } else if self.match_token(Token::Slash) {
                BinaryOp::Div
            } else {
                self.advance();
                BinaryOp::MatMul
            };
            let right = self.parse_primary()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.advance() {
            Token::Int(n) => Ok(Expr::Int(n)),
            Token::Float(f) => Ok(Expr::Float(f)),
            Token::Imaginary(im) => Ok(Expr::Imaginary(im)),
            Token::Str(s) => Ok(Expr::Str(s)),
            Token::Identifier(id) => Ok(Expr::Variable(id)),
            Token::KbInit => {
                if !self.match_token(Token::LParen) { return Err("LUM-P3001: Expected '(' after kb_init".into()); }
                if !self.match_token(Token::RParen) { return Err("LUM-P3002: Expected ')' in kb_init()".into()); }
                Ok(Expr::KbInit)
            }
            Token::KbFact => {
                if !self.match_token(Token::LParen) { return Err("LUM-P3003: Expected '(' after kb_fact".into()); }
                let kb = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P3004: Expected ',' after kb".into()); }
                let fact = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P3005: Expected ')' after kb_fact args".into()); }
                Ok(Expr::KbFact { kb: Box::new(kb), fact: Box::new(fact) })
            }
            Token::KbRule => {
                if !self.match_token(Token::LParen) { return Err("LUM-P3006: Expected '(' after kb_rule".into()); }
                let kb = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P3007: Expected ',' after kb".into()); }
                let head = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P3008: Expected ',' after rule head".into()); }
                let body = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P3009: Expected ')' after kb_rule args".into()); }
                Ok(Expr::KbRule { kb: Box::new(kb), head: Box::new(head), body: Box::new(body) })
            }
            Token::KbProve => {
                if !self.match_token(Token::LParen) { return Err("LUM-P3010: Expected '(' after kb_prove".into()); }
                let kb = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P3011: Expected ',' after kb".into()); }
                let query = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P3012: Expected ')' after kb_prove args".into()); }
                Ok(Expr::KbProve { kb: Box::new(kb), query: Box::new(query) })
            }
            Token::Conj => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2007: Expected '(' after conj".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Conj(Box::new(inner)))
            }
            Token::Intensity => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2008: Expected '(' after intensity".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Intensity(Box::new(inner)))
            }
            Token::Dft => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2009: Expected '(' after dft".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Dft(Box::new(inner)))
            }
            Token::Idft => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2010: Expected '(' after idft".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Idft(Box::new(inner)))
            }
            Token::Propagate => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2011: Expected '(' after propagate2d".into()); }
                let field = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P2012: Expected ',' after field".into()); }
                let z = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P2012: Expected ',' after z distance".into()); }
                let wavelength = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P2012: Expected ',' after wavelength".into()); }
                let dx = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Propagate2D {
                    field: Box::new(field),
                    z: Box::new(z),
                    wavelength: Box::new(wavelength),
                    dx: Box::new(dx),
                })
            }
            Token::Fdtd2D => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2013: Expected '(' after fdtd2d".into()); }
                let steps = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P2012: Expected ',' after steps".into()); }
                let size = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Fdtd2D {
                    steps: Box::new(steps),
                    size: Box::new(size),
                })
            }
            Token::QState => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2014: Expected '(' after qstate".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::QState(Box::new(inner)))
            }
            Token::BeamSplitter => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2015: Expected '(' after beamsplitter".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::BeamSplitter(Box::new(inner)))
            }
            Token::LParen => {
                let expr = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(expr)
            }
            Token::LBracket => {
                let mut elements = Vec::new();
                if self.current() != &Token::RBracket {
                    loop {
                        elements.push(self.parse_expr()?);
                        if self.current() == &Token::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                if !self.match_token(Token::RBracket) {
                    return Err("LUM-P2006: Expected ']' after array elements".into());
                }
                Ok(Expr::Array(elements))
            }
            other => Err(format!("LUM-P2005: Unexpected token {:?}", other)),
        }
    }
}
