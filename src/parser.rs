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
            Token::Fn => {
                self.advance(); // consume 'fn'
                let name = match self.advance() {
                    Token::Identifier(id) => id,
                    other => return Err(format!("LUM-FN01: Expected function name, found {:?}", other)),
                };

                // Expect '('
                if self.current() != &Token::LParen {
                    return Err(format!("LUM-FN02: Expected '(' after function name, found {:?}", self.current()));
                }
                self.advance(); // consume '('

                let mut params = Vec::new();
                while self.current() != &Token::RParen {
                    match self.advance() {
                        Token::Identifier(p) => params.push(p),
                        other => return Err(format!("LUM-FN03: Expected parameter name, found {:?}", other)),
                    }
                    if self.current() == &Token::Comma {
                        self.advance(); // consume ','
                    } else if self.current() == &Token::RParen {
                        break;
                    } else {
                        return Err(format!("LUM-FN03B: Expected ',' or ')' in parameter list, found {:?}", self.current()));
                    }
                }

                // Expect ')'
                if self.current() != &Token::RParen {
                    return Err(format!("LUM-FN04: Expected ')' after parameters, found {:?}", self.current()));
                }
                self.advance(); // consume ')'

                // Parse block body
                let body = Box::new(self.parse_statement()?);
                Ok(Stmt::FnDef { name, params, body })
            }
            Token::Return => {
                self.advance();
                let expr = if self.current() != &Token::Semicolon {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                self.match_token(Token::Semicolon);
                Ok(Stmt::Return(expr))
            }
            Token::If => {
                self.advance();
                let cond = self.parse_expr()?;
                let then_branch = Box::new(self.parse_statement()?);
                let mut else_branch = None;
                if self.match_token(Token::Else) {
                    else_branch = Some(Box::new(self.parse_statement()?));
                }
                Ok(Stmt::If { cond, then_branch, else_branch })
            }
            Token::Send => {
                self.advance();
                let expr = self.parse_expr()?;
                self.match_token(Token::Semicolon);
                Ok(Stmt::Send(expr))
            }
            Token::Spawn => {
                self.advance();
                let body = Box::new(self.parse_statement()?);
                Ok(Stmt::Spawn(body))
            }
            Token::While => {
                self.advance();
                let cond = self.parse_expr()?;
                let body = Box::new(self.parse_statement()?);
                Ok(Stmt::While { cond, body })
            }
            Token::LBrace => {
                self.advance();
                let mut stmts = Vec::new();
                while self.current() != &Token::RBrace && self.current() != &Token::EOF {
                    stmts.push(self.parse_statement()?);
                }
                if !self.match_token(Token::RBrace) {
                    return Err("LUM-P3000: Expected '}' to close block".into());
                }
                Ok(Stmt::Block(stmts))
            }
            Token::Diffuse => {
                self.advance();
                let prompt = self.parse_expr()?;
                if !self.match_token(Token::Steps) {
                    return Err("LUM-DIFF01: Expected 'steps' in diffuse statement".into());
                }
                let steps = self.parse_expr()?;
                if !self.match_token(Token::Guidance) {
                    return Err("LUM-DIFF02: Expected 'guidance' in diffuse statement".into());
                }
                let guidance = self.parse_expr()?;
                if !self.match_token(Token::To) {
                    return Err("LUM-DIFF03: Expected 'to' target".into());
                }
                let target_path = self.parse_expr()?;
                self.match_token(Token::Semicolon);
                Ok(Stmt::DiffusePipeline { prompt, steps, guidance, target_path })
            }
            Token::Canvas => {
                self.advance();
                let width = self.parse_expr()?;
                if !self.match_token(Token::Comma) {
                    return Err("LUM-GRA01: Expected ',' after canvas width".into());
                }
                let height = self.parse_expr()?;
                let mut properties = Vec::new();
                if self.match_token(Token::LBrace) {
                    while self.current() != &Token::RBrace && self.current() != &Token::EOF {
                        let key = match self.advance() {
                            Token::Background => "background".to_string(),
                            Token::Camera => "camera".to_string(),
                            Token::Shader => "shader".to_string(),
                            Token::Colormap => "colormap".to_string(),
                            Token::Identifier(id) => id,
                            other => return Err(format!("LUM-GRA02: Invalid config {:?}", other)),
                        };
                        if !self.match_token(Token::Colon) {
                            return Err("LUM-GRA03: Expected ':' after config key".into());
                        }
                        let val_expr = self.parse_expr()?;
                        self.match_token(Token::Semicolon);
                        properties.push((key, val_expr));
                    }
                    if !self.match_token(Token::RBrace) {
                        return Err("LUM-GRA04: Expected '}'".into());
                    }
                }
                self.match_token(Token::Semicolon);
                Ok(Stmt::CanvasDef { width, height, properties })
            }
            Token::Render => {
                self.advance();
                let source = self.parse_expr()?;
                let mut colormap = None;
                if self.match_token(Token::Using) {
                    let cmap_token = self.advance();
                    match cmap_token {
                        Token::Identifier(id) => colormap = Some(id),
                        Token::Str(s) => colormap = Some(s),
                        other => return Err(format!("LUM-GRA05: Expected colormap name, found {:?}", other)),
                    }
                }
                if !self.match_token(Token::To) {
                    return Err("LUM-GRA06: Expected 'to' in render".into());
                }
                let target_path = self.parse_expr()?;
                self.match_token(Token::Semicolon);
                Ok(Stmt::RenderPipeline { source, colormap, target_path })
            }
            Token::Let => {
                self.advance();
                let name = match self.advance() {
                    Token::Identifier(id) => id,
                    other => return Err(format!("LUM-P2001: Expected identifier, found {:?}", other)),
                };
                if !self.match_token(Token::Equal) {
                    return Err("LUM-P2002: Expected '='".into());
                }
                let expr = self.parse_expr()?;
                self.match_token(Token::Semicolon);
                Ok(Stmt::Let(name, expr))
            }
            Token::Print => {
                self.advance();
                let has_paren = self.match_token(Token::LParen);
                let expr = self.parse_expr()?;
                if has_paren && !self.match_token(Token::RParen) {
                    return Err("LUM-P2004: Expected ')'".into());
                }
                self.match_token(Token::Semicolon);
                Ok(Stmt::Print(expr))
            }
            Token::Import => {
                self.advance();
                let path = match self.advance() {
                    Token::Str(s) => s,
                    other => return Err(format!("LUM-P2003: Expected string path, found {:?}", other)),
                };
                self.match_token(Token::Semicolon);
                Ok(Stmt::Import(path))
            }
            Token::Identifier(id) => {
                if self.pos + 1 < self.tokens.len() && self.tokens[self.pos + 1] == Token::Equal {
                    let name = id.clone();
                    self.advance(); // consume ident
                    self.advance(); // consume '='
                    let expr = self.parse_expr()?;
                    self.match_token(Token::Semicolon);
                    return Ok(Stmt::Assign(name, expr));
                }
                let expr = self.parse_expr()?;
                self.match_token(Token::Semicolon);
                Ok(Stmt::Expr(expr))
            }
            _ => {
                let expr = self.parse_expr()?;
                self.match_token(Token::Semicolon);
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_addition()?;
        while matches!(self.current(), Token::EqualEqual | Token::Less | Token::Greater | Token::LessEqual | Token::GreaterEqual) {
            let op = match self.advance() {
                Token::EqualEqual => BinaryOp::Equal,
                Token::Less => BinaryOp::Less,
                Token::Greater => BinaryOp::Greater,
                Token::LessEqual => BinaryOp::LessEqual,
                Token::GreaterEqual => BinaryOp::GreaterEqual,
                _ => unreachable!(),
            };
            let right = self.parse_addition()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
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
            Token::Receive => Ok(Expr::Receive),
            Token::Int(n) => Ok(Expr::Int(n)),
            Token::Float(f) => Ok(Expr::Float(f)),
            Token::Imaginary(im) => Ok(Expr::Imaginary(im)),
            Token::Str(s) => Ok(Expr::Str(s)),
            Token::Identifier(id) => {
                if self.current() == &Token::LParen {
                    self.advance();
                    let mut args = Vec::new();
                    if self.current() != &Token::RParen {
                        loop {
                            args.push(self.parse_expr()?);
                            if !self.match_token(Token::Comma) { break; }
                        }
                    }
                    if !self.match_token(Token::RParen) {
                        return Err("LUM-FN05: Expected ')' after arguments".into());
                    }
                    Ok(Expr::Call { name: id, args })
                } else {
                    Ok(Expr::Variable(id))
                }
            }
            Token::Transpose => {
                if !self.match_token(Token::LParen) { return Err("LUM-P4001: Expected '('".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Transpose(Box::new(inner)))
            }
            Token::Slice => {
                if !self.match_token(Token::LParen) { return Err("LUM-P4002: Expected '('".into()); }
                let target = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P4003: Expected ','".into()); }
                let start = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P4003: Expected ','".into()); }
                let end = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Slice { target: Box::new(target), start: Box::new(start), end: Box::new(end) })
            }
            Token::Diff => {
                if !self.match_token(Token::LParen) { return Err("LUM-P4004: Expected '('".into()); }
                let expr = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P4005: Expected ','".into()); }
                let var = match self.advance() {
                    Token::Identifier(id) => id,
                    Token::Str(s) => s,
                    other => return Err(format!("LUM-P4006: Expected var, found {:?}", other)),
                };
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Diff { expr: Box::new(expr), var })
            }
            Token::KbInit => {
                if !self.match_token(Token::LParen) { return Err("LUM-P3001: Expected '('".into()); }
                if !self.match_token(Token::RParen) { return Err("LUM-P3002: Expected ')'".into()); }
                Ok(Expr::KbInit)
            }
            Token::KbFact => {
                if !self.match_token(Token::LParen) { return Err("LUM-P3003: Expected '('".into()); }
                let kb = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P3004: Expected ','".into()); }
                let fact = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P3005: Expected ')'".into()); }
                Ok(Expr::KbFact { kb: Box::new(kb), fact: Box::new(fact) })
            }
            Token::KbRule => {
                if !self.match_token(Token::LParen) { return Err("LUM-P3006: Expected '('".into()); }
                let kb = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P3007: Expected ','".into()); }
                let head = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P3008: Expected ','".into()); }
                let body = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P3009: Expected ')'".into()); }
                Ok(Expr::KbRule { kb: Box::new(kb), head: Box::new(head), body: Box::new(body) })
            }
            Token::KbProve => {
                if !self.match_token(Token::LParen) { return Err("LUM-P3010: Expected '('".into()); }
                let kb = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P3011: Expected ','".into()); }
                let query = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P3012: Expected ')'".into()); }
                Ok(Expr::KbProve { kb: Box::new(kb), query: Box::new(query) })
            }
            Token::Intensity => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2008: Expected '('".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Intensity(Box::new(inner)))
            }
            Token::Dft => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2009: Expected '('".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Dft(Box::new(inner)))
            }
            Token::Idft => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2010: Expected '('".into()); }
                let inner = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Idft(Box::new(inner)))
            }
            Token::Propagate => {
                if !self.match_token(Token::LParen) { return Err("LUM-P2011: Expected '('".into()); }
                let field = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P2012: Expected ','".into()); }
                let z = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P2012: Expected ','".into()); }
                let wavelength = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P2012: Expected ','".into()); }
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
                if !self.match_token(Token::LParen) { return Err("LUM-P2013: Expected '('".into()); }
                let steps = self.parse_expr()?;
                if !self.match_token(Token::Comma) { return Err("LUM-P2012: Expected ','".into()); }
                let size = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(Expr::Fdtd2D {
                    steps: Box::new(steps),
                    size: Box::new(size),
                })
            }
            Token::LParen => {
                let expr = self.parse_expr()?;
                if !self.match_token(Token::RParen) { return Err("LUM-P2004: Expected ')'".into()); }
                Ok(expr)
            }
            other => Err(format!("LUM-P2005: Unexpected token {:?}", other)),
        }
    }
}
