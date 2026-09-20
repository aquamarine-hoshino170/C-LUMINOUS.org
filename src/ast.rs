#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Imaginary(f64),
    Str(String),
    Variable(String),
    Array(Vec<Expr>),
    Conj(Box<Expr>),
    Intensity(Box<Expr>),
    Dft(Box<Expr>),
    Idft(Box<Expr>),
    Transpose(Box<Expr>),
    Slice {
        target: Box<Expr>,
        start: Box<Expr>,
        end: Box<Expr>,
    },
    Diff {
        expr: Box<Expr>,
        var: String,
    },
    Propagate2D {
        field: Box<Expr>,
        z: Box<Expr>,
        wavelength: Box<Expr>,
        dx: Box<Expr>,
    },
    Fdtd2D {
        steps: Box<Expr>,
        size: Box<Expr>,
    },
    QState(Box<Expr>),
    BeamSplitter(Box<Expr>),
    KbInit,
    KbFact {
        kb: Box<Expr>,
        fact: Box<Expr>,
    },
    KbRule {
        kb: Box<Expr>,
        head: Box<Expr>,
        body: Box<Expr>,
    },
    KbProve {
        kb: Box<Expr>,
        query: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Equal,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Expr),
    Print(Expr),
    Import(String),
    Expr(Expr),
}
