#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Imaginary(f64),
    Str(String),
    Variable(String),
    Call { name: String, args: Vec<Expr> },
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
    Receive,
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

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Equal,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    FnDef { name: String, params: Vec<String>, body: Box<Stmt> },
    Return(Option<Expr>),
    Let(String, Expr),
    Assign(String, Expr),
    Print(Expr),
    Import(String),
    Expr(Expr),
    Block(Vec<Stmt>),
    If {
        cond: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    Spawn(Box<Stmt>),
    Send(Expr),
    While {
        cond: Expr,
        body: Box<Stmt>,
    },
    CanvasDef {
        width: Expr,
        height: Expr,
        properties: Vec<(String, Expr)>,
    },
    RenderPipeline {
        source: Expr,
        colormap: Option<String>,
        target_path: Expr,
    },
    DiffusePipeline {
        prompt: Expr,
        steps: Expr,
        guidance: Expr,
        target_path: Expr,
    },
}
