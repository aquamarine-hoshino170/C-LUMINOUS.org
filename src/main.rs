mod ast;
mod evaluator;
mod lexer;
mod parser;
mod runtime;

use parser::Parser;
use runtime::Environment;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: lum <script.lum>");
        std::process::exit(1);
    }

    let file_path_str = &args[1];
    let path = Path::new(file_path_str);

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("LUM-IO00: Failed to read file '{}': {}", file_path_str, e);
            std::process::exit(1);
        }
    };

    let base_dir = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();

    let tokens = lexer::tokenize(&source);

    let mut parser = Parser::new(tokens);
    let ast = match parser.parse() {
        Ok(ast) => ast,
        Err(err) => {
            eprintln!("Parse Error: {}", err);
            std::process::exit(1);
        }
    };

    let mut env = Environment::new(base_dir);
    if let Err(err) = evaluator::execute_program(ast, &mut env) {
        eprintln!("Runtime Error: {}", err);
        std::process::exit(1);
    }
}
