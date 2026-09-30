pub mod jit_emitter;
mod jit;
mod ast;
mod bytecode;
mod compiler;
mod evaluator;
mod lexer;
mod parser;
mod runtime;
mod vm;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn print_help() {
    println!("Luminous Scientific Language v0.4.0 (Register Bytecode Ecosystem)");
    println!("Usage:");
    println!("  lum_rust <file.lum>                  : Compile and execute via VM");
    println!("  lum_rust <file.lum> --disasm         : Disassemble and print bytecode");
    println!("  lum_rust <file.lum> -c <out.lbc>     : Compile and serialize to binary bytecode");
    println!("  lum_rust <file.lbc>                  : Directly execute binary bytecode");
    println!("  lum_rust <file.lum> --ast            : Run via classical AST interpreter");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "-h" || args[1] == "--help" {
        print_help();
        return;
    }

    let input_path = &args[1];

    // If target is directly a precompiled .lbc binary
    if input_path.ends_with(".lbc") {
        println!("[*] Loading Precompiled Binary Bytecode: '{}'...", input_path);
        let chunk = match bytecode::Chunk::load_from_file(input_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error loading LBC: {}", e);
                process::exit(1);
            }
        };
        let mut vm = vm::SVM::new();
        if let Err(e) = vm.execute(&chunk) {
            eprintln!("VM Execution Error: {}", e);
            process::exit(1);
        }
        return;
    }

    let source = match fs::read_to_string(input_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file {}: {}", input_path, e);
            process::exit(1);
        }
    };

    let tokens = lexer::tokenize(&source);
    let mut parser = parser::Parser::new(tokens);
    let ast = match parser.parse() {
        Ok(tree) => tree,
        Err(e) => {
            eprintln!("Parse Error: {}", e);
            process::exit(1);
        }
    };

    let use_ast = args.iter().any(|arg| arg == "--ast");
    if use_ast {
        let base_dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut runtime_env = runtime::Environment::new(base_dir);
        if let Err(e) = evaluator::execute_program(ast, &mut runtime_env) {
            eprintln!("Runtime Error: {}", e);
            process::exit(1);
        }
        return;
    }

    let compiler = compiler::BytecodeCompiler::new();
    let chunk = match compiler.compile(ast) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Compilation Error: {}", e);
            process::exit(1);
        }
    };

    // Flag: --disasm
    if args.iter().any(|arg| arg == "--disasm" || arg == "--dump-bytecode") {
        let name = Path::new(input_path).file_name().unwrap().to_str().unwrap();
        chunk.disassemble(name);
        return;
    }

    // Flag: -c <out.lbc>
    if let Some(pos) = args.iter().position(|arg| arg == "-c") {
        if pos + 1 < args.len() {
            let out_lbc = &args[pos + 1];
            if let Err(e) = chunk.save_to_file(out_lbc) {
                eprintln!("Serialization Error: {}", e);
                process::exit(1);
            }
            println!("[✓] Successfully compiled and serialized bytecode to '{}'", out_lbc);
            return;
        } else {
            eprintln!("Error: -c requires output path (e.g. -c out.lbc)");
            process::exit(1);
        }
    }

    // Default: Direct VM execution
    let mut vm = vm::SVM::new();
    if let Err(e) = vm.execute(&chunk) {
        eprintln!("VM Execution Error: {}", e);
        process::exit(1);
    }
}
