use lum_rust::compiler::BytecodeCompiler;
use lum_rust::jit_emitter::{LimitSpec, LoopInspector, StepSpec};
use lum_rust::lexer::tokenize;
use lum_rust::parser::Parser;

#[test]
fn test_loop_inspector_on_pure_asm_loop() {
    let source = std::fs::read_to_string("examples/pure_asm_loop.lum")
        .expect("Failed to read pure_asm_loop.lum");
    let tokens = tokenize(&source);
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().expect("AST Parse failed");
    let compiler = BytecodeCompiler::new();
    let chunk = compiler.compile(ast).expect("Compilation failed");

    let spec = LoopInspector::inspect_chunk(&chunk);
    assert!(
        spec.is_some(),
        "LoopInspector should recognize pure_asm_loop"
    );

    let loop_spec = spec.unwrap();
    assert_eq!(loop_spec.loop_start_ip, 2);
    assert_eq!(loop_spec.loop_end_ip, 15);
    assert_eq!(loop_spec.counter_reg, 0);
    assert_eq!(loop_spec.accum_reg, 1);
    assert_eq!(loop_spec.limit, LimitSpec::Constant(10_000_000));
    assert_eq!(loop_spec.step, StepSpec::Constant(1));
}

#[test]
fn test_loop_inspector_rejects_side_effects() {
    let source = std::fs::read_to_string("examples/test_loop_logic.lum")
        .expect("Failed to read test_loop_logic.lum");
    let tokens = tokenize(&source);
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().expect("AST Parse failed");
    let compiler = BytecodeCompiler::new();
    let chunk = compiler.compile(ast).expect("Compilation failed");

    let spec = LoopInspector::inspect_chunk(&chunk);
    assert!(
        spec.is_none(),
        "LoopInspector MUST reject loops with PRINT/side-effects!"
    );
}
