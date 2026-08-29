#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define STACK_MAX 1024
#define CODE_MAX 4096

// --- 1. LUMINOUS BYTECODE INSTRUCTION SET (ISA) ---
typedef enum {
    OP_HALT = 0,
    OP_CONST,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_PRINT_NUM,
    OP_PRINT_MSG
} OpCode;

typedef struct {
    unsigned char code[CODE_MAX];
    double constants[256];
    char *strings[64];
    int code_size;
    int const_count;
    int str_count;
} BytecodeChunk;

typedef struct {
    double stack[STACK_MAX];
    int sp;
    int ip;
} LumVM;

// VM Initialization & Execution
static inline void vm_init(LumVM *vm) {
    vm->sp = 0;
    vm->ip = 0;
}

static inline void vm_push(LumVM *vm, double val) {
    if (vm->sp < STACK_MAX) {
        vm->stack[vm->sp++] = val;
    }
}

static inline double vm_pop(LumVM *vm) {
    if (vm->sp > 0) {
        return vm->stack[--vm->sp];
    }
    return 0.0;
}

void vm_run(LumVM *vm, BytecodeChunk *chunk) {
    vm_init(vm);
    while (vm->ip < chunk->code_size) {
        unsigned char op = chunk->code[vm->ip++];
        switch (op) {
            case OP_CONST: {
                int c_idx = chunk->code[vm->ip++];
                vm_push(vm, chunk->constants[c_idx]);
                break;
            }
            case OP_ADD: {
                double b = vm_pop(vm);
                double a = vm_pop(vm);
                vm_push(vm, a + b);
                break;
            }
            case OP_SUB: {
                double b = vm_pop(vm);
                double a = vm_pop(vm);
                vm_push(vm, a - b);
                break;
            }
            case OP_MUL: {
                double b = vm_pop(vm);
                double a = vm_pop(vm);
                vm_push(vm, a * b);
                break;
            }
            case OP_DIV: {
                double b = vm_pop(vm);
                double a = vm_pop(vm);
                vm_push(vm, a / b);
                break;
            }
            case OP_PRINT_NUM: {
                double val = vm_pop(vm);
                printf("[VM Stack Invariant]: %f\n", val);
                break;
            }
            case OP_PRINT_MSG: {
                int s_idx = chunk->code[vm->ip++];
                printf("%s\n", chunk->strings[s_idx]);
                break;
            }
            case OP_HALT:
                return;
            default:
                fprintf(stderr, "[VM Fault] Unknown opcode: 0x%02X\n", op);
                return;
        }
    }
}

int main() {
    BytecodeChunk chunk = {0};

    // Constant Pool
    chunk.constants[0] = 3.141592;
    chunk.constants[1] = 2.0;
    chunk.constants[2] = 10.5;
    chunk.constants[3] = 4.5;

    chunk.strings[0] = "==================================================";
    chunk.strings[1] = "  ❖ C LUMINOUS: BYTECODE VIRTUAL MACHINE (VM) ❖   ";
    chunk.strings[2] = "==================================================";
    chunk.strings[3] = "Evaluating: (3.141592 * 2.0) + (10.5 / 4.5)";

    // Emit Bytecode Stream
    // 1. Print Banner
    chunk.code[chunk.code_size++] = OP_PRINT_MSG; chunk.code[chunk.code_size++] = 0;
    chunk.code[chunk.code_size++] = OP_PRINT_MSG; chunk.code[chunk.code_size++] = 1;
    chunk.code[chunk.code_size++] = OP_PRINT_MSG; chunk.code[chunk.code_size++] = 2;
    chunk.code[chunk.code_size++] = OP_PRINT_MSG; chunk.code[chunk.code_size++] = 3;

    // 2. Expression Evaluation: (3.141592 * 2.0)
    chunk.code[chunk.code_size++] = OP_CONST; chunk.code[chunk.code_size++] = 0;
    chunk.code[chunk.code_size++] = OP_CONST; chunk.code[chunk.code_size++] = 1;
    chunk.code[chunk.code_size++] = OP_MUL;

    // 3. Expression Evaluation: (10.5 / 4.5)
    chunk.code[chunk.code_size++] = OP_CONST; chunk.code[chunk.code_size++] = 2;
    chunk.code[chunk.code_size++] = OP_CONST; chunk.code[chunk.code_size++] = 3;
    chunk.code[chunk.code_size++] = OP_DIV;

    // 4. Add Terms & Print Result
    chunk.code[chunk.code_size++] = OP_ADD;
    chunk.code[chunk.code_size++] = OP_PRINT_NUM;

    // 5. Halt VM
    chunk.code[chunk.code_size++] = OP_HALT;

    LumVM vm;
    vm_run(&vm, &chunk);
    return 0;
}
