#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>

#define MAX_VARS 512
#define MAX_LINE 1024

// --- DYNAMIC ENVIRONMENT (SYMBOL TABLE) ---
typedef struct {
    char name[64];
    double value;
    bool is_defined;
} LumSymbol;

static LumSymbol symtab[MAX_VARS];
static int sym_count = 0;

void set_var(const char *name, double val) {
    for (int i = 0; i < sym_count; i++) {
        if (strcmp(symtab[i].name, name) == 0) {
            symtab[i].value = val;
            return;
        }
    }
    if (sym_count < MAX_VARS) {
        strcpy(symtab[sym_count].name, name);
        symtab[sym_count].value = val;
        symtab[sym_count].is_defined = true;
        sym_count++;
    }
}

double get_var(const char *name) {
    for (int i = 0; i < sym_count; i++) {
        if (strcmp(symtab[i].name, name) == 0) {
            return symtab[i].value;
        }
    }
    return 0.0;
}

// --- EXPRESSION PARSER & EVALUATOR (Recursive Descent) ---
const char *expr_ptr;

double parse_expression();

double parse_factor() {
    while (isspace(*expr_ptr)) expr_ptr++;
    
    if (*expr_ptr == '(') {
        expr_ptr++; // skip '('
        double val = parse_expression();
        while (isspace(*expr_ptr)) expr_ptr++;
        if (*expr_ptr == ')') expr_ptr++; // skip ')'
        return val;
    }

    if (isdigit(*expr_ptr) || *expr_ptr == '.') {
        char *end;
        double val = strtod(expr_ptr, &end);
        expr_ptr = end;
        return val;
    }

    if (isalpha(*expr_ptr) || *expr_ptr == '_') {
        char name[64];
        int idx = 0;
        while (isalnum(*expr_ptr) || *expr_ptr == '_') {
            name[idx++] = *expr_ptr++;
        }
        name[idx] = '\0';
        
        while (isspace(*expr_ptr)) expr_ptr++;
        if (*expr_ptr == '(') {
            expr_ptr++; // function call
            double arg = parse_expression();
            while (isspace(*expr_ptr)) expr_ptr++;
            if (*expr_ptr == ')') expr_ptr++;
            
            if (strcmp(name, "sqrt") == 0) return sqrt(arg);
            if (strcmp(name, "sin") == 0) return sin(arg);
            if (strcmp(name, "cos") == 0) return cos(arg);
            if (strcmp(name, "exp") == 0) return exp(arg);
            if (strcmp(name, "log") == 0) return log(arg);
        }
        return get_var(name);
    }

    if (*expr_ptr == '-') {
        expr_ptr++;
        return -parse_factor();
    }

    return 0.0;
}

double parse_term() {
    double left = parse_factor();
    while (true) {
        while (isspace(*expr_ptr)) expr_ptr++;
        if (*expr_ptr == '*') {
            expr_ptr++;
            left *= parse_factor();
        } else if (*expr_ptr == '/') {
            expr_ptr++;
            double div = parse_factor();
            left = (div != 0.0) ? (left / div) : 0.0;
        } else if (*expr_ptr == '^') {
            expr_ptr++;
            left = pow(left, parse_factor());
        } else {
            break;
        }
    }
    return left;
}

double parse_expression() {
    double left = parse_term();
    while (true) {
        while (isspace(*expr_ptr)) expr_ptr++;
        if (*expr_ptr == '+') {
            expr_ptr++;
            left += parse_term();
        } else if (*expr_ptr == '-') {
            expr_ptr++;
            left -= parse_term();
        } else {
            break;
        }
    }
    return left;
}

// --- STATEMENT INTERPRETER ---
void execute_line(char *line) {
    while (isspace(*line)) line++;
    if (*line == '\0' || *line == '#') return; // Comment or empty line

    // Print Statement: print(...)
    if (strncmp(line, "print(", 6) == 0) {
        char *inner = line + 6;
        char *end = strrchr(inner, ')');
        if (end) *end = '\0';
        while (isspace(*inner)) inner++;
        
        if (*inner == '"' || *inner == '\'') {
            char quote = *inner++;
            char *closing = strrchr(inner, quote);
            if (closing) *closing = '\0';
            printf("%s\n", inner);
        } else {
            expr_ptr = inner;
            double result = parse_expression();
            printf("%f\n", result);
        }
        return;
    }

    // Assignment: x = expression
    char *eq = strchr(line, '=');
    if (eq) {
        *eq = '\0';
        char var_name[64];
        sscanf(line, "%s", var_name);
        char *rhs = eq + 1;
        expr_ptr = rhs;
        double val = parse_expression();
        set_var(var_name, val);
        return;
    }

    // Direct Evaluation (Like Python Interactive Mode)
    expr_ptr = line;
    double res = parse_expression();
    printf("%f\n", res);
}

// --- REPL & SCRIPT RUNNER ---
int main(int argc, char **argv) {
    if (argc > 1) {
        FILE *f = fopen(argv[1], "r");
        if (!f) {
            fprintf(stderr, "Error: Cannot open '%s'\n", argv[1]);
            return 1;
        }
        char line[MAX_LINE];
        while (fgets(line, sizeof(line), f)) {
            // Trim trailing newline
            line[strcspn(line, "\r\n")] = '\0';
            execute_line(line);
        }
        fclose(f);
        return 0;
    }

    // Interactive Shell Mode (Python REPL style)
    printf("==================================================\n");
    printf("  ❖ C LUMINOUS 1.0.0 (Interactive REPL Engine) ❖  \n");
    printf("  Type expressions directly or exit() to quit     \n");
    printf("==================================================\n");

    char buffer[MAX_LINE];
    while (true) {
        printf("lum>>> ");
        if (!fgets(buffer, sizeof(buffer), stdin)) break;
        buffer[strcspn(buffer, "\r\n")] = '\0';
        
        if (strcmp(buffer, "exit()") == 0 || strcmp(buffer, "quit()") == 0) {
            break;
        }
        execute_line(buffer);
    }
    return 0;
}
