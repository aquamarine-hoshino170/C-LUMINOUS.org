#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>

#define MAX_VARS 512
#define MAX_FUNCS 128
#define MAX_PARAMS 32
#define MAX_BODY_LINES 128
#define MAX_LINE_LEN 1024
#define MAX_DICT_ENTRIES 64

typedef enum { VAL_NUM, VAL_STR, VAL_LIST, VAL_DICT } ValType;

struct Value;

typedef struct {
    struct Value *items;
    int len;
    int cap;
} LumList;

typedef struct {
    char keys[MAX_DICT_ENTRIES][64];
    struct Value *vals;
    int len;
    int cap;
} LumDict;

typedef struct Value {
    ValType type;
    double num;
    char str[MAX_LINE_LEN];
    LumList list;
    LumDict dict;
} Value;

typedef struct {
    char name[64];
    Value val;
    bool is_defined;
} Variable;

typedef struct {
    char name[64];
    char params[MAX_PARAMS][64];
    int param_count;
    char body[MAX_BODY_LINES][MAX_LINE_LEN];
    int body_lines;
} Function;

static Variable globals[MAX_VARS];
static int global_count = 0;

static Function functions[MAX_FUNCS];
static int func_count = 0;

typedef struct {
    Variable locals[MAX_VARS];
    int count;
} Scope;

static Scope call_stack[64];
static int stack_depth = 0;

// Safe Constructors with Explicit Zero Initialization
Value make_num(double n) {
    Value v;
    memset(&v, 0, sizeof(Value));
    v.type = VAL_NUM;
    v.num = n;
    return v;
}

Value make_str(const char *s) {
    Value v;
    memset(&v, 0, sizeof(Value));
    v.type = VAL_STR;
    if (s != NULL) {
        strncpy(v.str, s, MAX_LINE_LEN - 1);
        v.str[MAX_LINE_LEN - 1] = '\0';
    }
    return v;
}

Value make_list() {
    Value v;
    memset(&v, 0, sizeof(Value));
    v.type = VAL_LIST;
    v.list.cap = 16;
    v.list.len = 0;
    v.list.items = (Value *)calloc(v.list.cap, sizeof(Value));
    return v;
}

Value make_dict() {
    Value v;
    memset(&v, 0, sizeof(Value));
    v.type = VAL_DICT;
    v.dict.cap = 16;
    v.dict.len = 0;
    v.dict.vals = (Value *)calloc(v.dict.cap, sizeof(Value));
    return v;
}

void list_append(Value *l, Value item) {
    if (!l || l->type != VAL_LIST) return;
    if (l->list.len >= l->list.cap) {
        l->list.cap *= 2;
        Value *new_items = (Value *)realloc(l->list.items, l->list.cap * sizeof(Value));
        if (!new_items) return;
        l->list.items = new_items;
    }
    l->list.items[l->list.len++] = item;
}

void dict_set(Value *d, const char *key, Value val) {
    if (!d || d->type != VAL_DICT || !key) return;
    for (int i = 0; i < d->dict.len; i++) {
        if (strcmp(d->dict.keys[i], key) == 0) {
            d->dict.vals[i] = val;
            return;
        }
    }
    if (d->dict.len >= d->dict.cap) {
        d->dict.cap *= 2;
        Value *new_vals = (Value *)realloc(d->dict.vals, d->dict.cap * sizeof(Value));
        if (!new_vals) return;
        d->dict.vals = new_vals;
    }
    strncpy(d->dict.keys[d->dict.len], key, 63);
    d->dict.keys[d->dict.len][63] = '\0';
    d->dict.vals[d->dict.len] = val;
    d->dict.len++;
}

Value dict_get(Value *d, const char *key) {
    if (!d || d->type != VAL_DICT || !key) return make_num(0.0);
    for (int i = 0; i < d->dict.len; i++) {
        if (strcmp(d->dict.keys[i], key) == 0) {
            return d->dict.vals[i];
        }
    }
    return make_num(0.0);
}

void set_var(const char *name, Value val) {
    if (!name) return;
    if (stack_depth > 0) {
        Scope *s = &call_stack[stack_depth - 1];
        for (int i = 0; i < s->count; i++) {
            if (strcmp(s->locals[i].name, name) == 0) {
                s->locals[i].val = val;
                return;
            }
        }
        if (s->count < MAX_VARS) {
            strncpy(s->locals[s->count].name, name, 63);
            s->locals[s->count].name[63] = '\0';
            s->locals[s->count].val = val;
            s->locals[s->count].is_defined = true;
            s->count++;
            return;
        }
    }
    for (int i = 0; i < global_count; i++) {
        if (strcmp(globals[i].name, name) == 0) {
            globals[i].val = val;
            return;
        }
    }
    if (global_count < MAX_VARS) {
        strncpy(globals[global_count].name, name, 63);
        globals[global_count].name[63] = '\0';
        globals[global_count].val = val;
        globals[global_count].is_defined = true;
        global_count++;
    }
}

Value *get_var_ref(const char *name) {
    if (!name) return NULL;
    if (stack_depth > 0) {
        Scope *s = &call_stack[stack_depth - 1];
        for (int i = 0; i < s->count; i++) {
            if (strcmp(s->locals[i].name, name) == 0) {
                return &s->locals[i].val;
            }
        }
    }
    for (int i = 0; i < global_count; i++) {
        if (strcmp(globals[i].name, name) == 0) {
            return &globals[i].val;
        }
    }
    return NULL;
}

Value get_var(const char *name) {
    Value *ref = get_var_ref(name);
    if (ref) return *ref;
    return make_num(0.0);
}

// Safe String Conversion with Zero Null-Overflow
void val_to_str(Value v, char *out, int max_len) {
    if (!out || max_len <= 0) return;
    memset(out, 0, max_len);

    if (v.type == VAL_NUM) {
        if (v.num == (long)v.num) snprintf(out, max_len - 1, "%ld", (long)v.num);
        else snprintf(out, max_len - 1, "%f", v.num);
    } else if (v.type == VAL_STR) {
        strncpy(out, v.str, max_len - 1);
    } else if (v.type == VAL_LIST) {
        strncpy(out, "[", max_len - 1);
        for (int i = 0; i < v.list.len; i++) {
            char item_s[256];
            val_to_str(v.list.items[i], item_s, sizeof(item_s));
            strncat(out, item_s, max_len - strlen(out) - 1);
            if (i < v.list.len - 1) strncat(out, ", ", max_len - strlen(out) - 1);
        }
        strncat(out, "]", max_len - strlen(out) - 1);
    } else if (v.type == VAL_DICT) {
        strncpy(out, "{", max_len - 1);
        for (int i = 0; i < v.dict.len; i++) {
            char item_s[256];
            val_to_str(v.dict.vals[i], item_s, sizeof(item_s));
            strncat(out, "\"", max_len - strlen(out) - 1);
            strncat(out, v.dict.keys[i], max_len - strlen(out) - 1);
            strncat(out, "\": ", max_len - strlen(out) - 1);
            strncat(out, item_s, max_len - strlen(out) - 1);
            if (i < v.dict.len - 1) strncat(out, ", ", max_len - strlen(out) - 1);
        }
        strncat(out, "}", max_len - strlen(out) - 1);
    }
    out[max_len - 1] = '\0';
}

void print_val(Value v) {
    char out[MAX_LINE_LEN];
    val_to_str(v, out, sizeof(out));
    printf("%s", out);
}

// Memory-Safe String Built-ins
Value lum_split(const char *src, const char *delim) {
    Value list = make_list();
    if (!src || !delim || strlen(delim) == 0) {
        list_append(&list, make_str(src));
        return list;
    }
    char buf[MAX_LINE_LEN];
    strncpy(buf, src, MAX_LINE_LEN - 1);
    buf[MAX_LINE_LEN - 1] = '\0';

    char *token = strtok(buf, delim);
    while (token != NULL) {
        list_append(&list, make_str(token));
        token = strtok(NULL, delim);
    }
    return list;
}

Value lum_join(Value list, const char *delim) {
    if (list.type != VAL_LIST || list.list.len == 0) return make_str("");
    char res[MAX_LINE_LEN];
    memset(res, 0, sizeof(res));

    for (int i = 0; i < list.list.len; i++) {
        char item_str[MAX_LINE_LEN];
        val_to_str(list.list.items[i], item_str, sizeof(item_str));
        strncat(res, item_str, MAX_LINE_LEN - strlen(res) - 1);
        if (i < list.list.len - 1 && delim) {
            strncat(res, delim, MAX_LINE_LEN - strlen(res) - 1);
        }
    }
    res[MAX_LINE_LEN - 1] = '\0';
    return make_str(res);
}

Value lum_replace(const char *src, const char *old_sub, const char *new_sub) {
    if (!src || !old_sub || !new_sub || strlen(old_sub) == 0) return make_str(src);
    char res[MAX_LINE_LEN];
    memset(res, 0, sizeof(res));

    const char *p = src;
    int old_len = strlen(old_sub);
    while (*p && strlen(res) < MAX_LINE_LEN - 1) {
        if (strncmp(p, old_sub, old_len) == 0) {
            strncat(res, new_sub, MAX_LINE_LEN - strlen(res) - 1);
            p += old_len;
        } else {
            int len = strlen(res);
            res[len] = *p;
            res[len + 1] = '\0';
            p++;
        }
    }
    res[MAX_LINE_LEN - 1] = '\0';
    return make_str(res);
}

Value lum_trim(const char *src) {
    if (!src) return make_str("");
    const char *start = src;
    while (isspace(*start)) start++;
    if (*start == '\0') return make_str("");
    const char *end = src + strlen(src) - 1;
    while (end > start && isspace(*end)) end--;
    char res[MAX_LINE_LEN];
    int len = end - start + 1;
    if (len > MAX_LINE_LEN - 1) len = MAX_LINE_LEN - 1;
    strncpy(res, start, len);
    res[len] = '\0';
    return make_str(res);
}

// Re-entrant Local Pointer Expression Parser
Value execute_block(char lines[][MAX_LINE_LEN], int count, bool *returned);
void load_file_into_runtime(const char *filename);

Value parse_or(const char **eptr_ref);

Value parse_atomic(const char **eptr_ref) {
    const char *p = *eptr_ref;
    while (isspace(*p)) p++;

    if (strncmp(p, "not ", 4) == 0 || strncmp(p, "not(", 4) == 0) {
        p += 3;
        *eptr_ref = p;
        Value val = parse_atomic(eptr_ref);
        return make_num((val.num == 0.0) ? 1.0 : 0.0);
    }

    if (*p == '"' || *p == '\'') {
        char q = *p++;
        char sbuf[MAX_LINE_LEN];
        int sidx = 0;
        while (*p != q && *p != '\0' && sidx < MAX_LINE_LEN - 1) {
            if (*p == '\\' && *(p + 1) == 'n') {
                sbuf[sidx++] = '\n';
                p += 2;
            } else {
                sbuf[sidx++] = *p++;
            }
        }
        if (*p == q) p++;
        sbuf[sidx] = '\0';
        *eptr_ref = p;
        return make_str(sbuf);
    }

    if (*p == '[') {
        p++;
        Value list = make_list();
        while (isspace(*p)) p++;
        if (*p != ']') {
            while (true) {
                *eptr_ref = p;
                Value item = parse_or(eptr_ref);
                p = *eptr_ref;
                list_append(&list, item);
                while (isspace(*p)) p++;
                if (*p == ',') p++;
                else if (*p == ']' || *p == '\0') {
                    if (*p == ']') p++;
                    break;
                }
            }
        } else {
            p++;
        }
        *eptr_ref = p;
        return list;
    }

    if (*p == '{') {
        p++;
        Value d = make_dict();
        while (isspace(*p)) p++;
        if (*p != '}') {
            while (true) {
                *eptr_ref = p;
                Value k = parse_or(eptr_ref);
                p = *eptr_ref;
                while (isspace(*p)) p++;
                if (*p == ':') p++;
                *eptr_ref = p;
                Value v = parse_or(eptr_ref);
                p = *eptr_ref;
                dict_set(&d, k.str, v);
                while (isspace(*p)) p++;
                if (*p == ',') p++;
                else if (*p == '}' || *p == '\0') {
                    if (*p == '}') p++;
                    break;
                }
            }
        } else {
            p++;
        }
        *eptr_ref = p;
        return d;
    }

    if (*p == '(') {
        p++;
        *eptr_ref = p;
        Value val = parse_or(eptr_ref);
        p = *eptr_ref;
        while (isspace(*p)) p++;
        if (*p == ')') p++;
        *eptr_ref = p;
        return val;
    }

    if (isdigit(*p) || (*p == '.' && isdigit(*(p + 1)))) {
        char *end;
        double val = strtod(p, &end);
        *eptr_ref = end;
        return make_num(val);
    }

    if (isalpha(*p) || *p == '_') {
        char ident[64];
        int idx = 0;
        while ((isalnum(*p) || *p == '_') && idx < 63) {
            ident[idx++] = *p++;
        }
        ident[idx] = '\0';
        while (isspace(*p)) p++;

        if (strcmp(ident, "true") == 0) { *eptr_ref = p; return make_num(1.0); }
        if (strcmp(ident, "false") == 0) { *eptr_ref = p; return make_num(0.0); }

        if (*p == '(') {
            p++;
            Value args[MAX_PARAMS];
            int arg_c = 0;
            while (isspace(*p)) p++;
            if (*p != ')') {
                while (true) {
                    *eptr_ref = p;
                    Value arg_val = parse_or(eptr_ref);
                    p = *eptr_ref;
                    if (arg_c < MAX_PARAMS) args[arg_c++] = arg_val;
                    while (isspace(*p)) p++;
                    if (*p == ',') p++;
                    else if (*p == ')') { p++; break; }
                    else if (*p == '\0') break;
                }
            } else {
                p++;
            }
            *eptr_ref = p;

            if (strcmp(ident, "split") == 0) {
                if (arg_c >= 2) return lum_split(args[0].str, args[1].str);
                if (arg_c == 1) return lum_split(args[0].str, " ");
                return make_list();
            }
            if (strcmp(ident, "join") == 0) {
                if (arg_c >= 2) return lum_join(args[0], args[1].str);
                if (arg_c == 1) return lum_join(args[0], "");
                return make_str("");
            }
            if (strcmp(ident, "replace") == 0) {
                if (arg_c >= 3) return lum_replace(args[0].str, args[1].str, args[2].str);
                return make_str(args[0].str);
            }
            if (strcmp(ident, "trim") == 0) {
                if (arg_c >= 1) return lum_trim(args[0].str);
                return make_str("");
            }
            if (strcmp(ident, "to_num") == 0) {
                if (arg_c > 0) {
                    if (args[0].type == VAL_NUM) return args[0];
                    if (args[0].type == VAL_STR) return make_num(atof(args[0].str));
                }
                return make_num(0.0);
            }
            if (strcmp(ident, "to_str") == 0) {
                if (arg_c > 0) {
                    char buf[MAX_LINE_LEN];
                    val_to_str(args[0], buf, sizeof(buf));
                    return make_str(buf);
                }
                return make_str("");
            }
            if (strcmp(ident, "write_file") == 0) {
                if (arg_c >= 2) {
                    FILE *fp = fopen(args[0].str, "w");
                    if (fp) {
                        char out_buf[MAX_LINE_LEN];
                        val_to_str(args[1], out_buf, sizeof(out_buf));
                        fputs(out_buf, fp);
                        fclose(fp);
                        return make_num(1.0);
                    }
                }
                return make_num(0.0);
            }
            if (strcmp(ident, "append_file") == 0) {
                if (arg_c >= 2) {
                    FILE *fp = fopen(args[0].str, "a");
                    if (fp) {
                        char out_buf[MAX_LINE_LEN];
                        val_to_str(args[1], out_buf, sizeof(out_buf));
                        fputs(out_buf, fp);
                        fclose(fp);
                        return make_num(1.0);
                    }
                }
                return make_num(0.0);
            }
            if (strcmp(ident, "read_file") == 0) {
                if (arg_c >= 1) {
                    FILE *fp = fopen(args[0].str, "r");
                    if (fp) {
                        char content[MAX_LINE_LEN];
                        memset(content, 0, sizeof(content));
                        int n = fread(content, 1, sizeof(content) - 1, fp);
                        content[n] = '\0';
                        fclose(fp);
                        return make_str(content);
                    }
                }
                return make_str("");
            }
            if (strcmp(ident, "len") == 0) {
                if (arg_c > 0 && args[0].type == VAL_LIST) return make_num(args[0].list.len);
                if (arg_c > 0 && args[0].type == VAL_DICT) return make_num(args[0].dict.len);
                if (arg_c > 0 && args[0].type == VAL_STR) return make_num(strlen(args[0].str));
                return make_num(0.0);
            }
            if (strcmp(ident, "sqrt") == 0) return make_num(sqrt(args[0].num));
            if (strcmp(ident, "sin") == 0) return make_num(sin(args[0].num));
            if (strcmp(ident, "cos") == 0) return make_num(cos(args[0].num));
            if (strcmp(ident, "exp") == 0) return make_num(exp(args[0].num));
            if (strcmp(ident, "log") == 0) return make_num(log(args[0].num));

            // User Function Dispatch
            for (int f = 0; f < func_count; f++) {
                if (strcmp(functions[f].name, ident) == 0) {
                    Scope local_s;
                    memset(&local_s, 0, sizeof(Scope));
                    for (int pa = 0; pa < functions[f].param_count && pa < arg_c; pa++) {
                        strncpy(local_s.locals[pa].name, functions[f].params[pa], 63);
                        local_s.locals[pa].name[63] = '\0';
                        local_s.locals[pa].val = args[pa];
                        local_s.locals[pa].is_defined = true;
                    }
                    local_s.count = functions[f].param_count;
                    call_stack[stack_depth++] = local_s;

                    bool ret_flag = false;
                    Value ret_val = execute_block(functions[f].body, functions[f].body_lines, &ret_flag);
                    stack_depth--;
                    return ret_val;
                }
            }
        }

        if (*p == '[') {
            p++;
            *eptr_ref = p;
            Value idx_val = parse_or(eptr_ref);
            p = *eptr_ref;
            while (isspace(*p)) p++;
            if (*p == ']') p++;
            *eptr_ref = p;
            Value var = get_var(ident);
            if (var.type == VAL_LIST) {
                int i = (int)idx_val.num;
                if (i >= 0 && i < var.list.len) return var.list.items[i];
            } else if (var.type == VAL_DICT) {
                return dict_get(&var, idx_val.str);
            }
            return make_num(0.0);
        }

        *eptr_ref = p;
        return get_var(ident);
    }

    if (*p == '-') {
        p++;
        *eptr_ref = p;
        Value atom = parse_atomic(eptr_ref);
        if (atom.type == VAL_NUM) atom.num = -atom.num;
        return atom;
    }

    *eptr_ref = p;
    return make_num(0.0);
}

Value parse_factor(const char **eptr_ref) {
    Value left = parse_atomic(eptr_ref);
    const char *p = *eptr_ref;
    while (isspace(*p)) p++;
    if (*p == '^') {
        p++;
        *eptr_ref = p;
        Value right = parse_factor(eptr_ref);
        if (left.type == VAL_NUM && right.type == VAL_NUM) {
            left.num = pow(left.num, right.num);
        }
    }
    return left;
}

Value parse_term(const char **eptr_ref) {
    Value left = parse_factor(eptr_ref);
    while (true) {
        const char *p = *eptr_ref;
        while (isspace(*p)) p++;
        if (*p == '*') {
            p++;
            *eptr_ref = p;
            Value right = parse_factor(eptr_ref);
            if (left.type == VAL_NUM && right.type == VAL_NUM) left.num *= right.num;
        } else if (*p == '/') {
            p++;
            *eptr_ref = p;
            Value right = parse_factor(eptr_ref);
            if (left.type == VAL_NUM && right.type == VAL_NUM) {
                left.num = (right.num != 0.0) ? (left.num / right.num) : 0.0;
            }
        } else {
            break;
        }
    }
    return left;
}

Value parse_math(const char **eptr_ref) {
    Value left = parse_term(eptr_ref);
    while (true) {
        const char *p = *eptr_ref;
        while (isspace(*p)) p++;
        if (*p == '+') {
            p++;
            *eptr_ref = p;
            Value right = parse_term(eptr_ref);
            if (left.type == VAL_NUM && right.type == VAL_NUM) left.num += right.num;
            else if (left.type == VAL_STR && right.type == VAL_STR) {
                int cur_len = strlen(left.str);
                strncat(left.str, right.str, MAX_LINE_LEN - cur_len - 1);
            }
        } else if (*p == '-') {
            p++;
            *eptr_ref = p;
            Value right = parse_term(eptr_ref);
            if (left.type == VAL_NUM && right.type == VAL_NUM) left.num -= right.num;
        } else {
            break;
        }
    }
    return left;
}

Value parse_relational(const char **eptr_ref) {
    Value left = parse_math(eptr_ref);
    const char *p = *eptr_ref;
    while (isspace(*p)) p++;
    if (*p == '<' && *(p + 1) == '=') {
        p += 2; *eptr_ref = p;
        return make_num((left.num <= parse_math(eptr_ref).num) ? 1.0 : 0.0);
    } else if (*p == '>' && *(p + 1) == '=') {
        p += 2; *eptr_ref = p;
        return make_num((left.num >= parse_math(eptr_ref).num) ? 1.0 : 0.0);
    } else if (*p == '=' && *(p + 1) == '=') {
        p += 2; *eptr_ref = p;
        Value right = parse_math(eptr_ref);
        if (left.type == VAL_NUM && right.type == VAL_NUM) return make_num((left.num == right.num) ? 1.0 : 0.0);
        else if (left.type == VAL_STR && right.type == VAL_STR) return make_num((strcmp(left.str, right.str) == 0) ? 1.0 : 0.0);
        return make_num(0.0);
    } else if (*p == '!' && *(p + 1) == '=') {
        p += 2; *eptr_ref = p;
        return make_num((left.num != parse_math(eptr_ref).num) ? 1.0 : 0.0);
    } else if (*p == '<') {
        p++; *eptr_ref = p;
        return make_num((left.num < parse_math(eptr_ref).num) ? 1.0 : 0.0);
    } else if (*p == '>') {
        p++; *eptr_ref = p;
        return make_num((left.num > parse_math(eptr_ref).num) ? 1.0 : 0.0);
    }
    return left;
}

Value parse_and(const char **eptr_ref) {
    Value left = parse_relational(eptr_ref);
    while (true) {
        const char *p = *eptr_ref;
        while (isspace(*p)) p++;
        if (strncmp(p, "and ", 4) == 0 || strncmp(p, "and(", 4) == 0) {
            p += 3; *eptr_ref = p;
            Value right = parse_relational(eptr_ref);
            left = make_num((left.num != 0.0 && right.num != 0.0) ? 1.0 : 0.0);
        } else {
            break;
        }
    }
    return left;
}

Value parse_or(const char **eptr_ref) {
    Value left = parse_and(eptr_ref);
    while (true) {
        const char *p = *eptr_ref;
        while (isspace(*p)) p++;
        if (strncmp(p, "or ", 3) == 0 || strncmp(p, "or(", 3) == 0) {
            p += 2; *eptr_ref = p;
            Value right = parse_and(eptr_ref);
            left = make_num((left.num != 0.0 || right.num != 0.0) ? 1.0 : 0.0);
        } else {
            break;
        }
    }
    return left;
}

Value eval_expr(const char *str) {
    const char *p = str;
    return parse_or(&p);
}

Value execute_block(char lines[][MAX_LINE_LEN], int count, bool *returned) {
    Value last_val = make_num(0.0);
    int i = 0;
    while (i < count) {
        char *line = lines[i];
        while (isspace(*line)) line++;
        if (*line == '\0' || *line == '#') { i++; continue; }

        if (strncmp(line, "import ", 7) == 0) {
            char *mstart = line + 7;
            while (isspace(*mstart)) mstart++;
            if (*mstart == '"' || *mstart == '\'') {
                char q = *mstart++;
                char mpath[256];
                int pidx = 0;
                while (*mstart != q && *mstart != '\0' && pidx < 255) mpath[pidx++] = *mstart++;
                mpath[pidx] = '\0';
                load_file_into_runtime(mpath);
            }
            i++;
            continue;
        }

        if (strncmp(line, "return ", 7) == 0) {
            *returned = true;
            return eval_expr(line + 7);
        }

        if (strncmp(line, "print(", 6) == 0) {
            char inner[MAX_LINE_LEN];
            strncpy(inner, line + 6, sizeof(inner) - 1);
            inner[sizeof(inner) - 1] = '\0';
            char *end = strrchr(inner, ')');
            if (end) *end = '\0';
            Value v = eval_expr(inner);
            print_val(v);
            printf("\n");
            i++;
            continue;
        }

        if (strncmp(line, "write_file(", 11) == 0 || strncmp(line, "append_file(", 12) == 0) {
            eval_expr(line);
            i++;
            continue;
        }

        char *dot = strchr(line, '.');
        if (dot && strstr(dot, ".append(")) {
            char vname[64];
            int nlen = dot - line;
            if (nlen > 63) nlen = 63;
            strncpy(vname, line, nlen);
            vname[nlen] = '\0';
            char *p = vname; while(isspace(*p)) p++;
            Value *target = get_var_ref(p);
            if (target && target->type == VAL_LIST) {
                char *app_start = strstr(dot, ".append(") + 8;
                char inner[MAX_LINE_LEN];
                strncpy(inner, app_start, sizeof(inner) - 1);
                inner[sizeof(inner) - 1] = '\0';
                char *end = strrchr(inner, ')');
                if (end) *end = '\0';
                Value item = eval_expr(inner);
                list_append(target, item);
            }
            i++;
            continue;
        }

        if (strncmp(line, "while ", 6) == 0) {
            char cond_str[MAX_LINE_LEN];
            char *start = line + 6;
            char *brace = strchr(start, '{');
            if (brace) {
                int clen = brace - start;
                if (clen > MAX_LINE_LEN - 1) clen = MAX_LINE_LEN - 1;
                strncpy(cond_str, start, clen);
                cond_str[clen] = '\0';
            } else {
                strncpy(cond_str, start, MAX_LINE_LEN - 1);
                cond_str[MAX_LINE_LEN - 1] = '\0';
            }

            char body[MAX_BODY_LINES][MAX_LINE_LEN];
            int bcount = 0;
            i++;
            int depth = 1;
            while (i < count && depth > 0) {
                if (strchr(lines[i], '{')) depth++;
                if (strchr(lines[i], '}')) depth--;
                if (depth == 0) break;
                if (bcount < MAX_BODY_LINES) {
                    strncpy(body[bcount++], lines[i], MAX_LINE_LEN - 1);
                }
                i++;
            }
            i++;

            while (eval_expr(cond_str).num != 0.0) {
                bool r = false;
                Value res = execute_block(body, bcount, &r);
                if (r) { *returned = true; return res; }
            }
            continue;
        }

        if (strncmp(line, "if ", 3) == 0) {
            char cond_str[MAX_LINE_LEN];
            char *start = line + 3;
            char *brace = strchr(start, '{');
            if (brace) {
                int clen = brace - start;
                if (clen > MAX_LINE_LEN - 1) clen = MAX_LINE_LEN - 1;
                strncpy(cond_str, start, clen);
                cond_str[clen] = '\0';
            } else {
                strncpy(cond_str, start, MAX_LINE_LEN - 1);
                cond_str[MAX_LINE_LEN - 1] = '\0';
            }

            char body[MAX_BODY_LINES][MAX_LINE_LEN];
            int bcount = 0;
            i++;
            int depth = 1;
            while (i < count && depth > 0) {
                if (strchr(lines[i], '{')) depth++;
                if (strchr(lines[i], '}')) depth--;
                if (depth == 0) break;
                if (bcount < MAX_BODY_LINES) {
                    strncpy(body[bcount++], lines[i], MAX_LINE_LEN - 1);
                }
                i++;
            }
            i++;

            if (eval_expr(cond_str).num != 0.0) {
                bool r = false;
                Value res = execute_block(body, bcount, &r);
                if (r) { *returned = true; return res; }
            }
            continue;
        }

        char *lbracket = strchr(line, '[');
        char *eq = strchr(line, '=');
        if (lbracket && eq && lbracket < eq && *(eq + 1) != '=' && *(eq - 1) != '!' && *(eq - 1) != '<' && *(eq - 1) != '>') {
            char vname[64];
            int nlen = lbracket - line;
            if (nlen > 63) nlen = 63;
            strncpy(vname, line, nlen);
            vname[nlen] = '\0';
            char *p = vname; while(isspace(*p)) p++;

            char *rbracket = strchr(line, ']');
            char idx_str[64];
            int ilen = rbracket - (lbracket + 1);
            if (ilen > 63) ilen = 63;
            strncpy(idx_str, lbracket + 1, ilen);
            idx_str[ilen] = '\0';

            Value idx_val = eval_expr(idx_str);
            Value rhs = eval_expr(eq + 1);

            Value *target = get_var_ref(p);
            if (target) {
                if (target->type == VAL_LIST) {
                    int idx = (int)idx_val.num;
                    if (idx >= 0 && idx < target->list.len) target->list.items[idx] = rhs;
                } else if (target->type == VAL_DICT) {
                    dict_set(target, idx_val.str, rhs);
                }
            }
            i++;
            continue;
        }

        if (eq && *(eq + 1) != '=' && *(eq - 1) != '!' && *(eq - 1) != '<' && *(eq - 1) != '>') {
            *eq = '\0';
            char vname[64];
            sscanf(line, "%s", vname);
            Value v = eval_expr(eq + 1);
            set_var(vname, v);
            last_val = v;
            i++;
            continue;
        }

        last_val = eval_expr(line);
        i++;
    }
    return last_val;
}

void load_file_into_runtime(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open '%s'\n", filename);
        return;
    }

    char all_lines[512][MAX_LINE_LEN];
    int line_total = 0;
    while (line_total < 512 && fgets(all_lines[line_total], MAX_LINE_LEN, f)) {
        all_lines[line_total][strcspn(all_lines[line_total], "\r\n")] = '\0';
        line_total++;
    }
    fclose(f);

    char main_lines[512][MAX_LINE_LEN];
    int main_total = 0;

    for (int i = 0; i < line_total; i++) {
        char *l = all_lines[i];
        while (isspace(*l)) l++;

        if (strncmp(l, "fn ", 3) == 0) {
            char fname[64];
            char *paren = strchr(l, '(');
            int nlen = paren - (l + 3);
            if (nlen > 63) nlen = 63;
            strncpy(fname, l + 3, nlen);
            fname[nlen] = '\0';
            while (fname[strlen(fname)-1] == ' ') fname[strlen(fname)-1] = '\0';

            strncpy(functions[func_count].name, fname, 63);

            char pstr[256];
            char *close_p = strchr(paren, ')');
            int plen = close_p - (paren + 1);
            if (plen > 255) plen = 255;
            strncpy(pstr, paren + 1, plen);
            pstr[plen] = '\0';

            int p_idx = 0;
            char *tok = strtok(pstr, ", ");
            while (tok && p_idx < MAX_PARAMS) {
                strncpy(functions[func_count].params[p_idx++], tok, 63);
                tok = strtok(NULL, ", ");
            }
            functions[func_count].param_count = p_idx;

            int bcount = 0;
            i++;
            int depth = 1;
            while (i < line_total && depth > 0) {
                if (strchr(all_lines[i], '{')) depth++;
                if (strchr(all_lines[i], '}')) depth--;
                if (depth == 0) break;
                if (bcount < MAX_BODY_LINES) {
                    strncpy(functions[func_count].body[bcount++], all_lines[i], MAX_LINE_LEN - 1);
                }
                i++;
            }
            functions[func_count].body_lines = bcount;
            func_count++;
        } else {
            strncpy(main_lines[main_total++], all_lines[i], MAX_LINE_LEN - 1);
        }
    }

    bool r = false;
    execute_block(main_lines, main_total, &r);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        load_file_into_runtime(argv[1]);
        return 0;
    }

    printf("==================================================\n");
    printf("  ❖ C LUMINOUS SYSTEMS LANGUAGE (v3.2: Secure) ❖  \n");
    printf("==================================================\n");

    char buf[MAX_LINE_LEN];
    while (true) {
        printf("lum>>> ");
        if (!fgets(buf, sizeof(buf), stdin)) break;
        buf[strcspn(buf, "\r\n")] = '\0';
        if (strcmp(buf, "exit()") == 0 || strcmp(buf, "quit()") == 0) break;

        char single[1][MAX_LINE_LEN];
        strncpy(single[0], buf, MAX_LINE_LEN - 1);
        bool r = false;
        Value res = execute_block(single, 1, &r);
        if (strncmp(buf, "print", 5) != 0 && strncmp(buf, "import", 6) != 0 && strncmp(buf, "write_file", 10) != 0 && strncmp(buf, "append_file", 11) != 0 && strchr(buf, '=') == NULL && strstr(buf, ".append") == NULL && strlen(buf) > 0) {
            print_val(res);
            printf("\n");
        }
    }
    return 0;
}
