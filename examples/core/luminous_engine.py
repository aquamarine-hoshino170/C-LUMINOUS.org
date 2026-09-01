import re
import math
from typing import List, Dict, Any

# ==============================================================================
# C Luminous: High-Performance Mathematical Vector Class
# ==============================================================================
class LumVector:
    def __init__(self, elements: List[float]):
        self.data = [float(x) for x in elements]

    def __repr__(self):
        formatted = ", ".join(f"{x:.4f}" for x in self.data)
        return f"Vec[{formatted}]"

    def __add__(self, other):
        if isinstance(other, LumVector):
            assert len(self.data) == len(other.data), "Vector dimensions must match for addition"
            return LumVector([a + b for a, b in zip(self.data, other.data)])
        return LumVector([a + other for a in self.data])

    def __sub__(self, other):
        if isinstance(other, LumVector):
            assert len(self.data) == len(other.data), "Vector dimensions must match for subtraction"
            return LumVector([a - b for a, b in zip(self.data, other.data)])
        return LumVector([a - other for a in self.data])

    def __mul__(self, other):
        if isinstance(other, LumVector):
            assert len(self.data) == len(other.data), "Vector dimensions must match for dot product"
            return sum(a * b for a, b in zip(self.data, other.data))
        return LumVector([a * other for a in self.data])

    def norm(self) -> float:
        """Euclidean L2-Norm"""
        return math.sqrt(sum(x ** 2 for x in self.data))

# ==============================================================================
# C Luminous: Runtime Engine
# ==============================================================================
class C_Luminous_Runtime:
    def __init__(self):
        self.global_env: Dict[str, Any] = {
            "math": math,
            "vec": lambda *args: LumVector(list(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else list(args))
        }
        self.functions: Dict[str, Dict[str, Any]] = {}

    def _eval_expr(self, expr_str: str, env: Dict[str, Any]) -> Any:
        context = dict(self.global_env)
        context.update(env)
        expr_str = expr_str.replace('^', '**')
        # Safe evaluation with standard builtins allowed for method lookups
        return eval(expr_str, {"__builtins__": __builtins__}, context)

    def execute(self, source_code: str):
        # 1. Strip comments (// ...)
        clean_code = re.sub(r'//.*', '', source_code)

        # 2. Parse and register multiline functions: fn name(arg1, arg2) { ... }
        fn_pattern = r'fn\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*\{([^}]+)\}'
        for match in re.finditer(fn_pattern, clean_code, re.DOTALL):
            fn_name, params_raw, body_raw = match.groups()
            params = [p.strip() for p in params_raw.split(',') if p.strip()]
            self.functions[fn_name] = {
                "params": params,
                "body": [line.strip() for line in body_raw.split(';') if line.strip()]
            }

            def make_caller(name):
                def caller(*args):
                    fn_data = self.functions[name]
                    local_env = dict(zip(fn_data["params"], args))
                    for stmt in fn_data["body"]:
                        if stmt.startswith("return "):
                            ret_expr = stmt[7:].strip()
                            return self._eval_expr(ret_expr, local_env)
                        elif stmt.startswith("let "):
                            m = re.match(r'let\s+([a-zA-Z_]\w*)\s*=\s*(.+)', stmt)
                            if m:
                                v_name, expr = m.groups()
                                local_env[v_name] = self._eval_expr(expr, local_env)
                        elif stmt.startswith("print(") and stmt.endswith(")"):
                            p_expr = stmt[6:-1]
                            print(f"\033[1;36m[C Luminous OUT]\033[0m {self._eval_expr(p_expr, local_env)}")
                    return None
                return caller

            self.global_env[fn_name] = make_caller(fn_name)

        # 3. Clean up functions and execute linear statements
        linear_code = re.sub(fn_pattern, '', clean_code, flags=re.DOTALL)
        statements = [stmt.strip() for stmt in linear_code.split(';') if stmt.strip()]

        for line in statements:
            if line.startswith('let '):
                match = re.match(r'let\s+([a-zA-Z_]\w*)\s*=\s*(.+)', line)
                if match:
                    var_name, expr = match.groups()
                    self.global_env[var_name] = self._eval_expr(expr, self.global_env)
            elif line.startswith('print(') and line.endswith(')'):
                expr = line[6:-1]
                val = self._eval_expr(expr, self.global_env)
                print(f"\033[1;36m[C Luminous OUT]\033[0m {val}")
            else:
                try:
                    self._eval_expr(line, self.global_env)
                except Exception:
                    pass
