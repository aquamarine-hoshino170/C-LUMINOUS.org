import re

with open("src/main.rs", "r") as f:
    code = f.read()

# eval_expr বা binop সেকশন খুঁজে বের করা
# সাধারণত Rust AST-তে match op বা if op == "+" থাকে
target = re.search(r'fn\s+eval_binop\s*\([^)]*\)\s*->\s*Value\s*\{', code)
if not target:
    target = re.search(r'fn\s+eval_expr\s*\([^)]*\)\s*->\s*Value\s*\{', code)

print("Target evaluated. Applying AST Type Coercion...")

# সরাসরি main.rs-এর মধ্যে স্ট্রিং ফরম্যাটিং হেল্পার নিশ্চিত করা
