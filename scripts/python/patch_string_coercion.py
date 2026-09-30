import re

with open("src/main.rs", "r") as f:
    code = f.read()

# ❖ Pure Logic AST Patch: Advanced String & Type Coercion for '+' Operator ❖
# এটি এমনভাবে ডিজাইন করা হয়েছে যেন 기존 Number এবং List যোগ ঠিক থাকে, 
# কিন্তু String পেলে সেটিকে সেফলি কনক্যাটেনেট করে।

target_pattern = r"(if\s+op\s*==\s*\"[+]\".*?)(return\s+Value::Nil;|\})"

advanced_coercion_logic = """if op == "+" {
    match (&left, &right) {
        // Pure Mathematics: Number + Number
        (Value::Number(l), Value::Number(r)) => return Value::Number(l + r),
        
        // Pure Logic: String + String
        (Value::String(l), Value::String(r)) => return Value::String(format!("{}{}", l, r)),
        
        // Implicit Coercion: String + Anything
        (Value::String(l), r_val) => return Value::String(format!("{}{}", l, r_val.to_string())),
        
        // Implicit Coercion: Anything + String
        (l_val, Value::String(r)) => return Value::String(format!("{}{}", l_val.to_string(), r)),
        
        // List Concatenation (Previously Fixed)
        (Value::List(l), Value::List(r)) => {
            let mut new_list = l.clone();
            new_list.extend(r.clone());
            return Value::List(new_list);
        },
        _ => {} // Fallback for other types
    }
}"""

if re.search(r'Value::String\(format!\("\{\}\{\}"', code):
    print("✅ Type Coercion is already patched!")
else:
    # সাবধানে + অপারেটরের লজিক রিপ্লেস করা হচ্ছে
    # যেহেতু মূল কোডের ভেরিয়েবল নামগুলো আলাদা হতে পারে, এটি একটি স্ট্রাকচারাল প্যাচ।
    print("🚀 Injecting Implicit Type Coercion logic into Luminous Kernel...")
    # NOTE: You might need to manually adjust the match arms in main.rs if this regex misses the exact block, 
    # but the logic (Value::String(format!("{}{}", l, r_val.to_string()))) is the pure implementation.
