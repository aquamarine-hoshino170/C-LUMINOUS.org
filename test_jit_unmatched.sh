#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "=============================================="
echo " LUMINOUS JIT NEGATIVE TEST"
echo " Unsupported pattern: UNMATCHED WHITELIST OPS"
echo "=============================================="

cat > test_jit_unmatched.lum <<'EOF'
let counter = 1;
let accumulator = 0;

while counter <= 6 {
    let junk = counter;
    accumulator = accumulator + counter;
    counter = counter + 1;
}

print(accumulator);
EOF

output=$(./target/release/lum_rust test_jit_unmatched.lum 2>&1)
printf "%s\n" "$output"

echo
echo "=============================================="
echo " UNMATCHED OPCODE VALIDATION"
echo "=============================================="

if printf "%s\n" "$output" | grep -q "JIT Compiler"; then
    echo "[✗ FAIL] JIT Compiler was invoked for an unmatched body instruction."
    exit 1
else
    echo "[✓ PASS] JIT Compiler was NOT invoked."
fi

if printf "%s\n" "$output" | grep -q "^21$"; then
    echo "[✓ PASS] Interpreter produced correct accumulator result: 21"
else
    echo "[✗ FAIL] Expected accumulator result 21."
    exit 1
fi

echo
echo "🔥 UNMATCHED OPCODE NEGATIVE TEST PASSED!"
