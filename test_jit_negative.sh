#!/data/data/com.termux/files/usr/bin/bash
set -u

echo "=============================================="
echo " LUMINOUS JIT NEGATIVE TEST"
echo " Unsupported opcode: PRINT"
echo "=============================================="

FILE="jit_negative_print.lum"

printf "let counter = 1;\nlet accumulator = 0;\n\nwhile counter <= 6 {\n    print(counter);\n    accumulator = accumulator + counter;\n    counter = counter + 1;\n}\n\nprint(accumulator);\n" > "$FILE"

OUTPUT=$(./target/release/lum_rust "$FILE" 2>&1)
echo "$OUTPUT"

echo
echo "=============================================="
echo " NEGATIVE TEST VALIDATION"
echo "=============================================="

FAIL=0

if echo "$OUTPUT" | grep -q "JIT Compiler"; then
    echo "[✗ FAIL] JIT Compiler was invoked for unsupported PRINT loop."
    FAIL=1
else
    echo "[✓ PASS] JIT Compiler was NOT invoked."
fi

if echo "$OUTPUT" | grep -q "^21$"; then
    echo "[✓ PASS] Interpreter produced correct accumulator result: 7"
else
    echo "[✗ FAIL] Expected final accumulator result 21."
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo
    echo "🔥 NEGATIVE JIT TEST PASSED!"
    exit 0
else
    echo
    echo "⚠️ NEGATIVE JIT TEST FAILED!"
    exit 1
fi
