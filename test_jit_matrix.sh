#!/data/data/com.termux/files/usr/bin/bash
set -u

echo "=============================================="
echo " LUMINOUS JIT CORRECTNESS MATRIX"
echo "=============================================="

PASS=0
FAIL=0

run_test() {
    STEP="$1"
    EXPECTED="$2"
    FILE="jit_step_${STEP}.lum"

    printf "let counter = 1;\nlet accumulator = 0;\n\nwhile counter <= 10000000 {\n    accumulator = accumulator + counter;\n    counter = counter + %s;\n}\n\nprint(accumulator);\n" "$STEP" > "$FILE"

    echo
    echo "[TEST] step=$STEP"
    echo "----------------------------------------------"

    OUTPUT=$(timeout 10s ./target/release/lum_rust "$FILE" 2>&1 || true)
    echo "$OUTPUT"

    if echo "$OUTPUT" | grep -qF "$EXPECTED"; then
        echo "[✓ PASS] step=$STEP result=$EXPECTED"
        PASS=$((PASS + 1))
    else
        echo "[✗ FAIL] step=$STEP"
        FAIL=$((FAIL + 1))
    fi
}

run_test 1 "50000005000000"
run_test 2 "25000000000000"
run_test 3 "16666671666667"

echo
echo "=============================================="
echo " RESULTS"
echo "=============================================="
echo "[✓] Passed: $PASS"
echo "[✗] Failed: $FAIL"

if [ "$FAIL" -eq 0 ]; then
    echo
    echo "🔥 ALL JIT CORRECTNESS TESTS PASSED!"
    exit 0
else
    echo
    echo "⚠️ JIT CORRECTNESS TEST FAILURE"
    exit 1
fi
