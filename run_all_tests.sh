#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "      LUMINOUS SCIENTIFIC COMPUTING TEST HARNESS          "
echo "=========================================================="

echo "[1/7] Running Complex Arithmetic Engine..."
cargo run --quiet -- test_complex.lum

echo "[2/7] Running 2D Optical Field Superposition..."
cargo run --quiet -- test_optical_tensor.lum

echo "[3/7] Running Jones Calculus & Wave Polarization..."
cargo run --quiet -- test_jones.lum

echo "[4/7] Running Optical Fourier Transform (DFT)..."
cargo run --quiet -- test_fourier.lum

echo "[5/7] Running 2D Angular Spectrum Propagation (Phase 2)..."
cargo run --quiet -- test_propagation.lum

echo "[6/7] Running 2D FDTD Maxwell Solver (Phase 3)..."
cargo run --quiet -- test_fdtd.lum

echo "[7/7] Running Quantum Beam Splitter Optics (Phase 4)..."
cargo run --quiet -- test_quantum.lum

echo "[BONUS] Running Pure First-Order Horn Clause Theorem Prover..."
cargo run --quiet -- test_horn_prover.lum

echo "=========================================================="
echo "   ALL SCIENTIFIC & LOGICAL SUITES PASSED CLEANLY (8/8)   "
echo "=========================================================="
