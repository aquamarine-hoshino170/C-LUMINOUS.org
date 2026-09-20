#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "      LUMINOUS SCIENTIFIC COMPUTING TEST HARNESS          "
echo "=========================================================="

echo "[1/8] Running Complex Arithmetic Engine..."
cargo run --quiet -- test_complex.lum

echo "[2/8] Running 2D Optical Field Superposition..."
cargo run --quiet -- test_optical_tensor.lum

echo "[3/8] Running Jones Calculus & Wave Polarization..."
cargo run --quiet -- test_jones.lum

echo "[4/8] Running Optical Fourier Transform (DFT)..."
cargo run --quiet -- test_fourier.lum

echo "[5/8] Running 2D Angular Spectrum Propagation (Phase 2)..."
cargo run --quiet -- test_propagation.lum

echo "[6/8] Running 2D FDTD Maxwell Solver (Phase 3)..."
cargo run --quiet -- test_fdtd.lum

echo "[7/8] Running Quantum Beam Splitter Optics (Phase 4)..."
cargo run --quiet -- test_quantum.lum

echo "[8/8] Running Pure First-Order Horn Clause Theorem Prover..."
cargo run --quiet -- test_horn_prover.lum

echo "[ML/CORE] Running Symbolic Calculus & Tensor Engine..."
cargo run --quiet -- test_scientific_ml.lum

echo "=========================================================="
echo "   ALL SCIENTIFIC & LOGICAL SUITES PASSED CLEANLY (9/9)   "
echo "=========================================================="
