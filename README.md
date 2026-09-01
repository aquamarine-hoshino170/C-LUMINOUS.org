# ✨ C-LUMINOUS — High‑Performance Optical Computing Framework

**সারসংক্ষেপ (BN):** C-LUMINOUS হল একটি উচ্চ‑পারফরম্যান্স C লাইব্রেরি অপটিক্যাল কম্পিউটিং, ফটোনিক সিমুলেশন এবং কোয়ান্টাম অপটিক্সের জন্য। (See full English README below.)

---

## Overview

C-LUMINOUS is a high-performance C library and toolkit for optical computing, photonic simulations, and quantum optical phenomena. It focuses on speed and numerical precision and includes modules for wave propagation, photonic circuits, quantum states, nonlinear optics, and optical signal processing.

Key areas:
- Quantum optics (coherent, squeezed, entangled states; measurement models)
- Photonic computing and circuit simulation (beam splitters, phase shifters, interferometers)
- Wave propagation and Maxwell/FDTD solvers
- Nonlinear optics (SHG/THG, parametric processes)
- Optical signal processing (FFT-based analysis, filtering)

---

## Repository layout (high level)

- src/        — core C sources and optical engines
- include/    — public headers (luminous.h)
- examples/   — example programs and demos
- docs/       — site content (GitHub Pages/Jekyll)
- vscode-extension/ — VS Code language extension and packaging
- pkg-manager/ — helper tools (Rust) for package tasks
- index.html  — repository root demo / landing page

---

## Build / Install

Prerequisites:
- C compiler (GCC 9+ or Clang 10+)
- Make and/or CMake
- Optional: CUDA for GPU acceleration

Build (Make):

```bash
git clone https://github.com/aquamarine-hoshino170/C-LUMINOUS.org.git
cd C-LUMINOUS.org
make all
# run tests
make test
```

Build (CMake):

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
ctest
```

Python bindings (if present):

```bash
pip install -e .
# or
python setup.py bdist_wheel
```

Notes:
- If you need GPU acceleration, install CUDA and enable via Makefile/CMake flags (if available).

---

## Examples

Look in the examples/ directory for many sample programs (Gaussian beam propagation, quantum circuit demos, SHG examples). To run a simple demo (C):

```bash
cd examples
# compile/run example as instructed in examples/README.md
```

---

## Documentation & GitHub Pages

Project documentation is maintained under docs/. A minimal Jekyll config has been added to docs/_config.yml. The repository includes a GitHub Actions workflow (.github/workflows/pages.yml) to build and deploy the Pages site.

If you want to preview the site locally:

```bash
# from repo root
bundle install
bundle exec jekyll serve --source docs --destination docs/_site
```

Note: the GitHub Actions workflow expects site sources under docs/. If your site files are elsewhere, edit `.github/workflows/pages.yml` to adjust the `source` input or move the site files into docs/.

---

## Development & Debugging

- To inspect the repo locally:
  - ls -la, open examples/, src/ and docs/ to see current content
- There are several CI workflow runs in `.github/workflows/` that build and deploy pages; recent runs are visible at: https://github.com/aquamarine-hoshino170/C-LUMINOUS.org/actions

---

## Contributing

Contributions welcome. Common areas:
- Add/extend optical components and solvers
- Improve performance and add GPU kernels
- Expand documentation and tutorials under docs/
- Tests and CI improvements

Please open issues or pull requests.

---

## License

MIT © 2026 aquamarine-hoshino170

---

## Contact

GitHub: https://github.com/aquamarine-hoshino170/C-LUMINOUS.org

If you'd like changes to this README (language, more technical details, Bengali translation), tell me which sections to expand and I'll update it.