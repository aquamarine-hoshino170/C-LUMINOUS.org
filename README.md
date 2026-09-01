# ✨ C-LUMINOUS — High-Performance Optical Computing Framework

C-LUMINOUS is a high-performance C library and toolkit for optical computing, photonic simulations, and quantum optical research. It focuses on numerical accuracy and runtime performance and provides modules for wave propagation, photonic circuits, quantum state simulation, nonlinear optics, and optical signal processing.

Table of Contents
- Overview
- Key Features
- Repository Layout
- Build & Install
- Examples
- Documentation & GitHub Pages
- Development & Debugging
- Contributing
- License
- Contact

---

## Overview

C-LUMINOUS aims to provide a fast, extensible foundation for researchers and engineers building photonic simulations and optical computation systems. The codebase is primarily C for performance-critical components, with auxiliary tools and examples in Python and scripts.

## Key Features

- Core optical field engine (2D/3D)
- Photonic circuit simulation (beam splitters, phase shifters, interferometers)
- Quantum optics primitives (Fock states, coherent and squeezed states)
- Wave propagation and Maxwell/FDTD solvers
- Nonlinear optics (second/third harmonic generation, parametric processes)
- Optical signal processing (FFT-based analysis, filters)
- Performance optimizations (SIMD, multithreading; optional CUDA acceleration)

---

## Repository Layout

- src/                — core C sources and numerical engines
- include/            — public headers (e.g. luminous.h)
- examples/           — example programs and demos
- docs/               — documentation and GitHub Pages site sources
- vscode-extension/   — VS Code language extension and packaging
- pkg-manager/        — helper tools (Rust) for package tasks
- index.html          — repository root demo / landing page
- .github/workflows/  — CI and Pages deployment workflows

---

## Build & Install

Prerequisites
- C compiler (GCC 9+ or Clang 10+)
- Make and/or CMake
- Optional: CUDA toolkit (for GPU acceleration)

Build with Make

```bash
git clone https://github.com/aquamarine-hoshino170/C-LUMINOUS.org.git
cd C-LUMINOUS.org
make all
# run tests
make test
```

Build with CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
ctest
```

Python bindings (if available)

```bash
pip install -e .
# or
python setup.py bdist_wheel
```

Notes
- If your system has CUDA and the project supports it, enable GPU builds via Makefile/CMake options and install appropriate drivers.

---

## Examples

See the `examples/` directory for sample programs. Example usage (C):

```c
#include <luminous.h>

int main() {
    optical_field_t *field = optical_field_create(128, 128, 1.55e-6);
    gaussian_beam_t beam = {.wavelength = 1.55e-6, .waist = 5e-6, .amplitude = 1.0};
    optical_field_gaussian(field, beam);
    optical_field_propagate(field, 1e-3);
    optical_field_save_vtk(field, "output.vtk");
    optical_field_free(field);
    return 0;
}
```

Python example (if bindings installed):

```python
import luminous as lum
field = lum.OpticalField(size=256, wavelength=1.55e-6)
beam = lum.GaussianBeam(waist=5e-6, amplitude=1.0)
field.apply_beam(beam)
field.propagate(distance=1e-3)
intensity = field.intensity()
```

---

## Documentation & GitHub Pages

Project documentation lives under `docs/`. A minimal Jekyll config has been added at `docs/_config.yml`. There is a GitHub Actions workflow at `.github/workflows/pages.yml` that builds and deploys the site.

If you want to preview the site locally:

```bash
# from repo root
bundle install
bundle exec jekyll serve --source docs --destination docs/_site
```

Note: The current Pages workflow expects site sources under `docs/`. If your content is in a different location, update `.github/workflows/pages.yml` to set the correct `source` input or move the site content into `docs/`.

---

## Development & Debugging

- Inspect repository contents locally:
  - `ls -la`, open `examples/`, `src/`, and `docs/` to review files
- There are CI workflow runs in Settings → Actions or `https://github.com/aquamarine-hoshino170/C-LUMINOUS.org/actions`
- If GitHub Actions fails during Pages build with errors like `No such file or directory @ dir_chdir0 - /github/workspace/docs`, ensure the `docs/` directory exists and contains the site sources.

---

## Contributing

Contributions are welcome. Common areas for contribution:
- Add or extend optical components and solvers
- Improve performance and add GPU kernels
- Expand documentation and tutorials under `docs/`
- Tests and CI improvements

Please open issues or pull requests. Follow standard GitHub workflow: fork → branch → PR.

---

## License

MIT © 2026 aquamarine-hoshino170

---

## Contact

GitHub: https://github.com/aquamarine-hoshino170/C-LUMINOUS.org

If you want additional sections (detailed API docs, full examples, or detailed install matrix), tell me what to add and I will update the README or add files under `docs/`.
