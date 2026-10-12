# Contributing to C-LUMINOUS

Thanks for your interest in improving C-LUMINOUS.

We welcome bug fixes, performance work, documentation improvements, feature
ideas, and experimentation around photonic and optical computing.

## Code of conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- Report bugs and suggest improvements via GitHub issues
- Submit pull requests for fixes or enhancements
- Improve documentation, examples, and tutorials
- Share benchmarks, reproducible test cases, or design feedback

## Development setup

### Prerequisites

- Rust toolchain for helper tooling and package management tasks
- C compiler such as GCC or Clang
- Make or CMake
- Optional: CUDA toolkit for GPU-related work

### Clone the repository

```bash
git clone https://github.com/aquamarine-hoshino170/C-LUMINOUS.org.git
cd C-LUMINOUS.org
```

### Build

```bash
make all
```

Or with CMake:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Run tests

```bash
make test
```

## Contribution workflow

1. Fork the repository.
2. Create a feature branch.
3. Make your changes with focused commits.
4. Ensure the project still builds and tests pass locally.
5. Open a pull request with a clear description of the problem and fix.

## Coding guidelines

- Prefer clear, readable C code and maintainable abstractions.
- Keep changes narrowly scoped and documented.
- Add or update tests when fixing bugs or adding features.
- For performance-sensitive changes, include benchmark context when relevant.
- Do not commit generated or temporary files unrelated to the issue.

## Commit and PR expectations

- Keep the subject line concise and descriptive.
- Include a summary of what changed and why.
- Link to the relevant issue when applicable.
- Mention validation steps in the PR description.

## Reporting bugs

Before opening a bug report, check whether the issue already exists. Include:

- a clear description of the problem
- reproduction steps
- expected vs. actual behavior
- relevant environment details, compiler version, OS, and build flags

## Need help?

If you are unsure where to begin, open a discussion or issue and the maintainers
will help guide you.
