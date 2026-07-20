# Contributing to SkillOpt

Thank you for your interest in contributing! SkillOpt welcomes contributions of all kinds.

## Getting Started

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
pip install -e ".[dev]"
```

## How to Contribute

### 🐛 Bug Reports
Open a GitHub issue with reproduction steps, expected/actual behavior, and your config file (remove API keys).

### 🔧 Add a Benchmark
See the [guide](docs/guide/new-benchmark.md) and use the scaffold at `skillopt/envs/_template/`.

### 🤖 Add a Model Backend
See the [guide](docs/guide/new-backend.md).

### 📝 Improve Documentation
```bash
pip install -e ".[docs]"
mkdocs serve   # Preview at http://localhost:8000
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make changes and test with an existing benchmark
3. Submit a PR with a clear description
4. Ensure CI passes

## Code Style
- Follow existing patterns in the codebase
- Use type hints for function signatures
- Keep docstrings concise

## License
By contributing, you agree your contributions are licensed under the [MIT License](LICENSE).
