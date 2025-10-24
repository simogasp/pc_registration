# Development Guide

## Development Workflow

### Before Committing

Run all checks locally with the Python check script:

```bash
# Full check suite (linting + formatting + tests with coverage)
./check.py

# Auto-fix linting and formatting issues
./check.py --fix

# Quick check (faster, no coverage)
./check.py --quick

# Only run linting (skip tests)
./check.py --lint-only

# Only run tests (skip linting)
./check.py --test-only

# Verbose test output
./check.py -v

# Combine flags
./check.py --fix -v
```

### Check Script Options

| Flag | Description |
|------|-------------|
| `--fix` | Auto-fix linting and formatting issues |
| `--quick` | Skip coverage reports for faster checks |
| `--lint-only` | Only run linting (skip tests) |
| `--test-only` | Only run tests (skip linting) |
| `-v, --verbose` | Verbose test output |

### Example Workflow

```bash
# 1. Make your changes
vim src/registration/utils/transforms.py

# 2. Run quick check during development
./check.py --quick

# 3. Auto-fix any style issues
./check.py --fix

# 4. Run full check before commit
./check.py

# 5. If all passes, commit and push
git add .
git commit -m "Add new feature"
git push
```

### Manual Testing

If you prefer to run commands manually:

```bash
# Linting
uv run ruff check src/ scripts/ tests/
uv run ruff check --fix src/ scripts/ tests/  # Auto-fix

# Formatting
uv run ruff format --check src/ scripts/ tests/
uv run ruff format src/ scripts/ tests/  # Auto-format

# Testing
uv run pytest                           # Quick tests
uv run pytest -v                        # Verbose
uv run pytest --cov=src --cov-report=html  # With coverage

# Run specific test file
uv run pytest tests/utils/test_transforms.py -v
```

Test reports are generated in the `reports/` directory:

- `reports/coverage/` - HTML coverage report (open `reports/coverage/index.html` in browser)
- `reports/coverage.xml` - XML coverage report (for CI)
- `reports/coverage.json` - JSON coverage report
