# Contributing to econAPI

Thank you for your interest in contributing to **econAPI**! We welcome contributions from quant developers, researchers, and systems architects.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/econAPI.git
   cd econAPI
   ```

2. Create and activate a Python 3.11+ virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running Tests

All mathematical engines and API endpoints must pass the automated test suite before opening a PR:

```bash
# Run all tests
pytest -v

# Run the benchmark verification script
python scripts/verify_benchmarks.py
```

## Pull Request Guidelines

1. **Precision & Proof**: Any new quantitative models must include analytical benchmark tests in `tests/` comparing outputs against published academic papers or standard reference textbooks.
2. **Type Hints**: All functions must use strict Python type annotations and Pydantic v2 schemas.
3. **No Placeholders**: Code must be executable and handle edge cases (e.g., $T \to 0$, division by zero, non-invertible covariance matrices).
4. **Code Formatting**: Ensure code adheres to PEP 8 standards.
