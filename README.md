# Ghost

[![CI](https://github.com/Mubashir-Rafiq/autotest-ghost/actions/workflows/ci.yml/badge.svg)](https://github.com/Mubashir-Rafiq/autotest-ghost/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/autotest-ghost.svg)](https://pypi.org/project/autotest-ghost/)
[![Python versions](https://img.shields.io/pypi/pyversions/autotest-ghost.svg)](https://pypi.org/project/autotest-ghost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ghost watches your Python files. Every time you save one, it writes a `pytest`
test for it using an LLM, runs that test in a real subprocess, and — if the test
breaks — tries to fix it.

The interesting part is what it does when a test *fails an assertion*. A naive
tool rewrites the test until it passes, which happily rubber-stamps a genuine
bug in your code. Ghost asks a second question first — **is the test wrong, or
is the source wrong?** — and when the answer is "the source", it stops and tells
you instead of touching the test.

---

## Installation

Install from PyPI with `pip` or `uv`:

```bash
pip install autotest-ghost
```

Or install with optional provider extras:

```bash
pip install "autotest-ghost[all]"       # includes openai + anthropic SDKs
pip install "autotest-ghost[openai]"    # OpenAI SDK only
pip install "autotest-ghost[anthropic]" # Anthropic SDK only
```

Or run directly without installing via `uvx`:

```bash
uvx autotest-ghost --help
```

---

## Quick Start

### 1. Initialize configuration

Set your provider API key (e.g. `export GROQ_API_KEY=gsk_...` or in `.env`) and initialize:

```bash
ghost init
```

This interactive wizard detects available providers (Groq, OpenAI, Anthropic, Ollama, LM Studio, OpenRouter, Custom) and creates `ghost.toml`.

### 2. Generate tests

Generate, run, and automatically heal tests for a Python source file:

```bash
ghost generate src/calculator.py
```

Or batch-generate tests across the whole project with coverage reporting:

```bash
ghost generate --all --cov
```

### 3. File Watcher & Daemon

Watch for file saves in the foreground:

```bash
ghost watch
```

Or run as a detached background daemon:

```bash
ghost start       # start background watcher daemon
ghost status      # inspect daemon state, uptime, and queue
ghost logs -f     # stream daemon logs
ghost stop        # cleanly drain and stop daemon
```

### 4. History, Rollback & Stats

```bash
ghost history                 # list files with test snapshots
ghost history src/calc.py     # inspect attempts and error classifications
ghost rollback src/calc.py    # roll back test file to original baseline
ghost stats                   # display cumulative token usage and requests
```

---

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Mubashir-Rafiq/autotest-ghost.git
cd autotest-ghost
make install  # uv sync
make check    # every quality gate: ruff lint, format, mypy strict, import-linter, pytest
make test     # run pytest test suite
make build    # build sdist and wheel
```

The quality gates are not decorative:
- `mypy` runs in strict mode across `src/` and `tests/`.
- `ruff` enforces strict linting, formatting, and banned APIs.
- `import-linter` enforces the unidirectional architecture layering contracts.

---

## Documentation

| Document | Description |
|---|---|
| [`ROADMAP.md`](ROADMAP.md) | Build stages, implementation status, and defect fixes |
| [`SPEC.md`](SPEC.md) | Authoritative technical specification |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | Key architectural and safety improvements over earlier designs |
| [`EXPLAINED.md`](EXPLAINED.md) | Background: state machine design and self-healing mechanics |

---

## License

[MIT](LICENSE) © Mubashir Rafiq

