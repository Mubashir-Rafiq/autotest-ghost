.PHONY: help install check fmt lint types imports test cov clean

help:
	@echo "install  Install the package and dev dependencies"
	@echo "check    Run every gate (what CI runs)"
	@echo "fmt      Format the code in place"
	@echo "lint     ruff lint"
	@echo "types    mypy (strict)"
	@echo "imports  import-linter architectural contracts"
	@echo "test     pytest"
	@echo "cov      pytest with a coverage report"

install:
	uv sync

# The single command to run before calling any change done. Ordered cheapest
# first so the fastest feedback arrives soonest.
check: fmt-check lint types imports test

fmt:
	uv run ruff format .
	uv run ruff check --fix .

fmt-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

types:
	uv run mypy

imports:
	uv run lint-imports

test:
	uv run pytest -q

cov:
	uv run pytest -q --cov=ghost --cov-report=term-missing

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
