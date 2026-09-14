"""Shared pytest fixtures.

Fixtures live here rather than being duplicated across test modules. Ghost's own
test suite follows the convention in SPEC.md 14: exercise real behaviour against
``tmp_path`` wherever practical, and reserve mocking for the genuinely external
and expensive -- LLM provider calls above all.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root (the directory holding pyproject.toml)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def source_root(repo_root: Path) -> Path:
    """Absolute path to the installed package source, ``src/ghost``."""
    return repo_root / "src" / "ghost"
