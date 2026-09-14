"""Ghost -- LLM-driven test generation with a self-healing loop.

This module owns Ghost's public API surface. Anything not re-exported here is
an internal detail and may change without notice.

The version is read from installed package metadata rather than duplicated as a
literal, so ``pyproject.toml`` remains the single source of truth for it.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autotest-ghost")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
