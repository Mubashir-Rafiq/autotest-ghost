"""Tests that pin structural invariants the linters cannot express.

Division of labour between the gates, so nothing is checked twice:

* **ruff** bans individual constructs wherever they appear -- ``print``,
  ``requests``, ``subprocess.run``, ``asyncio.get_event_loop``. A rule about a
  single line of code belongs there.
* **import-linter** enforces the layering and the "presentation never leaks into
  logic" contract. A rule about which module may import which belongs there.
* **This file** covers what neither can say: facts about *how many times*
  something appears, and agreement between the code and metadata outside it.

Adding a check here that ruff or import-linter already makes is duplication --
the exact failure mode this project exists to avoid.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import ghost


def _python_sources(source_root: Path) -> list[Path]:
    return sorted(source_root.rglob("*.py"))


def test_version_matches_pyproject(repo_root: Path) -> None:
    """``ghost.__version__`` must agree with the packaged metadata version.

    These can drift whenever the package is installed in editable mode and the
    version is bumped without reinstalling, which makes ``ghost --version`` lie.
    """
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = pyproject["project"]["version"]
    assert ghost.__version__ == declared, (
        f"ghost.__version__ is {ghost.__version__!r} but pyproject.toml declares "
        f"{declared!r}. Reinstall the package (`uv sync`) after bumping the version."
    )


def test_asyncio_run_is_called_in_exactly_one_place(source_root: Path) -> None:
    """``asyncio.run()`` may appear only in ``cli.py``, and only once.

    Ghost is an asyncio program with exactly one synchronous boundary: the CLI.
    A second ``asyncio.run`` means either a nested event loop (a runtime error)
    or a second entry point that will drift from the first. Counting occurrences
    is not something a linter can express, which is why it is a test.
    """
    offenders: dict[str, int] = {}
    for path in _python_sources(source_root):
        count = path.read_text(encoding="utf-8").count("asyncio.run(")
        if count:
            offenders[path.name] = count

    outside_cli = {name: n for name, n in offenders.items() if name != "cli.py"}
    assert not outside_cli, (
        f"asyncio.run() must only appear in cli.py, but was found in: {outside_cli}. "
        f"Async work belongs behind the single boundary in cli.py."
    )
    assert offenders.get("cli.py", 0) <= 1, (
        f"asyncio.run() appears {offenders['cli.py']} times in cli.py; there must be "
        f"exactly one event-loop entry point."
    )


def test_every_module_has_a_docstring(source_root: Path) -> None:
    """Every module states its contract: what it owns and what it does not do.

    SKILL.md's definition of done requires this, and it is the cheapest possible
    defence against a module quietly accumulating a second responsibility.
    """
    missing = [
        path.name
        for path in _python_sources(source_root)
        if not path.read_text(encoding="utf-8").lstrip().startswith('"""')
    ]
    assert not missing, (
        f"these modules have no contract docstring: {missing}. State what the module "
        f"owns, what it explicitly does not do, and what it guarantees."
    )
