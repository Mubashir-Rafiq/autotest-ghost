"""End-to-end integration test.

Everything is real except the LLM network call:
- Real project initialization (ghost init)
- Real AST static analysis and context generation (ghost index)
- Real configuration loading and precedence
- Real prompt construction
- Real isolated subprocess test runner with process-group kill
- Real first-party pytest plugin extracting exception classes
- Real change tracker caching (SHA-256 persistence)
- Real self-healing loop with judge evaluation
- Real daemon process lifecycle queries
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from ghost.cli import cli
from ghost.client import JudgeOutcome, LLMClient


def test_e2e_pipeline_and_change_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    # 1. Project Initialization via CLI wizard
    init_res = runner.invoke(cli, ["init"], input="\n\n\n")
    assert init_res.exit_code == 0
    assert (tmp_path / "ghost.toml").is_file()
    assert (tmp_path / ".ghost").is_dir()

    # 2. Create real application source code
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    calc_file = src_dir / "calc.py"
    calc_file.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    return a - b\n",
        encoding="utf-8",
    )

    # 3. Indexing via CLI
    index_res = runner.invoke(cli, ["index", "--show"])
    assert index_res.exit_code == 0
    assert "calc.py" in index_res.output

    # 4. Prompt verification via CLI
    prompt_res = runner.invoke(cli, ["prompt", "src/calc.py"])
    assert prompt_res.exit_code == 0
    assert "def add(a: int, b: int)" in prompt_res.output
    assert "def subtract(a: int, b: int)" in prompt_res.output

    # 5. One-shot generation with real isolated test execution
    valid_test_code = (
        "# Generated at: 24-09-2026 12:00:00 | Source: src/calc.py\n"
        "from src.calc import add, subtract\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n\n"
        "def test_subtract():\n"
        "    assert subtract(5, 3) == 2\n"
    )

    async def mock_generate_test(self: LLMClient, *args: Any, **kwargs: Any) -> str:
        return valid_test_code

    monkeypatch.setattr(LLMClient, "generate_test", mock_generate_test)

    gen_res = runner.invoke(cli, ["generate", "src/calc.py"])
    assert gen_res.exit_code == 0, gen_res.output
    assert "PASS" in gen_res.output

    test_file = tmp_path / "tests" / "test_calc.py"
    assert test_file.is_file()
    assert "test_add" in test_file.read_text(encoding="utf-8")

    # 6. Change tracking cache: --if-changed skips unchanged file
    gen_cached = runner.invoke(cli, ["generate", "src/calc.py", "--if-changed"])
    assert gen_cached.exit_code == 0
    assert "SKIPPED: calc.py is unchanged (--if-changed)." in gen_cached.output


def test_e2e_self_healing_and_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    runner.invoke(cli, ["init"], input="\n\n\n")
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    calc_file = src_dir / "calc.py"
    calc_file.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b + 10  # modified behavior\n\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    return a - b\n",
        encoding="utf-8",
    )

    # First attempt generates test expecting old behavior;
    # heal attempt generates test for new behavior
    attempts = 0

    async def mock_generate_or_heal(self: LLMClient, *args: Any, **kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return (
                "# Generated at: 24-09-2026 12:00:00 | Source: src/calc.py\n"
                "from src.calc import add\n\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n"
            )
        return (
            "# Generated at: 24-09-2026 12:00:00 | Source: src/calc.py\n"
            "from src.calc import add\n\n"
            "def test_add():\n"
            "    assert add(1, 2) == 13\n"
        )

    async def mock_judge(self: LLMClient, *args: Any, **kwargs: Any) -> JudgeOutcome:
        return JudgeOutcome.FIX_TEST

    monkeypatch.setattr(LLMClient, "generate_test", mock_generate_or_heal)
    monkeypatch.setattr(LLMClient, "heal_test", mock_generate_or_heal)
    monkeypatch.setattr(LLMClient, "consult_the_judge", mock_judge)

    test_file = tmp_path / "tests" / "test_calc.py"

    heal_res = runner.invoke(cli, ["generate", "src/calc.py", "--force", "--heal", "--judge"])
    assert heal_res.exit_code == 0, heal_res.output
    assert "PASS (healed after 1 attempt(s))" in heal_res.output
    assert "assert add(1, 2) == 13" in test_file.read_text(encoding="utf-8")

    # Run tests command via CLI directly on the generated file
    run_res = runner.invoke(cli, ["run-tests", "tests/test_calc.py"])
    assert run_res.exit_code == 0
    assert "PASS: tests/test_calc.py" in run_res.output

    # Verify providers and models CLI commands
    prov_res = runner.invoke(cli, ["providers"])
    assert prov_res.exit_code == 0
    assert "Supported Providers" in prov_res.output

    # Daemon query commands
    status_res = runner.invoke(cli, ["status"])
    assert status_res.exit_code == 0
    assert "Ghost daemon is not running." in status_res.output
