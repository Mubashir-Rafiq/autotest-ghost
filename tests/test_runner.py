"""Tests for subprocess test runner, timeout enforcement, and error classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost.pytest_plugin import (
    pytest_collectreport,
    pytest_runtest_makereport,
    pytest_sessionfinish,
)
from ghost.runner import (
    ErrorClassification,
    classify_error,
    find_project_python,
    run_test,
)


def test_classify_error_exact_lookup() -> None:
    # Direct lookup matches
    assert classify_error("AssertionError") == ErrorClassification.LOGIC
    assert classify_error("SyntaxError") == ErrorClassification.SYNTAX
    assert classify_error("IndentationError") == ErrorClassification.SYNTAX
    assert classify_error("ModuleNotFoundError") == ErrorClassification.SYNTAX
    assert classify_error("ImportError") == ErrorClassification.SYNTAX
    assert classify_error("AttributeError") == ErrorClassification.RUNTIME
    assert classify_error("ZeroDivisionError") == ErrorClassification.RUNTIME
    assert classify_error("TypeError") == ErrorClassification.RUNTIME
    assert classify_error("KeyError") == ErrorClassification.RUNTIME


def test_classify_error_substring_fallback() -> None:
    # When exception_type is None or unknown, fallback to stderr/stdout
    assert classify_error(None, stdout="AssertionError: assert 1 == 2") == ErrorClassification.LOGIC
    assert classify_error(None, stderr="SyntaxError: invalid syntax") == ErrorClassification.SYNTAX
    assert (
        classify_error(None, stderr="AttributeError: 'None' has no 'x'")
        == ErrorClassification.RUNTIME
    )
    assert (
        classify_error(None, stdout="some obscure failure message") == ErrorClassification.UNKNOWN
    )


def test_find_project_python_finds_virtualenv(tmp_path: Path) -> None:
    """Regression test: sys.executable lacks target dependencies; must discover project venv."""
    is_win = sys.platform == "win32"
    sub_bin = "Scripts" if is_win else "bin"
    py_name = "python.exe" if is_win else "python"

    venv_bin = tmp_path / ".venv" / sub_bin
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / py_name
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    discovered = find_project_python(tmp_path)
    assert discovered == fake_python


@pytest.mark.asyncio
async def test_run_test_passing_test(tmp_path: Path) -> None:
    """Regression test: a passing test must have returncode 0 and classification None."""
    test_file = tmp_path / "test_ok.py"
    test_file.write_text("def test_pass(): assert True\n", encoding="utf-8")

    result = await run_test(test_file, tmp_path)
    assert result.passed
    assert result.return_code == 0
    assert result.classification is None
    assert not result.timed_out


@pytest.mark.asyncio
async def test_run_test_assertion_error_classified_as_logic(tmp_path: Path) -> None:
    """Test assertion failure is classified as LOGIC via the pytest plugin."""
    test_file = tmp_path / "test_assert.py"
    test_file.write_text("def test_fail(): assert 1 == 2\n", encoding="utf-8")

    result = await run_test(test_file, tmp_path)
    assert not result.passed
    assert result.return_code != 0
    assert result.classification == ErrorClassification.LOGIC
    assert result.exception_type == "AssertionError"


@pytest.mark.asyncio
async def test_run_test_runtime_error_classified_as_runtime(tmp_path: Path) -> None:
    """Test runtime exception is classified as RUNTIME via the pytest plugin."""
    test_file = tmp_path / "test_runtime.py"
    test_file.write_text("def test_div(): return 1 / 0\n", encoding="utf-8")

    result = await run_test(test_file, tmp_path)
    assert not result.passed
    assert result.classification == ErrorClassification.RUNTIME
    assert result.exception_type == "ZeroDivisionError"


@pytest.mark.asyncio
async def test_run_test_syntax_error_classified_as_syntax(tmp_path: Path) -> None:
    """Test syntax/collection failure is classified as SYNTAX."""
    test_file = tmp_path / "test_broken.py"
    test_file.write_text("def broken( syntax error:\n", encoding="utf-8")

    result = await run_test(test_file, tmp_path)
    assert not result.passed
    assert result.classification == ErrorClassification.SYNTAX


@pytest.mark.asyncio
async def test_run_test_timeout_kills_process_group_and_reaps(tmp_path: Path) -> None:
    """Regression test: no timeout in original wedged forever; timeout kills and reaps."""
    test_file = tmp_path / "test_hanging.py"
    test_file.write_text("import time\ndef test_hang(): time.sleep(10)\n", encoding="utf-8")

    result = await run_test(test_file, tmp_path, timeout_seconds=0.5)
    assert not result.passed
    assert result.timed_out
    assert result.classification == ErrorClassification.TIMEOUT
    assert "timeout limit" in (result.message or "")


def test_pytest_plugin_hooks_direct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_out = tmp_path / "report.json"
    monkeypatch.setenv("GHOST_REPORT_PATH", str(report_out))

    # Test runtest report
    mock_item = MagicMock()
    mock_item.nodeid = "test_file.py::test_fn"
    mock_call = MagicMock()
    mock_call.when = "call"
    mock_call.excinfo.typename = "ValueError"
    mock_call.excinfo.value = "invalid value"

    pytest_runtest_makereport(mock_item, mock_call)

    # Test collection report
    mock_collect = MagicMock()
    mock_collect.failed = True
    mock_collect.nodeid = "bad_file.py"
    mock_collect.longrepr = "SyntaxError: invalid syntax"

    pytest_collectreport(mock_collect)

    # Test session finish
    pytest_sessionfinish(MagicMock(), exitstatus=1)

    assert report_out.is_file()
    data = json.loads(report_out.read_text(encoding="utf-8"))
    assert data["exit_code"] == 1
    assert data["failed_count"] >= 2
    types = [f["exception_type"] for f in data["failures"]]
    assert "ValueError" in types
    assert "SyntaxError" in types
