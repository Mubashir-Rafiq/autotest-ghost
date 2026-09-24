"""Subprocess test runner, timeout enforcement, process group reaping, and error classification.

Owns:
- Executing pytest suites in an isolated subprocess with process-group boundaries.
- Enforcing hard execution timeouts with process-group kill (SIGTERM -> SIGKILL) and zombie reaping.
- Discovering project-specific virtualenv Python interpreters.
- O(1) error classification lookup via true exception types extracted by the pytest plugin.
- Fallback heuristic classification when structured reports are unavailable.

Does NOT:
- Implement the retry/heal state machine (that belongs to :class:`~ghost.pipeline.TestPipeline`).
- Call LLM providers or build prompts.
- Overwrite source files.

Guarantees:
- Passing tests (return code 0) are never misclassified or sent for healing.
- Infinite loops or hanging subprocesses are guaranteed killed and reaped; never wedges.
- Subprocess execution inherits the target project's virtualenv dependencies when present.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

__all__ = [
    "ErrorClassification",
    "TestRunResult",
    "classify_error",
    "find_project_python",
    "run_test",
]


class ErrorClassification(StrEnum):
    """Classification of test failure causes."""

    SYNTAX = "SYNTAX"
    RUNTIME = "RUNTIME"
    LOGIC = "LOGIC"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


_EXCEPTION_LOOKUP: Final[dict[str, ErrorClassification]] = {
    # Syntax & import errors -> SYNTAX (healed by LLM)
    "SyntaxError": ErrorClassification.SYNTAX,
    "IndentationError": ErrorClassification.SYNTAX,
    "TabError": ErrorClassification.SYNTAX,
    "ModuleNotFoundError": ErrorClassification.SYNTAX,
    "ImportError": ErrorClassification.SYNTAX,
    # Assertion errors -> LOGIC (evaluated by Judge)
    "AssertionError": ErrorClassification.LOGIC,
    # Runtime exceptions -> RUNTIME (healed by LLM)
    "AttributeError": ErrorClassification.RUNTIME,
    "TypeError": ErrorClassification.RUNTIME,
    "ValueError": ErrorClassification.RUNTIME,
    "NameError": ErrorClassification.RUNTIME,
    "UnboundLocalError": ErrorClassification.RUNTIME,
    "IndexError": ErrorClassification.RUNTIME,
    "KeyError": ErrorClassification.RUNTIME,
    "ZeroDivisionError": ErrorClassification.RUNTIME,
    "FileNotFoundError": ErrorClassification.RUNTIME,
    "PermissionError": ErrorClassification.RUNTIME,
    "IsADirectoryError": ErrorClassification.RUNTIME,
    "NotADirectoryError": ErrorClassification.RUNTIME,
    "RecursionError": ErrorClassification.RUNTIME,
    "OverflowError": ErrorClassification.RUNTIME,
    "RuntimeError": ErrorClassification.RUNTIME,
    "StopIteration": ErrorClassification.RUNTIME,
    "LookupError": ErrorClassification.RUNTIME,
    "ArithmeticError": ErrorClassification.RUNTIME,
    "OSError": ErrorClassification.RUNTIME,
}


def classify_error(
    exception_type: str | None,
    stdout: str = "",
    stderr: str = "",
) -> ErrorClassification:
    """Classify a test failure using direct exception type lookup with substring fallback."""
    if exception_type and exception_type in _EXCEPTION_LOOKUP:
        return _EXCEPTION_LOOKUP[exception_type]

    combined = f"{exception_type or ''}\n{stdout}\n{stderr}"
    if "Assertion" in combined:
        return ErrorClassification.LOGIC

    syntax_markers = (
        "IndentationError",
        "SyntaxError",
        "ModuleNotFoundError",
        "ImportError",
        "Syntax",
        "Import",
    )
    if any(marker in combined for marker in syntax_markers):
        return ErrorClassification.SYNTAX

    runtime_markers = (
        "AttributeError",
        "TypeError",
        "ValueError",
        "RuntimeError",
        "NameError",
        "Error",
        "Exception",
    )
    if any(marker in combined for marker in runtime_markers):
        return ErrorClassification.RUNTIME

    return ErrorClassification.UNKNOWN


def find_project_python(project_root: Path) -> Path:
    """Find the target project's Python interpreter to preserve its installed dependencies."""
    is_win = sys.platform == "win32"
    sub_bin = "Scripts" if is_win else "bin"
    py_name = "python.exe" if is_win else "python"

    candidates = [
        project_root / ".venv" / sub_bin / py_name,
        project_root / "venv" / sub_bin / py_name,
    ]

    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand

    env_venv = os.environ.get("VIRTUAL_ENV")
    if env_venv:
        venv_py = Path(env_venv) / sub_bin / py_name
        if venv_py.is_file() and os.access(venv_py, os.X_OK):
            return venv_py

    return Path(sys.executable)


def _prepare_runner_environment(
    test_file: Path,
    project_root: Path,
    python_executable: Path | None,
    env_vars: dict[str, str] | None,
) -> tuple[Path, Path, Path, dict[str, str]]:
    resolved_test = test_file.resolve()
    resolved_root = project_root.resolve()
    python_bin = python_executable or find_project_python(resolved_root)

    env = dict(os.environ)
    if env_vars:
        env.update(env_vars)

    ghost_src_dir = str(Path(__file__).resolve().parent.parent)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{resolved_root}:{ghost_src_dir}:{existing_pp}"
        if existing_pp
        else f"{resolved_root}:{ghost_src_dir}"
    )

    return resolved_test, resolved_root, python_bin, env


@dataclass(frozen=True)
class TestRunResult:
    """Structured outcome of running a test file in a subprocess."""

    test_file: Path
    passed: bool
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    classification: ErrorClassification | None = None
    exception_type: str | None = None
    message: str | None = None


async def run_test(
    test_file: Path,
    project_root: Path,
    *,
    timeout_seconds: float = 30.0,
    python_executable: Path | None = None,
    env_vars: dict[str, str] | None = None,
) -> TestRunResult:
    """Run a pytest test file asynchronously with timeout and process-group isolation."""
    resolved_test, resolved_root, python_bin, env = await asyncio.to_thread(
        _prepare_runner_environment, test_file, project_root, python_executable, env_vars
    )

    with tempfile.TemporaryDirectory(prefix="ghost_runner_") as tmp_dir:
        report_file = Path(tmp_dir) / "ghost_report.json"
        env["GHOST_REPORT_PATH"] = str(report_file)

        cmd = [
            str(python_bin),
            "-m",
            "pytest",
            str(resolved_test),
            "-p",
            "ghost.pytest_plugin",
            "-o",
            "addopts=",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(resolved_root),
            env=env,
            start_new_session=True,
        )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            timed_out = True
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except TimeoutError:
                    os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout_bytes, stderr_bytes = await proc.communicate()

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if timed_out:
            return TestRunResult(
                test_file=resolved_test,
                passed=False,
                return_code=-9,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                classification=ErrorClassification.TIMEOUT,
                exception_type="TimeoutError",
                message=f"test exceeded timeout limit of {timeout_seconds:.1f}s",
            )

        return_code = proc.returncode if proc.returncode is not None else 1
        if return_code == 0:
            return TestRunResult(
                test_file=resolved_test,
                passed=True,
                return_code=0,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                classification=None,
                exception_type=None,
                message=None,
            )

        # Parse structured report emitted by pytest_plugin
        exception_type: str | None = None
        message: str | None = None

        if report_file.is_file():
            try:
                report_data = json.loads(report_file.read_text(encoding="utf-8"))
                failures = report_data.get("failures", [])
                if failures:
                    first_failure = failures[0]
                    exception_type = first_failure.get("exception_type")
                    message = first_failure.get("message")
            except (json.JSONDecodeError, OSError):
                pass

        classification = classify_error(exception_type, stdout=stdout, stderr=stderr)

        return TestRunResult(
            test_file=resolved_test,
            passed=False,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
            classification=classification,
            exception_type=exception_type,
            message=message,
        )
