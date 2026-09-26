"""First-party pytest plugin extracting true exception classes and structured test results.

Owns:
- Intercepting pytest test run and collection events.
- Extracting the true Python exception type name directly from ``call.excinfo``.
- Writing structured JSON test summaries to the path designated by ``GHOST_REPORT_PATH``.

Does NOT:
- Classify errors (that belongs to :class:`~ghost.runner.ErrorClassification`).
- Run subprocesses or manage execution timeouts.
- Format console output.

Guarantees:
- Captures the exact runtime exception class name, eliminating flaky substring grepping.
- Emits clean, deterministic JSON with per-test failure outcomes and collection errors.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

_GHOST_FAILURES: dict[str, dict[str, Any]] = {}


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    """Record test outcome and exception type from the test run call."""
    if call.excinfo is None:
        return

    exc_type = call.excinfo.typename
    exc_val = str(call.excinfo.value)

    _GHOST_FAILURES[item.nodeid] = {
        "nodeid": item.nodeid,
        "phase": call.when,
        "outcome": "failed",
        "exception_type": exc_type,
        "message": exc_val,
    }


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Record collection errors (e.g. syntax or import errors in test files)."""
    if not report.failed:
        return

    longrepr_str = str(report.longrepr)
    exc_type = "ImportError"
    match = re.search(r"(?:^E\s+|\b)([A-Z]\w*(?:Error|Exception)):", longrepr_str, re.MULTILINE)
    if match:
        exc_type = match.group(1)
    elif "SyntaxError" in longrepr_str:
        exc_type = "SyntaxError"
    elif "ModuleNotFoundError" in longrepr_str:
        exc_type = "ModuleNotFoundError"
    elif "IndentationError" in longrepr_str:
        exc_type = "IndentationError"

    _GHOST_FAILURES[report.nodeid] = {
        "nodeid": report.nodeid,
        "phase": "collection",
        "outcome": "failed",
        "exception_type": exc_type,
        "message": longrepr_str.splitlines()[-1] if longrepr_str else "",
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Write the structured failure report to GHOST_REPORT_PATH on session completion."""
    report_path_str = os.environ.get("GHOST_REPORT_PATH")
    if not report_path_str:
        return

    failures = list(_GHOST_FAILURES.values())

    payload = {
        "exit_code": int(exitstatus),
        "failed_count": len(failures),
        "failures": failures,
    }

    with contextlib.suppress(OSError):
        Path(report_path_str).write_text(json.dumps(payload, indent=2), encoding="utf-8")
