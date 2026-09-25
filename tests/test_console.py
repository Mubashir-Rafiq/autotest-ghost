"""Tests for ghost.console rich terminal presentation layer."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from ghost.console import (
    BANNER,
    RichPipelineListener,
    get_console,
    print_banner,
    print_batch_summary,
    print_error,
    print_history_table,
    print_info,
    print_panel,
    print_providers_table,
    print_success,
    print_syntax,
    print_usage_stats,
    print_warning,
)
from ghost.history import AttemptRecord, UsageRecord
from ghost.pipeline import PipelineEvent, PipelineResult, PipelineStatus
from ghost.providers import POPULAR_MODELS
from ghost.runner import TestRunResult


def test_get_console_returns_singleton() -> None:
    c1 = get_console()
    c2 = get_console()
    assert c1 is c2
    assert isinstance(c1, Console)


def test_print_banner() -> None:
    buf = io.StringIO()
    c = Console(file=buf, force_terminal=True, color_system="standard")
    print_banner(c=c)
    out = buf.getvalue()
    assert "Ghost" in out or "ghost" in out or "____" in out
    assert "Self-healing" in out
    assert BANNER is not None


def test_print_status_helpers() -> None:
    buf = io.StringIO()
    c = Console(file=buf, force_terminal=True, color_system="standard")

    print_success("Operation succeeded", c=c)
    print_error("Operation failed", c=c)
    print_warning("Operation warning", c=c)
    print_info("Operation info", c=c)

    out = buf.getvalue()
    assert "Operation succeeded" in out
    assert "Operation failed" in out
    assert "Operation warning" in out
    assert "Operation info" in out


def test_print_panel() -> None:
    buf = io.StringIO()
    c = Console(file=buf, force_terminal=True, color_system="standard")
    print_panel("Panel body content", title="Status", c=c)
    out = buf.getvalue()
    assert "Panel body content" in out
    assert "Status" in out


def test_print_syntax() -> None:
    buf = io.StringIO()
    c = Console(file=buf, no_color=True, color_system=None)
    code = "def test_add():\n    assert 1 + 1 == 2\n"
    print_syntax(code, c=c)
    out = buf.getvalue()
    assert "test_add" in out
    assert "assert 1 + 1 == 2" in out


def test_print_providers_table() -> None:
    buf = io.StringIO()
    c = Console(file=buf, force_terminal=True, color_system="standard")
    availability = {"groq": True, "openai": False, "ollama": False}
    print_providers_table(availability, POPULAR_MODELS, c=c)
    out = buf.getvalue()
    assert "Supported Providers" in out
    assert "Popular Models" in out
    assert "groq" in out
    assert "available" in out
    assert "not configured" in out


def test_rich_pipeline_listener_events() -> None:
    buf = io.StringIO()
    c = Console(file=buf, no_color=True, color_system=None)
    listener = RichPipelineListener(console=c, verbose=True, show_results=True)

    test_file = Path("tests/test_example.py")
    source_file = Path("src/example.py")

    listener.on_event(PipelineEvent.GENERATING, {"source_file": source_file})
    listener.on_event(PipelineEvent.RUNNING, {"source_file": source_file, "attempt": 0})
    listener.on_event(PipelineEvent.RUNNING, {"source_file": source_file, "attempt": 1})
    listener.on_event(
        PipelineEvent.HEALING,
        {"source_file": source_file, "attempt": 1, "classification": "ASSERTION"},
    )
    listener.on_event(PipelineEvent.JUDGING, {"source_file": source_file})
    listener.on_event(
        PipelineEvent.JUDGE_RESULT, {"source_file": source_file, "outcome": "TEST_DEFECT"}
    )
    listener.on_event(
        PipelineEvent.PASSED,
        {"source_file": source_file, "test_file": test_file, "attempt": 1},
    )
    listener.on_event(
        PipelineEvent.PASSED,
        {"source_file": source_file, "test_file": test_file, "attempt": 0},
    )
    listener.on_event(
        PipelineEvent.FAILED,
        {"source_file": source_file, "test_file": test_file},
    )
    listener.on_event(PipelineEvent.SKIPPED, {"source_file": source_file})

    out = buf.getvalue()
    assert "Generating tests for example.py" in out
    assert "Running tests..." in out
    assert "Re-running test (attempt 1)..." in out
    assert "Healing failure (ASSERTION, attempt 1)" in out
    assert "Assertion failure: consulting Judge..." in out
    assert "Judge evaluated failure as: TEST_DEFECT" in out
    assert "PASS" in out
    assert "healed after 1 attempt(s)" in out
    assert "FAIL" in out
    assert "SKIPPED: example.py (unchanged)" in out


def test_print_batch_summary() -> None:
    buf = io.StringIO()
    c = Console(file=buf, no_color=True, color_system=None)

    results = [
        PipelineResult(
            source_file=Path("src/a.py"),
            test_file=Path("tests/test_a.py"),
            status=PipelineStatus.PASSED,
            passed=True,
            attempts=0,
            last_run=TestRunResult(
                test_file=Path("tests/test_a.py"),
                passed=True,
                return_code=0,
                stdout="",
                stderr="",
                coverage_summary="TOTAL 10 0 100%",
            ),
        ),
        PipelineResult(
            source_file=Path("src/b.py"),
            test_file=Path("tests/test_b.py"),
            status=PipelineStatus.HEALED,
            passed=True,
            attempts=1,
        ),
        PipelineResult(
            source_file=Path("src/c.py"),
            test_file=Path("tests/test_c.py"),
            status=PipelineStatus.FAILED,
            passed=False,
            attempts=3,
            error_message="timeout",
        ),
        PipelineResult(
            source_file=Path("src/d.py"),
            test_file=Path("tests/test_d.py"),
            status=PipelineStatus.SKIPPED,
            passed=True,
            attempts=0,
        ),
    ]

    print_batch_summary(results, c=c)
    out = buf.getvalue()
    assert "Batch Generation Summary" in out
    assert "a.py" in out
    assert "b.py" in out
    assert "c.py" in out
    assert "d.py" in out
    assert "Total: 4" in out
    assert "Passed: 2" in out
    assert "Failed: 1" in out
    assert "Skipped: 1" in out


def test_print_history_table() -> None:
    buf = io.StringIO()
    c = Console(file=buf, no_color=True, color_system=None)

    records = [
        AttemptRecord(
            attempt=0,
            timestamp="2026-09-25T11:00:00Z",
            status="generated",
            snapshot_file="attempt_0.py",
        ),
        AttemptRecord(
            attempt=1,
            timestamp="2026-09-25T11:05:00Z",
            status="healed",
            classification="LOGIC",
            snapshot_file="attempt_1.py",
        ),
    ]

    print_history_table(records, "math.py", c=c)
    out = buf.getvalue()
    assert "History for math.py" in out
    assert "attempt_0.py" in out
    assert "attempt_1.py" in out
    assert "LOGIC" in out


def test_print_usage_stats() -> None:
    buf = io.StringIO()
    c = Console(file=buf, no_color=True, color_system=None)

    usage = UsageRecord(
        total_requests=10,
        prompt_tokens=1500,
        completion_tokens=600,
        total_tokens=2100,
        heal_attempts=3,
        healed_tests=2,
    )

    print_usage_stats(usage, c=c)
    out = buf.getvalue()
    assert "AI Usage & Token Statistics" in out
    assert "1,500" in out
    assert "600" in out
    assert "2,100" in out
    assert "2" in out
