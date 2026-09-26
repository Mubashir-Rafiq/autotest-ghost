"""Comprehensive chaos engineering, fuzzing, and defensive edge-case test suite.

Mandate:
Validate zero-defect reliability across concurrency, boundary conditions,
malformed inputs, regex escaping, and unexpected state transitions.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import pytest
from click.testing import CliRunner

from ghost.change_tracker import ChangeTracker
from ghost.cli import _offer_save_to_env, cli
from ghost.client import (
    LLMClient,
    _prepare_generation_context,
    _prepare_healing_context,
    clean_llm_response,
    ensure_can_overwrite_test,
    is_ghost_managed_test,
    validate_test_code,
)
from ghost.config import GhostConfig
from ghost.daemon import DaemonLock, daemon_pid_path, is_daemon_running, start_daemon
from ghost.history import HistoryTracker, UsageTracker
from ghost.indexer import _write_context_atomic, walk_and_modify_json
from ghost.job_queue import JobQueue
from ghost.providers import BaseProvider
from ghost.pytest_plugin import _GHOST_FAILURES, pytest_collectreport
from ghost.runner import ErrorClassification, classify_error, run_test
from ghost.watcher import is_test_file


class DummyProvider(BaseProvider):
    def __init__(self, response: str = "") -> None:
        super().__init__()
        self.response = response

    @property
    @override
    def name(self) -> str:
        return "dummy"

    @override
    async def _call_api(
        self, messages: list[dict[str, str]], model: str, temperature: float
    ) -> str:
        return self.response

    @override
    async def list_models(self) -> list[str]:
        return ["dummy-model"]

    @override
    async def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# 1. Concurrency & Daemon PID File Mutual Exclusion
# ---------------------------------------------------------------------------
def test_daemon_lock_failed_acquire_never_truncates_pid(tmp_path: Path) -> None:
    """A secondary process attempting acquire() must not truncate the running PID."""
    pid_file = daemon_pid_path(tmp_path)
    lock1 = DaemonLock(pid_file)
    assert lock1.acquire() is True

    running_pid_str = pid_file.read_text(encoding="utf-8").strip()
    assert running_pid_str == str(os.getpid())

    # Second lock fails
    lock2 = DaemonLock(pid_file)
    assert lock2.acquire() is False

    # CRITICAL: Verify pid_file content was NOT wiped or truncated to 0 bytes
    preserved_pid_str = pid_file.read_text(encoding="utf-8").strip()
    assert preserved_pid_str == running_pid_str

    running, pid = is_daemon_running(tmp_path)
    assert running is True
    assert pid == os.getpid()

    lock1.release()
    assert not pid_file.exists()


# ---------------------------------------------------------------------------
# 2. Worker Pool & Task Cancellation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_job_queue_cancellation_no_double_task_done() -> None:
    """Cancelling in-flight workers must never raise task_done() called too many times."""
    executed = 0

    async def slow_handler(path: Path) -> None:
        nonlocal executed
        executed += 1
        await asyncio.sleep(5.0)

    jq = JobQueue(handler=slow_handler, debounce_seconds=0.01, max_workers=2)
    jq.start()

    dummy_path = Path("module.py")
    jq.submit(dummy_path)
    # Wait until debounce window expires and worker picks it up
    await asyncio.sleep(0.05)
    assert jq.in_flight_count == 1

    # Stop queue with drain=False, which cancels all active worker tasks
    await jq.stop(drain=False, timeout_seconds=1.0)

    assert jq.in_flight_count == 0
    assert jq.is_running is False


# ---------------------------------------------------------------------------
# 3. LLM Response Cleaning & Fuzzing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw_llm_output",
    [
        # Preambles and postambles
        (
            "Certainly! Here is your test code:\n"
            "```python\n"
            "import pytest\n"
            "def test_example():\n"
            "    assert True\n"
            "```\n"
            "Hope this helps you achieve 100% test coverage!"
        ),
        # Multiple markdown code blocks (e.g. explanation snippet then test)
        (
            "First, note this helper:\n"
            "```python\n"
            "x = 42\n"
            "```\n"
            "And here is the complete pytest suite:\n"
            "```python\n"
            "import pytest\n"
            "def test_answer():\n"
            "    assert 42 == 42\n"
            "```\n"
            "Done."
        ),
        # Unfenced code with leading/trailing whitespace
        (
            "\n\nimport pytest\n\n"
            "def test_bare():\n"
            "    assert 1 == 1\n\n"
        ),
        # Markdown without python language tag
        (
            "Here is the test:\n"
            "```\n"
            "import pytest\n"
            "def test_notag():\n"
            "    assert 2 == 2\n"
            "```\n"
        ),
    ],
)
def test_clean_llm_response_extracts_valid_python(raw_llm_output: str) -> None:
    cleaned = clean_llm_response(raw_llm_output)
    ast_tree = validate_test_code(cleaned)
    assert ast_tree is not None
    assert "def test_" in cleaned


# ---------------------------------------------------------------------------
# 4. Defensive Path Auditing: Source Files Outside Project Root
# ---------------------------------------------------------------------------
def test_prepare_context_with_external_source_path(tmp_path: Path) -> None:
    """Source files outside project_root must not crash with ValueError."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "ghost.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    cfg = GhostConfig()

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_file = external_dir / "external_source.py"
    external_file.write_text("def external_fn(): return 1\n", encoding="utf-8")

    # Generation context
    code, rel, _tree, _bud = _prepare_generation_context(external_file, root, cfg)
    assert rel == "external_source.py"
    assert "def external_fn" in code

    # Healing context
    test_file = root / "tests" / "test_external.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_ext(): assert True\n", encoding="utf-8")

    code_h, test_h, rel_h, _tree_h, _bud_h = _prepare_healing_context(
        external_file, test_file, root, cfg
    )
    assert rel_h == "external_source.py"
    assert "def external_fn" in code_h
    assert "def test_ext" in test_h


# ---------------------------------------------------------------------------
# 5. Concurrent ChangeTracker Stress Testing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_concurrent_change_tracker_writes(tmp_path: Path) -> None:
    """Multiple concurrent workers updating hashes must not clash on temp files."""
    tracker = ChangeTracker(tmp_path)

    async def worker(worker_id: int) -> None:
        file_path = tmp_path / f"worker_file_{worker_id}.py"
        content = f"value = {worker_id}\n"
        for _ in range(5):
            await tracker.amark_processed(file_path, content)
            await asyncio.sleep(0.001)

    tasks = [worker(i) for i in range(8)]
    await asyncio.gather(*tasks)

    hashes = tracker.load_hashes()
    assert len(hashes) == 8
    for i in range(8):
        key = f"worker_file_{i}.py"
        assert key in hashes


# ---------------------------------------------------------------------------
# 6. CLI Boundary Value Analysis & Regex Escaping
# ---------------------------------------------------------------------------
def test_offer_save_to_env_with_regex_escape_characters(tmp_path: Path) -> None:
    """Keys containing backslashes (e.g. \\1, \\g) must not crash re.sub."""
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=old_dummy\nOTHER_VAR=123\n", encoding="utf-8")

    runner = CliRunner()
    # Enter 'y' to confirm saving
    key_with_escapes = r"gsk_secret\1\g<0>\test"
    with runner.isolation(input="y\n"):
        _offer_save_to_env(tmp_path, "GROQ_API_KEY", key_with_escapes)

    content = env_file.read_text(encoding="utf-8")
    assert key_with_escapes in content
    assert "OTHER_VAR=123" in content


def test_cli_option_boundary_validation() -> None:
    """Negative and zero bounds for CLI numerical options must be rejected cleanly."""
    runner = CliRunner()

    # Negative timeout in run-tests
    res_run = runner.invoke(cli, ["run-tests", "tests/test_cli.py", "--timeout", "-1"])
    assert res_run.exit_code != 0
    assert "Invalid value for '--timeout'" in res_run.output or "not in the range" in res_run.output

    # Zero timeout in generate
    res_gen = runner.invoke(cli, ["generate", "src/ghost/cli.py", "--timeout", "0"])
    assert res_gen.exit_code != 0
    assert "Invalid value for '--timeout'" in res_gen.output or "not in the range" in res_gen.output

    # Negative budget in index
    res_idx = runner.invoke(cli, ["index", "--budget", "-5"])
    assert res_idx.exit_code != 0
    assert "Invalid value for '--budget'" in res_idx.output or "not in the range" in res_idx.output

    # Negative lines in logs
    res_log = runner.invoke(cli, ["logs", "-n", "0"])
    assert res_log.exit_code != 0
    assert "Invalid value for '--lines'" in res_log.output or "not in the range" in res_log.output

    # Negative attempt in rollback
    res_rb = runner.invoke(cli, ["rollback", "tests/test_cli.py", "--attempt", "-1"])
    assert res_rb.exit_code != 0
    assert "Invalid value for" in res_rb.output or "not in the range" in res_rb.output


def test_doctor_tolerates_invalid_ghost_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ghost doctor must report invalid ghost.toml without crashing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[project]\nlanguage = 9999\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code in {0, 1}
    # Must not crash with unhandled ConfigError traceback
    assert "ghost.toml           invalid" in result.output
    assert "Required dependencies" in result.output


# ---------------------------------------------------------------------------
# 7. Nested Test Directory Detection
# ---------------------------------------------------------------------------
def test_is_test_file_nested_output_dir(tmp_path: Path) -> None:
    """Nested test output directories (e.g. tests/unit) must be recognized."""
    root = tmp_path
    nested_test = root / "custom_suite" / "unit" / "test_math.py"
    assert is_test_file(nested_test, root, test_output_dir="custom_suite/unit") is True

    # Non-test in regular directory
    src_file = root / "src" / "math_pkg" / "contest.py"
    assert is_test_file(src_file, root, test_output_dir="custom_suite/unit") is False


# ---------------------------------------------------------------------------
# 8. Error Classification & Pytest Plugin Collection Heuristics
# ---------------------------------------------------------------------------
def test_classify_error_syntax_over_assertion_substring() -> None:
    """Syntax errors in classes/functions named Assertion* must classify as SYNTAX."""
    res = classify_error(
        None,
        stdout="class AssertionHelper:\n    def run(): = 1\n",
        stderr="SyntaxError: invalid syntax",
    )
    assert res == ErrorClassification.SYNTAX


class DummyCollectReport:
    def __init__(self, nodeid: str, *, failed: bool, longrepr: str) -> None:
        self.nodeid = nodeid
        self.failed = failed
        self.longrepr = longrepr


def test_pytest_plugin_collectreport_extracts_true_exception() -> None:
    """Collection failures with IndentationError or NameError are correctly extracted."""
    _GHOST_FAILURES.clear()
    dummy_report = DummyCollectReport(
        nodeid="tests/test_broken.py",
        failed=True,
        longrepr="E   IndentationError: unexpected indent (line 12)",
    )
    pytest_collectreport(dummy_report)  # type: ignore[arg-type]

    entry = _GHOST_FAILURES.get("tests/test_broken.py")
    assert entry is not None
    assert entry["exception_type"] == "IndentationError"


# ---------------------------------------------------------------------------
# 9. LLM Header Injection on Omission (Defensive Overwrite Protection)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_client_header_auto_prepended_when_omitted_by_llm(tmp_path: Path) -> None:
    """When LLM omits the header, client auto-prepends it to avoid handwritten lockouts."""
    src = tmp_path / "calc.py"
    src.write_text("def add(a, b): return a + b\n", encoding="utf-8")

    # LLM outputs raw code with NO header comment
    raw_code = "import pytest\n\ndef test_add():\n    assert 1 + 1 == 2\n"
    provider = DummyProvider(response=raw_code)
    client = LLMClient(provider=provider, config=GhostConfig())

    generated = await client.generate_test(source_path=src, project_root=tmp_path)
    assert "# Generated at:" in generated
    assert "def test_add():" in generated

    # Verify that a test file written with this code is recognized as Ghost-managed
    test_file = tmp_path / "test_calc.py"
    test_file.write_text(generated, encoding="utf-8")
    assert is_ghost_managed_test(test_file) is True
    # And overwrite protection does not block healing/regeneration
    ensure_can_overwrite_test(test_file, force=False)

    # Same guarantee for heal_test
    healed = await client.heal_test(
        source_path=src,
        test_file=test_file,
        error_output="AssertionError",
        project_root=tmp_path,
    )
    assert "# Generated at:" in healed


# ---------------------------------------------------------------------------
# 10. Indexer Path Traversal & Atomic Write Concurrency
# ---------------------------------------------------------------------------
def test_indexer_walk_and_modify_json_outside_root(tmp_path: Path) -> None:
    """Modifying a file outside the project root must not crash with ValueError."""
    root = tmp_path / "project"
    root.mkdir()
    outside_dir = tmp_path / "external"
    outside_dir.mkdir()
    outside_file = outside_dir / "helper.py"
    outside_file.write_text("def helper(): pass\n", encoding="utf-8")

    # Must complete successfully without ValueError from relative_to()
    res = walk_and_modify_json(root, outside_file)
    assert res is not None
    assert outside_file.name in res or outside_file.as_posix() in res


def test_indexer_atomic_context_writes_under_concurrency(tmp_path: Path) -> None:
    """Concurrent atomic writes to context.json must never leave corrupt JSON."""
    context_file = tmp_path / "context.json"
    errors: list[Exception] = []

    def writer(idx: int) -> None:
        try:
            for i in range(10):
                data = {f"file_{idx}_{i}.py": f"summary {idx} {i}"}
                _write_context_atomic(context_file, data)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert context_file.is_file()
    # Content must parse cleanly as valid JSON
    loaded = json.loads(context_file.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)


# ---------------------------------------------------------------------------
# 11. Thread-Safe History & Usage Tracking
# ---------------------------------------------------------------------------
def test_history_and_usage_tracker_concurrency(tmp_path: Path) -> None:
    """Concurrent record_attempt and record_call invocations must be race-free."""
    hist = HistoryTracker(tmp_path)
    usage = UsageTracker(tmp_path)
    src = tmp_path / "app.py"
    src.write_text("def run(): pass\n", encoding="utf-8")

    errors: list[Exception] = []

    def record_worker(worker_id: int) -> None:
        try:
            for i in range(10):
                hist.record_attempt(
                    src,
                    attempt=worker_id * 10 + i,
                    test_code=f"def test_{worker_id}_{i}(): pass",
                    status="passed",
                )
                usage.record_call(prompt_tokens=10, completion_tokens=5)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=record_worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    records = hist.get_history(src)
    assert len(records) == 40

    final_usage = usage.load_usage()
    assert final_usage.total_requests == 40
    assert final_usage.prompt_tokens == 400
    assert final_usage.completion_tokens == 200
    assert final_usage.total_tokens == 600


# ---------------------------------------------------------------------------
# 12. Daemon Startup Crash and Exit Detection
# ---------------------------------------------------------------------------
def test_start_daemon_detects_immediate_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """start_daemon must raise RuntimeError immediately if the spawned process crashes."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1
    mock_proc.returncode = 1
    monkeypatch.setattr("subprocess.Popen", lambda *_args, **_kwargs: mock_proc)
    monkeypatch.setattr("ghost.daemon.is_daemon_running", lambda _root: (False, None))

    with pytest.raises(RuntimeError, match="daemon exited immediately with code 1"):
        start_daemon(tmp_path)


# ---------------------------------------------------------------------------
# 13. Runner Subprocess Timeout & Hanging Pipe Safety
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runner_timeout_deadlock_prevention(tmp_path: Path) -> None:
    """A hanging test exceeding timeout must terminate cleanly and report TIMEOUT."""
    hanging_test = tmp_path / "test_hanging.py"
    hanging_test.write_text(
        "import time\n"
        "def test_hang():\n"
        "    time.sleep(5.0)\n",
        encoding="utf-8",
    )

    res = await run_test(hanging_test, tmp_path, timeout_seconds=0.2)
    assert res.passed is False
    assert res.timed_out is True
    assert res.classification == ErrorClassification.TIMEOUT
    assert res.message is not None
    assert "timeout" in res.message.lower()
