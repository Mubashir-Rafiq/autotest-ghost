"""Tests for ghost.daemon process management and background runner."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from ghost.config import write_default_config
from ghost.daemon import (
    DaemonLock,
    DaemonPipelineListener,
    DaemonStatus,
    daemon_log_path,
    daemon_pid_path,
    follow_log_stream,
    is_daemon_running,
    main,
    query_daemon_status,
    run_daemon,
    setup_daemon_logging,
    start_daemon,
    stop_daemon,
    tail_log,
)
from ghost.pipeline import PipelineEvent
from ghost.watcher import FileWatcher


def test_daemon_paths(tmp_path: Path) -> None:
    pid_path = daemon_pid_path(tmp_path)
    log_path = daemon_log_path(tmp_path)
    assert pid_path == tmp_path / ".ghost" / "daemon.pid"
    assert log_path == tmp_path / ".ghost" / "daemon.log"


def test_is_daemon_running_no_file(tmp_path: Path) -> None:
    running, pid = is_daemon_running(tmp_path)
    assert running is False
    assert pid is None


def test_is_daemon_running_invalid_pid(tmp_path: Path) -> None:
    pid_file = daemon_pid_path(tmp_path)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("not-a-pid\n", encoding="utf-8")

    # Without an active flock, an unheld PID file is treated as stale and cleaned up
    running, pid = is_daemon_running(tmp_path)
    assert running is False
    assert pid is None
    assert not pid_file.exists()


def test_daemon_lock_mutual_exclusion(tmp_path: Path) -> None:
    pid_file = daemon_pid_path(tmp_path)
    lock1 = DaemonLock(pid_file)
    assert lock1.acquire() is True
    assert pid_file.is_file()
    assert int(pid_file.read_text(encoding="utf-8").strip()) == os.getpid()

    # Confirms daemon is reported running while lock is held
    running, pid = is_daemon_running(tmp_path)
    assert running is True
    assert pid == os.getpid()

    # Second lock acquisition attempt fails because lock1 holds flock
    lock2 = DaemonLock(pid_file)
    assert lock2.acquire() is False

    # Release first lock
    lock1.release()
    assert not pid_file.exists()

    # Now lock2 can acquire
    assert lock2.acquire() is True
    lock2.release()
    assert not pid_file.exists()


def test_tail_log_missing_file(tmp_path: Path) -> None:
    lines = tail_log(tmp_path, lines=10)
    assert lines == []


def test_tail_log_existing_file(tmp_path: Path) -> None:
    log_file = daemon_log_path(tmp_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join(f"line {i}" for i in range(25)) + "\n", encoding="utf-8")

    lines = tail_log(tmp_path, lines=5)
    assert len(lines) == 5
    assert lines == ["line 20", "line 21", "line 22", "line 23", "line 24"]


def test_query_daemon_status(tmp_path: Path) -> None:
    log_file = daemon_log_path(tmp_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("info: watcher active\n", encoding="utf-8")

    status: DaemonStatus = query_daemon_status(tmp_path, tail_lines=5)
    assert status.running is False
    assert status.pid is None
    assert status.log_tail == ["info: watcher active"]


def test_start_daemon_already_running_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate daemon currently running via is_daemon_running
    monkeypatch.setattr("ghost.daemon.is_daemon_running", lambda _root: (True, 9999))

    with pytest.raises(RuntimeError, match="daemon already running"):
        start_daemon(tmp_path)


def test_stop_daemon_not_running(tmp_path: Path) -> None:
    stopped = stop_daemon(tmp_path)
    assert stopped is False


def test_daemon_pipeline_listener_logs_events(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    logger = setup_daemon_logging(log_file)
    listener = DaemonPipelineListener(logger)

    listener.on_event(PipelineEvent.GENERATING, {"source_file": "math_pkg/calc.py"})
    listener.on_event(PipelineEvent.RUNNING, {"source_file": "math_pkg/calc.py", "attempt": 0})
    listener.on_event(
        PipelineEvent.HEALING,
        {"source_file": "math_pkg/calc.py", "attempt": 1, "classification": "SYNTAX"},
    )
    listener.on_event(PipelineEvent.JUDGING, {"source_file": "math_pkg/calc.py"})
    listener.on_event(
        PipelineEvent.JUDGE_RESULT, {"source_file": "math_pkg/calc.py", "outcome": "TEST_DEFECT"}
    )
    listener.on_event(
        PipelineEvent.PASSED,
        {"source_file": "math_pkg/calc.py", "test_file": "test_calc.py", "attempt": 1},
    )
    listener.on_event(
        PipelineEvent.PASSED,
        {"source_file": "math_pkg/calc.py", "test_file": "test_calc.py", "attempt": 0},
    )
    listener.on_event(
        PipelineEvent.FAILED, {"source_file": "math_pkg/calc.py", "test_file": "test_calc.py"}
    )
    listener.on_event(PipelineEvent.SKIPPED, {"source_file": "math_pkg/calc.py"})

    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    content = log_file.read_text(encoding="utf-8")
    assert "Generating tests for calc.py" in content
    assert "Running tests for calc.py (attempt 0)" in content
    assert "Healing test failure for calc.py (SYNTAX, attempt 1)" in content
    assert "Assertion failure in calc.py: consulting Judge..." in content
    assert "Judge evaluated calc.py failure as: TEST_DEFECT" in content
    assert "PASS (healed after 1 attempt(s)): test_calc.py" in content
    assert "PASS: test_calc.py" in content
    assert "FAIL: test_calc.py" in content


@pytest.mark.asyncio
async def test_run_daemon_lifecycle_and_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_default_config(tmp_path)

    watcher_started = False
    watcher_stopped = False

    def mock_start(self: FileWatcher) -> None:
        nonlocal watcher_started
        watcher_started = True

    async def mock_stop(self: FileWatcher) -> None:
        nonlocal watcher_stopped
        watcher_stopped = True

    monkeypatch.setattr(FileWatcher, "start", mock_start)
    monkeypatch.setattr(FileWatcher, "stop", mock_stop)

    # Launch run_daemon task in background and simulate termination event
    daemon_task = asyncio.create_task(run_daemon(tmp_path))

    # Give the task a moment to initialize and acquire lock
    await asyncio.sleep(0.1)

    assert watcher_started is True
    pid_file = daemon_pid_path(tmp_path)
    assert pid_file.is_file()

    running, pid = is_daemon_running(tmp_path)
    assert running is True
    assert pid == os.getpid()

    # Cancel the wait to trigger the nine-step shutdown sequence
    daemon_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await daemon_task

    # Verify step 4 (watcher stopped) and step 9 (lock released and pid unlinked)
    assert watcher_stopped is True
    assert not pid_file.exists()


@pytest.mark.asyncio
async def test_run_daemon_already_running_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_default_config(tmp_path)

    lock = DaemonLock(daemon_pid_path(tmp_path))
    assert lock.acquire() is True

    try:
        with pytest.raises(RuntimeError, match="daemon already running"):
            await run_daemon(tmp_path)
    finally:
        lock.release()


def test_daemon_main_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_default_config(tmp_path)

    async def dummy_run(_root: Path) -> None:
        return

    monkeypatch.setattr("ghost.daemon.run_daemon", dummy_run)

    exit_code = main([str(tmp_path)])
    assert exit_code == 0


def test_follow_log_stream_reads_and_stops_on_exit(tmp_path: Path) -> None:
    log_file = daemon_log_path(tmp_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("line a\nline b\n", encoding="utf-8")

    captured: list[str] = []

    def callback(line: str) -> None:
        captured.append(line)

    # Daemon is not running, so follow_log_stream notices dead daemon and exits
    follow_log_stream(tmp_path, callback, poll_interval=0.01)
    assert any("[daemon exited]" in c for c in captured)
