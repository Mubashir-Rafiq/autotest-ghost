"""Tests for ghost.daemon process management helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ghost.daemon import (
    DaemonStatus,
    daemon_log_path,
    daemon_pid_path,
    is_daemon_running,
    query_daemon_status,
    start_daemon,
    stop_daemon,
    tail_log,
)


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

    running, pid = is_daemon_running(tmp_path)
    assert running is False
    assert pid is None


def test_is_daemon_running_dead_process_cleans_up(tmp_path: Path) -> None:
    pid_file = daemon_pid_path(tmp_path)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    # 99999999 is overwhelmingly likely not an existing pid
    pid_file.write_text("99999999\n", encoding="utf-8")

    running, pid = is_daemon_running(tmp_path)
    assert running is False
    assert pid is None
    # Verify stale pid file was removed
    assert not pid_file.exists()


def test_is_daemon_running_alive_process(tmp_path: Path) -> None:
    pid_file = daemon_pid_path(tmp_path)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()
    pid_file.write_text(f"{current_pid}\n", encoding="utf-8")

    running, pid = is_daemon_running(tmp_path)
    assert running is True
    assert pid == current_pid


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


def test_start_daemon_already_running_raises(tmp_path: Path) -> None:
    pid_file = daemon_pid_path(tmp_path)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="daemon already running"):
        start_daemon(tmp_path)


def test_stop_daemon_not_running(tmp_path: Path) -> None:
    stopped = stop_daemon(tmp_path)
    assert stopped is False
