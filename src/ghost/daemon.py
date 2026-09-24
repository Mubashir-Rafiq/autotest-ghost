"""Ghost's background daemon management helpers.

Owns: PID file operations, daemon process lifecycle queries (start, stop, status),
and log tailing/following.

Does NOT: implement the file watcher itself (that is ``watcher.py``'s job) or
the command-line interface (that is ``cli.py``'s job).

Stage 10 provides the lifecycle management functions and PID tracking used by
the CLI commands. Stage 11 expands this module with flock-based mutual exclusion,
asyncio signal handling, and log rotation for the background process runner.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULT_LOG_LINES",
    "GRACE_PERIOD_SECONDS",
    "DaemonStatus",
    "daemon_log_path",
    "daemon_pid_path",
    "follow_log_stream",
    "is_daemon_running",
    "query_daemon_status",
    "start_daemon",
    "stop_daemon",
    "tail_log",
]

DEFAULT_LOG_LINES: Final[int] = 20
GRACE_PERIOD_SECONDS: Final[float] = 10.0
_POLL_INTERVAL: Final[float] = 0.5
_STARTUP_WAIT_SECONDS: Final[float] = 0.5


class DaemonStatus:
    """Snapshot of daemon state at query time."""

    __slots__ = ("log_tail", "pid", "running")

    def __init__(self, *, pid: int | None, running: bool, log_tail: list[str]) -> None:
        self.pid = pid
        self.running = running
        self.log_tail = log_tail


def daemon_pid_path(project_root: Path) -> Path:
    """Return the path to the daemon PID file."""
    return project_root / ".ghost" / "daemon.pid"


def daemon_log_path(project_root: Path) -> Path:
    """Return the path to the daemon log file."""
    return project_root / ".ghost" / "daemon.log"


def _read_pid(pid_path: Path) -> int | None:
    """Read a PID from *pid_path*, returning ``None`` if absent or invalid."""
    try:
        content = pid_path.read_text(encoding="utf-8").strip()
        return int(content) if content else None
    except (FileNotFoundError, ValueError):
        return None


def _is_process_alive(pid: int) -> bool:
    """Check whether a process with *pid* is still running."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _cleanup_pid_file(pid_path: Path) -> None:
    """Remove *pid_path* if it exists, ignoring missing file errors."""
    with contextlib.suppress(FileNotFoundError):
        pid_path.unlink()


def is_daemon_running(project_root: Path) -> tuple[bool, int | None]:
    """Check if the daemon for *project_root* is running.

    Returns ``(running, pid)``. If the PID file exists but points to a dead
    process, the stale file is cleaned up and ``(False, None)`` is returned.
    """
    pid_file = daemon_pid_path(project_root)
    pid = _read_pid(pid_file)
    if pid is None:
        return False, None

    if _is_process_alive(pid):
        return True, pid

    # Stale PID file
    _cleanup_pid_file(pid_file)
    return False, None


def query_daemon_status(project_root: Path, tail_lines: int = 10) -> DaemonStatus:
    """Build a :class:`DaemonStatus` snapshot for *project_root*."""
    running, pid = is_daemon_running(project_root)
    recent_logs = tail_log(project_root, lines=tail_lines)
    return DaemonStatus(pid=pid, running=running, log_tail=recent_logs)


def tail_log(project_root: Path, *, lines: int = DEFAULT_LOG_LINES) -> list[str]:
    """Return the last *lines* lines from the daemon log file."""
    log_file = daemon_log_path(project_root)
    if not log_file.is_file():
        return []
    try:
        content = log_file.read_text(encoding="utf-8")
        all_lines = content.splitlines()
        return all_lines[-lines:] if lines < len(all_lines) else all_lines
    except OSError:
        return []


def start_daemon(project_root: Path) -> int:
    """Spawn a background daemon process for *project_root*.

    Returns the PID of the spawned process.
    Raises :class:`RuntimeError` if the daemon is already running or crashes
    during the initial startup window.
    """
    running, pid = is_daemon_running(project_root)
    if running:
        msg = f"daemon already running (PID {pid})"
        raise RuntimeError(msg)

    ghost_dir = project_root / ".ghost"
    ghost_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "ghost.daemon", str(project_root)],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(_STARTUP_WAIT_SECONDS)
    if proc.poll() is not None:
        msg = f"daemon exited immediately with code {proc.returncode}"
        raise RuntimeError(msg)

    pid_path = daemon_pid_path(project_root)
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def stop_daemon(project_root: Path, *, grace_period: float = GRACE_PERIOD_SECONDS) -> bool:
    """Stop the running daemon for *project_root*.

    Attempts graceful termination with ``SIGTERM``, escalating to ``SIGKILL``
    if the daemon does not terminate within *grace_period* seconds.

    Returns ``True`` if the daemon was running and stopped, or ``False`` if no
    daemon was running.
    """
    running, pid = is_daemon_running(project_root)
    if not running or pid is None:
        return False

    pid_file = daemon_pid_path(project_root)

    # 1. Graceful: send SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        _cleanup_pid_file(pid_file)
        return False

    # 2. Poll for termination
    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            _cleanup_pid_file(pid_file)
            return True
        time.sleep(_POLL_INTERVAL)

    # 3. Forceful: escalate to SIGKILL
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)

    _cleanup_pid_file(pid_file)
    return True


def follow_log_stream(
    project_root: Path,
    line_callback: Callable[[str], None],
    *,
    poll_interval: float = 0.1,
) -> None:
    """Follow the daemon log file and invoke *line_callback* on new lines."""
    log_file = daemon_log_path(project_root)
    if not log_file.is_file():
        return

    with log_file.open(encoding="utf-8") as fh:
        fh.seek(0, 2)  # Seek to end of file
        while True:
            line = fh.readline()
            if line:
                line_callback(line)
            else:
                running, _ = is_daemon_running(project_root)
                if not running:
                    line_callback("\n[daemon exited]\n")
                    break
                time.sleep(poll_interval)
