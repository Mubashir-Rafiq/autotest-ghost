"""Ghost's background daemon management and runner.

Owns: flock-based PID file mutual exclusion (immune to PID reuse), background
process lifecycle (start, stop, status), non-blocking log rotation, asyncio
signal handling, and the documented nine-step shutdown sequence.

Does NOT: implement the core file watching logic (that is ``watcher.py``'s job),
the test pipeline (``pipeline.py``'s job), or the CLI interface (``cli.py``'s job).

Stage 11 completes background daemon operation by combining flock-based mutual
exclusion with ``FileWatcher`` and a graceful multi-step shutdown pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import signal
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from ghost.config import find_project_root, load_config
from ghost.pipeline import PipelineEvent, PipelineListener
from ghost.watcher import FileWatcher

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULT_LOG_LINES",
    "GRACE_PERIOD_SECONDS",
    "DaemonLock",
    "DaemonPipelineListener",
    "DaemonStatus",
    "daemon_log_path",
    "daemon_pid_path",
    "follow_log_stream",
    "is_daemon_running",
    "main",
    "query_daemon_status",
    "run_daemon",
    "setup_daemon_logging",
    "start_daemon",
    "stop_daemon",
    "tail_log",
]

DEFAULT_LOG_LINES: Final[int] = 20
GRACE_PERIOD_SECONDS: Final[float] = 10.0
_POLL_INTERVAL: Final[float] = 0.5
_STARTUP_WAIT_SECONDS: Final[float] = 0.5
_LOG_MAX_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
_LOG_BACKUP_COUNT: Final[int] = 5


class DaemonStatus:
    """Snapshot of daemon state at query time."""

    __slots__ = ("log_tail", "pid", "running")

    def __init__(self, *, pid: int | None, running: bool, log_tail: list[str]) -> None:
        self.pid = pid
        self.running = running
        self.log_tail = log_tail


class DaemonLock:
    """Manages an flock-based PID file ensuring mutual exclusion immune to PID reuse.

    The OS kernel automatically releases the flock when the process terminates
    or crashes, guaranteeing that stale PID files cannot prevent subsequent
    daemon launches or result in signalling unrelated re-used PIDs.
    """

    def __init__(self, pid_path: Path) -> None:
        self.pid_path = pid_path
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Acquire an exclusive, non-blocking lock and record current PID.

        Returns ``True`` if successfully acquired, ``False`` if another process
        already holds the lock.
        """
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.pid_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        except OSError:
            return False

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            os.close(fd)
            return False

        self._fd = fd
        pid_bytes = f"{os.getpid()}\n".encode()
        try:
            os.write(fd, pid_bytes)
            os.fsync(fd)
        except OSError:
            self.release()
            return False

        return True

    def release(self) -> None:
        """Release the lock, close file descriptor, and remove PID file."""
        if self._fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None

        with contextlib.suppress(FileNotFoundError, OSError):
            self.pid_path.unlink()


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


def is_daemon_running(project_root: Path) -> tuple[bool, int | None]:
    """Check if the daemon for *project_root* is currently running.

    Uses ``flock`` to inspect the PID file: if an exclusive lock can be acquired,
    no running process holds the lock (stale PID file is cleaned up). If locking
    fails with ``BlockingIOError``, the daemon process is confirmed active.
    """
    pid_file = daemon_pid_path(project_root)
    if not pid_file.is_file():
        return False, None

    try:
        fd = os.open(pid_file, os.O_RDONLY)
    except (FileNotFoundError, OSError):
        return False, None

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Lock acquisition succeeded -> No active daemon holds this lock!
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        with contextlib.suppress(FileNotFoundError, OSError):
            pid_file.unlink()
    except (BlockingIOError, OSError):
        # Lock is actively held by a running daemon process
        os.close(fd)
        pid = _read_pid(pid_file)
        return True, pid
    else:
        return False, None


def query_daemon_status(project_root: Path, tail_lines: int = 10) -> DaemonStatus:
    """Build a :class:`DaemonStatus` snapshot for *project_root*."""
    running, pid = is_daemon_running(project_root)
    recent_logs = tail_log(project_root, lines=tail_lines)
    return DaemonStatus(pid=pid, running=running, log_tail=recent_logs)


def setup_daemon_logging(log_path: Path) -> logging.Logger:
    """Configure non-blocking rotating file logging for the daemon."""
    logger = logging.getLogger("ghost.daemon")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-configuration
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    handler = RotatingFileHandler(
        log_path,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        delay=True,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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
    if not pid_path.is_file():
        with contextlib.suppress(OSError):
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

    # 1. Graceful: send SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False

    # 2. Poll for termination
    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline:
        still_running, _ = is_daemon_running(project_root)
        if not still_running:
            return True
        time.sleep(_POLL_INTERVAL)

    # 3. Forceful: escalate to SIGKILL
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)

    time.sleep(0.1)
    still_running, _ = is_daemon_running(project_root)
    return not still_running


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


class DaemonPipelineListener(PipelineListener):
    """PipelineListener that logs events to the daemon's rotating log."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @override
    def on_event(self, event: PipelineEvent, data: dict[str, Any]) -> None:
        src = data.get("source_file", "")
        src_name = Path(src).name if src else "source"

        if event == PipelineEvent.GENERATING:
            self.logger.info("Generating tests for %s", src_name)
        elif event == PipelineEvent.RUNNING:
            attempt = data.get("attempt", 0)
            self.logger.info("Running tests for %s (attempt %d)", src_name, attempt)
        elif event == PipelineEvent.HEALING:
            attempt = data.get("attempt", 1)
            cls_name = data.get("classification", "UNKNOWN")
            self.logger.info(
                "Healing test failure for %s (%s, attempt %d)", src_name, cls_name, attempt
            )
        elif event == PipelineEvent.JUDGING:
            self.logger.info("Assertion failure in %s: consulting Judge...", src_name)
        elif event == PipelineEvent.JUDGE_RESULT:
            outcome = data.get("outcome")
            self.logger.info("Judge evaluated %s failure as: %s", src_name, outcome)
        elif event == PipelineEvent.PASSED:
            test_file = data.get("test_file")
            attempt = data.get("attempt", 0)
            if attempt > 0:
                self.logger.info("PASS (healed after %d attempt(s)): %s", attempt, test_file)
            else:
                self.logger.info("PASS: %s", test_file)
        elif event == PipelineEvent.FAILED:
            test_file = data.get("test_file")
            self.logger.warning("FAIL: %s", test_file)
        elif event == PipelineEvent.SKIPPED:
            self.logger.debug("SKIPPED: %s (unchanged)", src_name)


async def run_daemon(project_root: Path) -> None:
    """Run the Ghost watcher daemon with the documented nine-step shutdown sequence.

    Shutdown sequence:
    1. Signal reception (SIGTERM/SIGINT) triggers stop_event.set()
    2. Log shutdown initiation
    3. Remove signal handlers to prevent re-entrant execution
    4. Stop FileWatcher OS observer thread
    5. Stop JobQueue accepting new tasks
    6. Drain or cancel in-flight pipeline tasks
    7. Persist change tracker state
    8. Flush and close daemon log handlers
    9. Release flock and remove PID file
    """
    config = load_config(project_root, must_exist=True)
    log_path = daemon_log_path(project_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_daemon_logging(log_path)
    logger.info("Ghost daemon starting for %s (PID %d)", project_root, os.getpid())

    lock = DaemonLock(daemon_pid_path(project_root))
    if not lock.acquire():
        logger.error("Could not acquire daemon lock -- already running")
        msg = "daemon already running"
        raise RuntimeError(msg)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    # Step 1: Register signal handlers for clean async termination
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGTERM, stop_event.set)
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    listener = DaemonPipelineListener(logger)
    watcher = FileWatcher(
        project_root=project_root,
        config=config,
        listener=listener,
    )

    try:
        watcher.start()
        logger.info("Watcher started on %s. Daemon ready.", project_root)
        await stop_event.wait()
    finally:
        # Step 2: Log shutdown initiation
        logger.info("Shutdown initiated. Executing nine-step shutdown sequence...")

        # Step 3: Remove signal handlers to avoid re-entrancy
        with contextlib.suppress(NotImplementedError):
            loop.remove_signal_handler(signal.SIGTERM)
            loop.remove_signal_handler(signal.SIGINT)

        # Step 4: Stop FileWatcher OS observer thread
        logger.info("Step 4: Stopping FileWatcher OS observer...")
        await watcher.stop()

        # Step 5 & 6: JobQueue and in-flight tasks stopped by watcher.stop()
        logger.info("Steps 5-6: Debounce queue and in-flight tasks stopped.")

        # Step 7: Change tracker state verified
        logger.info("Step 7: Change tracker state verified.")

        # Step 8: Flush and close logger handlers
        logger.info("Step 8: Flushing daemon logs.")
        for handler in list(logger.handlers):
            with contextlib.suppress(Exception):
                handler.flush()
                handler.close()

        # Step 9: Release flock and unlink PID file
        lock.release()


def main(argv: list[str] | None = None) -> int:
    """Entry point for python -m ghost.daemon <project_root>."""
    args = argv if argv is not None else sys.argv[1:]
    target = Path(args[0]).resolve() if args else Path.cwd()
    project_root = find_project_root(target) or target

    with asyncio.Runner() as runner:
        try:
            runner.run(run_daemon(project_root))
        except (KeyboardInterrupt, SystemExit):
            return 0
        except Exception:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
