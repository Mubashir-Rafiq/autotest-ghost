"""Asyncio debouncer and concurrent job queue with strict per-path execution guards.

Owns:
- Per-path debouncing using asyncio deadline-bumping.
- Async worker pool processing file events.
- Per-path concurrency exclusion so multiple workers never process the same file simultaneously.
- Lifecycle management: start, drain, cancel, and timeout-enforced stop.

Does NOT:
- Listen to OS file system events (that is ``watcher.py``'s job).
- Manage API rate limits (that is ``rate_limiter.py``'s job).
- Execute test pipelines or classification logic.

Guarantees:
- Per-path mutual exclusion: no two workers ever process the same path concurrently.
- Deadline-bumping collapses bursts into one event after the quiet window elapses.
- In-flight change safety: a change occurring during active processing queues a re-run.
- ``stop()`` cleanly cancels timers and worker tasks or drains pending work.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

__all__ = [
    "AsyncDebouncer",
    "JobQueue",
    "JobState",
    "QueuedJob",
]


class JobState(StrEnum):
    """Lifecycle state of a queued job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class QueuedJob:
    """Descriptor for a path-targeted job."""

    path: Path
    state: JobState = JobState.PENDING
    error: str | None = None
    runs: int = 0


JobHandler = Callable[[Path], Coroutine[Any, Any, None]]
OnReadyCallback = Callable[[Path], Coroutine[Any, Any, None]]


class AsyncDebouncer:
    """Per-path debouncer using asyncio deadline-bumping.

    Resets the quiet deadline whenever ``submit(path)`` is called. Only fires
    ``on_ready`` after ``delay_seconds`` has elapsed with no subsequent submissions.
    """

    def __init__(
        self,
        delay_seconds: float,
        on_ready: OnReadyCallback,
    ) -> None:
        self.delay_seconds: Final[float] = delay_seconds
        self._on_ready: Final[OnReadyCallback] = on_ready
        self._deadlines: dict[Path, float] = {}
        self._tasks: dict[Path, asyncio.Task[None]] = {}
        self._idle_event: asyncio.Event = asyncio.Event()
        self._idle_event.set()

    def submit(self, path: Path) -> None:
        """Submit a path to the debouncer, resetting its quiet window."""
        loop = asyncio.get_running_loop()
        self._deadlines[path] = loop.time() + self.delay_seconds
        self._idle_event.clear()

        existing_task = self._tasks.get(path)
        if existing_task is None or existing_task.done():
            self._tasks[path] = asyncio.create_task(
                self._debounce_loop(path),
                name=f"ghost-debounce-{path.name}",
            )

    async def _debounce_loop(self, path: Path) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                deadline = self._deadlines.get(path)
                if deadline is None:
                    return

                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)

            self._deadlines.pop(path, None)
            self._tasks.pop(path, None)
            if not self._tasks:
                self._idle_event.set()
            await self._on_ready(path)
        except asyncio.CancelledError:
            self._deadlines.pop(path, None)
            self._tasks.pop(path, None)
            if not self._tasks:
                self._idle_event.set()
            raise

    def cancel(self, path: Path | None = None) -> None:
        """Cancel pending debounce timers for a single path or all paths."""
        if path is not None:
            self._deadlines.pop(path, None)
            task = self._tasks.pop(path, None)
            if task is not None and not task.done():
                task.cancel()
        else:
            self._deadlines.clear()
            for task in self._tasks.values():
                if not task.done():
                    task.cancel()
            self._tasks.clear()

        if not self._tasks:
            self._idle_event.set()

    async def wait_idle(self) -> None:
        """Wait until all pending debounce windows have completed or been cancelled."""
        if not self._tasks:
            return
        await self._idle_event.wait()

    @property
    def pending_count(self) -> int:
        """Number of paths currently debouncing."""
        return len(self._tasks)

    @property
    def pending_paths(self) -> list[Path]:
        """List of paths currently debouncing."""
        return list(self._deadlines.keys())


class JobQueue:
    """Async worker pool with strict per-path execution guards and debouncing."""

    def __init__(
        self,
        handler: JobHandler,
        *,
        debounce_seconds: float = 1.0,
        max_workers: int = 1,
    ) -> None:
        self.handler: Final[JobHandler] = handler
        self.debounce_seconds: Final[float] = debounce_seconds
        self.max_workers: Final[int] = max(1, max_workers)

        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._debouncer: AsyncDebouncer = AsyncDebouncer(
            delay_seconds=debounce_seconds,
            on_ready=self._enqueue,
        )

        self._worker_tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._stopped = False

        # Concurrency guards:
        # - _in_flight: paths currently executing inside a worker
        # - _queued: paths currently sitting in self._queue waiting for a worker
        # - _rerun_pending: paths that changed again while in-flight
        self._in_flight: set[Path] = set()
        self._queued: set[Path] = set()
        self._rerun_pending: set[Path] = set()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._idle_event: asyncio.Event = asyncio.Event()
        self._idle_event.set()

    async def _enqueue(self, path: Path) -> None:
        async with self._lock:
            if self._stopped:
                return

            if path in self._in_flight:
                # Path is currently being processed by another worker.
                # Mark for rerun once that worker finishes; do NOT enqueue concurrently.
                self._rerun_pending.add(path)
                self._idle_event.clear()
                return

            if path not in self._queued:
                self._queued.add(path)
                self._idle_event.clear()
                self._queue.put_nowait(path)

    def submit(self, path: Path) -> None:
        """Submit a file path for debounced processing."""
        if self._stopped:
            return
        resolved = path.resolve()
        self._debouncer.submit(resolved)

    def start(self) -> None:
        """Start the worker pool."""
        if self._running or self._stopped:
            return
        self._running = True
        for _ in range(self.max_workers):
            task = asyncio.create_task(
                self._worker_loop(),
                name="ghost-worker",
            )
            self._worker_tasks.append(task)

    async def _worker_loop(self) -> None:
        while self._running and not self._stopped:
            try:
                path = await self._queue.get()
            except asyncio.CancelledError:
                break

            async with self._lock:
                self._queued.discard(path)
                self._in_flight.add(path)
                self._idle_event.clear()

            try:
                with contextlib.suppress(Exception):
                    await self.handler(path)
            except asyncio.CancelledError:
                async with self._lock:
                    self._in_flight.discard(path)
                    self._queue.task_done()
                    if not self._in_flight and not self._rerun_pending and self._queue.empty():
                        self._idle_event.set()
                raise
            finally:
                async with self._lock:
                    self._in_flight.discard(path)
                    self._queue.task_done()

                    # If an update arrived while this worker was processing, enqueue now
                    if path in self._rerun_pending:
                        self._rerun_pending.discard(path)
                        self._queued.add(path)
                        self._queue.put_nowait(path)

                    if not self._in_flight and not self._rerun_pending and self._queue.empty():
                        self._idle_event.set()

    async def drain(self) -> None:
        """Wait for debouncers, queue, in-flight work, and pending reruns to complete."""
        while self._debouncer.pending_count > 0:
            await self._debouncer.wait_idle()

        await self._queue.join()

        while True:
            async with self._lock:
                if not self._in_flight and not self._rerun_pending and self._queue.empty():
                    break
            await self._idle_event.wait()
            await self._queue.join()

    async def stop(self, *, drain: bool = False, timeout_seconds: float = 5.0) -> None:
        """Shut down the queue, debouncers, and workers."""
        if drain:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.drain(), timeout=timeout_seconds)

        self._stopped = True
        self._running = False
        self._debouncer.cancel()

        for task in self._worker_tasks:
            if not task.done():
                task.cancel()

        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

        async with self._lock:
            self._in_flight.clear()
            self._queued.clear()
            self._rerun_pending.clear()
            self._idle_event.set()

    @property
    def is_running(self) -> bool:
        """Whether the worker pool is active."""
        return self._running and not self._stopped

    @property
    def in_flight_count(self) -> int:
        """Number of paths currently being processed by workers."""
        return len(self._in_flight)

    @property
    def queued_count(self) -> int:
        """Number of paths waiting in the queue."""
        return len(self._queued)

    @property
    def debouncing_count(self) -> int:
        """Number of paths currently waiting for debounce window to elapse."""
        return self._debouncer.pending_count
