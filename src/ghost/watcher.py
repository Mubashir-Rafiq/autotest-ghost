"""Filesystem event watcher and bridge between OS events and the async job queue.

Owns:
- Watching the filesystem using ``watchdog.Observer`` and dispatching events.
- Bridging the synchronous watchdog OS thread to the asyncio event loop via
  ``loop.call_soon_threadsafe``.
- Filtering filesystem events to ignore directories, editor swap/temp files, test files,
  and paths ignored by :class:`~ghost.config.ScannerConfig`.
- Handling atomic-rename saves (``FileMovedEvent``) so saves from vim, JetBrains IDEs,
  and external formatters are never missed.
- Distinguishing test files strictly by test filename conventions and directories,
  preventing false-positive filtering of files like ``contest.py``, ``latest.py``,
  or ``manifest.py``.
- Cleaning up removed source files from the change tracker and AST index.
- Coordinating debouncing, queue processing, and pipeline execution.

Does NOT:
- Manage OS background daemons, PID files, or detached processes (daemon.py owns this).
- Directly invoke LLM APIs or prompt builders (client.py and prompts.py own this).
- Subprocess execution or pytest execution (runner.py and pipeline.py own this).
- Render rich terminal presentation (console.py / cli.py own this).

Guarantees:
- Never misses atomic-rename saves (dest_path of FileMovedEvent is tracked).
- Never ignores valid source files containing "test" as a substring (e.g. contest.py).
- Thread-safe handoff from the watchdog observer thread to the asyncio loop.
- Clean shutdown that stops the observer thread and drains/stops the job queue.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from ghost.indexer import (
    index_file,
    should_ignore_path,
    walk_and_delete_json,
    walk_and_modify_json,
)
from ghost.job_queue import JobQueue
from ghost.pipeline import PipelineListener, PipelineResult, TestPipeline

if TYPE_CHECKING:
    from ghost.config import GhostConfig

logger = logging.getLogger(__name__)

__all__ = [
    "FileWatcher",
    "GhostEventHandler",
    "is_temporary_file",
    "is_test_file",
    "should_watch_path",
]

ResultCallback = Callable[[PipelineResult], Any | Coroutine[Any, Any, Any]]


def _to_path(raw: str | bytes) -> Path:
    """Safely convert a string or bytes filesystem path into a pathlib.Path."""
    if isinstance(raw, bytes):
        return Path(os.fsdecode(raw))
    return Path(raw)


def is_test_file(
    path: Path,
    root: Path,
    test_output_dir: str = "tests",
) -> bool:
    """Determine whether *path* is a test file or resides in a test directory.

    Guarantees:
    - Never uses naive substring matching like ``"test" in path.name``, which
      erroneously filters legitimate source files such as ``contest.py``,
      ``latest.py``, or ``manifest.py``.
    - Matches standard Python test naming conventions: ``test_*.py`` and ``*_test.py``.
    - Matches any path residing within the configured test directory or a directory named ``tests``.
    """
    resolved_path = path.resolve()
    name = resolved_path.name

    if name.startswith("test_") or name.endswith("_test.py"):
        return True

    resolved_root = root.resolve()
    try:
        resolved_out = (resolved_root / test_output_dir).resolve()
        if resolved_out in resolved_path.parents or resolved_path == resolved_out:
            return True
    except (ValueError, OSError):
        pass

    try:
        rel = resolved_path.relative_to(resolved_root)
        for part in rel.parts[:-1]:
            if part in (test_output_dir, "tests"):
                return True
    except ValueError:
        pass

    return False


def is_temporary_file(path: Path) -> bool:
    """Determine whether *path* is an editor swap file or temporary artifact."""
    name = path.name

    # Editor swap / backup / temp file patterns
    if (
        name.endswith(("~", ".tmp", ".swp", ".swo", ".py~"))
        or name.startswith(".")
        or (name.startswith("#") and name.endswith("#"))
    ):
        return True

    # Python bytecode artifacts
    if path.suffix in (".pyc", ".pyo"):
        return True

    return any(part == "__pycache__" for part in path.parts)


def should_watch_path(
    path: Path,
    root: Path,
    config: GhostConfig,
) -> bool:
    """Determine whether a filesystem path should be watched and processed.

    Checks:
    1. Not a temporary or editor swap file.
    2. Has a valid Python extension or matches ``config.watcher.patterns``.
    3. Not a test file or in a test directory.
    4. Not ignored by ``config.scanner`` (e.g. .venv, .git, node_modules, etc.).
    """
    if is_temporary_file(path):
        return False

    name = path.name
    matches_pattern = any(fnmatch.fnmatch(name, pat) for pat in config.watcher.patterns)
    if not matches_pattern and path.suffix != ".py":
        return False

    if is_test_file(path, root, config.tests.output_dir):
        return False

    return not should_ignore_path(path, root, config.scanner, is_dir=False)


class GhostEventHandler(FileSystemEventHandler):
    """Watchdog event handler that bridges OS filesystem events to an asyncio event loop."""

    def __init__(
        self,
        project_root: Path,
        config: GhostConfig,
        loop: asyncio.AbstractEventLoop,
        on_modified_callback: Callable[[Path], None],
        on_deleted_callback: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__()
        self.project_root: Final[Path] = project_root.resolve()
        self.config: Final[GhostConfig] = config
        self.loop: Final[asyncio.AbstractEventLoop] = loop
        self.on_modified_callback: Final[Callable[[Path], None]] = on_modified_callback
        self.on_deleted_callback: Final[Callable[[Path], None] | None] = on_deleted_callback

    @override
    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        if event.is_directory:
            return
        path = _to_path(event.src_path)
        if should_watch_path(path, self.project_root, self.config):
            self.loop.call_soon_threadsafe(self.on_modified_callback, path)

    @override
    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = _to_path(event.src_path)
        if should_watch_path(path, self.project_root, self.config):
            self.loop.call_soon_threadsafe(self.on_modified_callback, path)

    @override
    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        if event.is_directory:
            return

        # Defect fix: Handle atomic-rename saves (vim, JetBrains, formatters).
        # In an atomic rename, the file is written to a temporary name and then
        # renamed over the destination file.
        dest_path_raw = getattr(event, "dest_path", None)
        if dest_path_raw:
            dest_path = _to_path(dest_path_raw)
            if should_watch_path(dest_path, self.project_root, self.config):
                self.loop.call_soon_threadsafe(self.on_modified_callback, dest_path)

        # If a watched file was moved away from a watched location, handle deletion
        src_path = _to_path(event.src_path)
        if self.on_deleted_callback and should_watch_path(src_path, self.project_root, self.config):
            self.loop.call_soon_threadsafe(self.on_deleted_callback, src_path)

    @override
    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        if event.is_directory:
            return
        path = _to_path(event.src_path)
        if self.on_deleted_callback and should_watch_path(path, self.project_root, self.config):
            self.loop.call_soon_threadsafe(self.on_deleted_callback, path)


class FileWatcher:
    """Coordinates filesystem watching, debounced queuing, and test pipeline execution."""

    def __init__(
        self,
        project_root: Path,
        config: GhostConfig,
        pipeline: TestPipeline | None = None,
        *,
        listener: PipelineListener | None = None,
        on_result: ResultCallback | None = None,
        max_workers: int = 1,
        debounce_seconds: float | None = None,
    ) -> None:
        self.project_root: Final[Path] = project_root.resolve()
        self.config: Final[GhostConfig] = config
        self.pipeline: Final[TestPipeline] = pipeline or TestPipeline(
            config=config,
            project_root=self.project_root,
        )
        self.listener: Final[PipelineListener | None] = listener
        self._on_result: Final[ResultCallback | None] = on_result

        delay = (
            debounce_seconds
            if debounce_seconds is not None
            else float(config.watcher.debounce_seconds)
        )
        self.queue: Final[JobQueue] = JobQueue(
            handler=self._process_file,
            debounce_seconds=delay,
            max_workers=max_workers,
        )
        self._observer: BaseObserver | None = None
        self._event_handler: GhostEventHandler | None = None
        self._running: bool = False
        self._background_tasks: set[asyncio.Task[None]] = set()

    def is_running(self) -> bool:
        """Return True if the file watcher observer is currently active."""
        return self._running

    @property
    def observer(self) -> BaseObserver | None:
        """Return the underlying watchdog BaseObserver if started."""
        return self._observer

    async def _process_file(self, path: Path) -> None:
        """Process a modified source file through index update and test pipeline."""
        resolved = await asyncio.to_thread(path.resolve)
        is_file = await asyncio.to_thread(resolved.is_file)
        if not is_file:
            return

        # Syntax check and AST index update
        # If the file has a syntax error, index_file returns None. We skip running
        # tests on an incomplete/unparseable file until the user finishes typing.
        indexed = await asyncio.to_thread(index_file, resolved, self.project_root)
        if indexed is None:
            logger.debug("Skipping unparseable or unreadable file: %s", resolved)
            return

        await asyncio.to_thread(
            walk_and_modify_json,
            self.project_root,
            resolved,
            scanner_config=self.config.scanner,
        )

        result = await self.pipeline.run(
            resolved,
            if_changed=True,
            auto_heal=self.config.tests.auto_heal,
            use_judge=self.config.tests.use_judge,
            listener=self.listener,
        )

        if self._on_result is not None:
            cb_res = self._on_result(result)
            if asyncio.iscoroutine(cb_res):
                await cb_res

    def _on_file_deleted(self, path: Path) -> None:
        """Clean up change tracker and context index when a file is deleted."""
        task = asyncio.create_task(self._async_file_deleted(path))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _async_file_deleted(self, path: Path) -> None:
        try:
            await self.pipeline.tracker.aremove(path)
            await asyncio.to_thread(walk_and_delete_json, self.project_root, path)
        except Exception:
            logger.exception("Error handling file deletion for %s", path)

    def start(self) -> None:
        """Start the file watcher observer and job queue worker pool."""
        if self._running:
            return
        self._running = True
        loop = asyncio.get_running_loop()

        self.queue.start()

        self._event_handler = GhostEventHandler(
            project_root=self.project_root,
            config=self.config,
            loop=loop,
            on_modified_callback=self.queue.submit,
            on_deleted_callback=self._on_file_deleted,
        )

        observer = Observer()
        observer.schedule(
            self._event_handler,
            str(self.project_root),
            recursive=True,
        )
        observer.start()
        self._observer = observer

    async def stop(self, *, timeout_seconds: float = 5.0) -> None:
        """Stop the file watcher, joining the observer thread and stopping workers."""
        if not self._running:
            return
        self._running = False

        if self._observer is not None:
            self._observer.stop()
            await asyncio.to_thread(self._observer.join, timeout=timeout_seconds)
            self._observer = None

        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
            self._background_tasks.clear()

        await self.queue.stop(timeout_seconds=timeout_seconds)

    async def __aenter__(self) -> FileWatcher:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.stop()
