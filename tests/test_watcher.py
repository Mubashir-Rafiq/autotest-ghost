"""Tests for filesystem watcher, event filtering, atomic renames, and job queue integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from watchdog.events import (
    DirCreatedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from ghost.config import GhostConfig, ScannerConfig, WatcherConfig
from ghost.pipeline import PipelineResult, PipelineStatus, TestPipeline
from ghost.runner import TestRunResult
from ghost.watcher import (
    FileWatcher,
    GhostEventHandler,
    is_temporary_file,
    is_test_file,
    should_watch_path,
)


def test_is_test_file_standard_patterns(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = root / "tests"
    tests_dir.mkdir()

    # Prefixes and suffixes
    assert is_test_file(root / "test_math.py", root)
    assert is_test_file(root / "math_test.py", root)
    assert is_test_file(root / "sub" / "test_nested.py", root)
    assert is_test_file(root / "sub" / "nested_test.py", root)

    # Files inside tests directory
    assert is_test_file(tests_dir / "helper.py", root)
    assert is_test_file(tests_dir / "unit" / "test_logic.py", root)
    assert is_test_file(tests_dir / "unit" / "raw_logic.py", root)

    # Regular source files
    assert not is_test_file(root / "math.py", root)
    assert not is_test_file(root / "app.py", root)
    assert not is_test_file(root / "src" / "pkg" / "module.py", root)


def test_test_in_name_does_not_ignore_contest_latest_manifest(tmp_path: Path) -> None:
    """Regression test for defect: '"test" in name' silently ignores valid source files.

    Files like contest.py, latest.py, manifest.py, protest.py, and attestation.py
    contain 'test' as a substring but are not test suites. They must NOT be ignored.
    """
    root = tmp_path
    config = GhostConfig()

    problem_files = [
        root / "contest.py",
        root / "latest.py",
        root / "manifest.py",
        root / "protest.py",
        root / "attestation.py",
        root / "pkg" / "contest.py",
        root / "src" / "latest.py",
    ]

    for file_path in problem_files:
        assert not is_test_file(file_path, root), f"{file_path} falsely classified as test"
        assert should_watch_path(file_path, root, config), f"{file_path} should be watched"

    # In contrast, actual test files must be detected
    assert is_test_file(root / "test_contest.py", root)
    assert is_test_file(root / "contest_test.py", root)
    assert not should_watch_path(root / "test_contest.py", root, config)


def test_is_temporary_file(tmp_path: Path) -> None:
    root = tmp_path
    # Editor backups, swap files, temporary files
    assert is_temporary_file(root / "app.py~")
    assert is_temporary_file(root / "app.py.swp")
    assert is_temporary_file(root / "app.py.swo")
    assert is_temporary_file(root / ".app.py.tmp")
    assert is_temporary_file(root / ".main.py.swp")
    assert is_temporary_file(root / "#app.py#")
    assert is_temporary_file(root / "app.py~")

    # Bytecode
    assert is_temporary_file(root / "app.pyc")
    assert is_temporary_file(root / "app.pyo")
    assert is_temporary_file(root / "__pycache__" / "app.cpython-313.pyc")

    # Regular files
    assert not is_temporary_file(root / "app.py")
    assert not is_temporary_file(root / "calc.py")


def test_should_watch_path_scanner_ignores(tmp_path: Path) -> None:
    root = tmp_path
    config = GhostConfig(
        scanner=ScannerConfig(
            ignore_dirs=[".venv", "venv", "node_modules", ".git", ".ghost", "tests"],
            ignore_files=["setup.py", "conftest.py", "__init__.py"],
        )
    )

    # Ignored directories
    assert not should_watch_path(root / ".venv" / "lib" / "module.py", root, config)
    assert not should_watch_path(root / "node_modules" / "pkg" / "script.py", root, config)
    assert not should_watch_path(root / ".git" / "hooks" / "pre-commit.py", root, config)

    # Ignored files
    assert not should_watch_path(root / "setup.py", root, config)
    assert not should_watch_path(root / "conftest.py", root, config)
    assert not should_watch_path(root / "src" / "__init__.py", root, config)

    # Non-python files
    assert not should_watch_path(root / "README.md", root, config)
    assert not should_watch_path(root / "data.json", root, config)

    # Valid Python source
    assert should_watch_path(root / "src" / "main.py", root, config)
    assert should_watch_path(root / "app.py", root, config)


def test_on_moved_handles_atomic_rename_saves(tmp_path: Path) -> None:
    """Regression test for defect: 'on_moved' unhandled — misses saves from atomic editors.

    Vim, JetBrains IDEs, and formatters (Black, Ruff) perform atomic saves by
    writing to a temporary file and renaming it over the destination file.
    The watcher must handle FileMovedEvent and inspect the destination path.
    """
    root = tmp_path
    config = GhostConfig()
    loop = MagicMock(spec=asyncio.AbstractEventLoop)

    modified_calls: list[Path] = []
    deleted_calls: list[Path] = []

    def on_modified(p: Path) -> None:
        modified_calls.append(p)

    def on_deleted(p: Path) -> None:
        deleted_calls.append(p)

    # Spy on call_soon_threadsafe to execute callback immediately
    loop.call_soon_threadsafe.side_effect = lambda cb, p: cb(p)

    handler = GhostEventHandler(
        project_root=root,
        config=config,
        loop=loop,
        on_modified_callback=on_modified,
        on_deleted_callback=on_deleted,
    )

    # 1. Simulate an atomic save of calc.py from a temporary file
    temp_swap = root / ".calc.py.4913.tmp"
    dest_file = root / "calc.py"
    move_event = FileMovedEvent(src_path=str(temp_swap), dest_path=str(dest_file))

    handler.on_moved(move_event)

    assert len(modified_calls) == 1
    assert modified_calls[0] == dest_file
    assert len(deleted_calls) == 0

    # 2. Moving a watched file away to a backup or trash triggers deletion handling
    backup_file = root / "calc.py.bak"
    move_away_event = FileMovedEvent(src_path=str(dest_file), dest_path=str(backup_file))

    handler.on_moved(move_away_event)
    assert len(deleted_calls) == 1
    assert deleted_calls[0] == dest_file


def test_event_handler_dispatches_events(tmp_path: Path) -> None:
    root = tmp_path
    config = GhostConfig()
    loop = MagicMock(spec=asyncio.AbstractEventLoop)

    modified_paths: list[Path] = []
    deleted_paths: list[Path] = []

    loop.call_soon_threadsafe.side_effect = lambda cb, p: cb(p)

    handler = GhostEventHandler(
        project_root=root,
        config=config,
        loop=loop,
        on_modified_callback=modified_paths.append,
        on_deleted_callback=deleted_paths.append,
    )

    # Modified event on watched file
    py_file = root / "service.py"
    handler.on_modified(FileModifiedEvent(src_path=str(py_file)))
    assert py_file in modified_paths

    # Created event on watched file
    new_file = root / "utils.py"
    handler.on_created(FileCreatedEvent(src_path=str(new_file)))
    assert new_file in modified_paths

    # Deleted event on watched file
    del_file = root / "old.py"
    handler.on_deleted(FileDeletedEvent(src_path=str(del_file)))
    assert del_file in deleted_paths


def test_event_handler_ignores_directories_and_temp(tmp_path: Path) -> None:
    root = tmp_path
    config = GhostConfig()
    loop = MagicMock(spec=asyncio.AbstractEventLoop)

    modified_paths: list[Path] = []
    loop.call_soon_threadsafe.side_effect = lambda cb, p: cb(p)

    handler = GhostEventHandler(
        project_root=root,
        config=config,
        loop=loop,
        on_modified_callback=modified_paths.append,
    )

    # Directory events
    handler.on_modified(DirModifiedEvent(src_path=str(root / "src")))
    handler.on_created(DirCreatedEvent(src_path=str(root / "src")))
    handler.on_moved(DirMovedEvent(src_path=str(root / "src"), dest_path=str(root / "pkg")))

    # Temporary files
    handler.on_modified(FileModifiedEvent(src_path=str(root / "temp.py~")))
    handler.on_created(FileCreatedEvent(src_path=str(root / ".app.swp")))

    assert len(modified_paths) == 0


@pytest.mark.asyncio
async def test_file_watcher_process_file_updates_index_and_pipeline(tmp_path: Path) -> None:
    root = tmp_path
    src_file = root / "calculator.py"
    src_file.write_text(
        "def multiply(a: int, b: int) -> int:\n    return a * b\n", encoding="utf-8"
    )

    config = GhostConfig()
    mock_pipeline = MagicMock(spec=TestPipeline)
    mock_pipeline.config = config
    mock_pipeline.project_root = root

    expected_result = PipelineResult(
        source_file=src_file,
        test_file=root / "tests" / "test_calculator.py",
        status=PipelineStatus.PASSED,
        passed=True,
        attempts=0,
        last_run=TestRunResult(
            test_file=root / "tests" / "test_calculator.py",
            passed=True,
            return_code=0,
            stdout="1 passed",
            stderr="",
        ),
    )
    mock_pipeline.run = AsyncMock(return_value=expected_result)

    results_received: list[PipelineResult] = []

    watcher = FileWatcher(
        project_root=root,
        config=config,
        pipeline=mock_pipeline,
        on_result=results_received.append,
        debounce_seconds=0.01,
    )

    await watcher._process_file(src_file)

    # Verify context.json was updated
    context_file = root / ".ghost" / "context.json"
    assert context_file.is_file()
    assert "calculator.py" in context_file.read_text(encoding="utf-8")

    # Verify pipeline was invoked with if_changed=True
    mock_pipeline.run.assert_awaited_once_with(
        src_file.resolve(),
        if_changed=True,
        auto_heal=config.tests.auto_heal,
        use_judge=config.tests.use_judge,
        listener=None,
    )
    assert len(results_received) == 1
    assert results_received[0] == expected_result


@pytest.mark.asyncio
async def test_file_watcher_skips_unparseable_syntax_errors(tmp_path: Path) -> None:
    root = tmp_path
    broken_file = root / "broken.py"
    broken_file.write_text("def incomplete_syntax(", encoding="utf-8")

    config = GhostConfig()
    mock_pipeline = MagicMock(spec=TestPipeline)
    mock_pipeline.config = config
    mock_pipeline.project_root = root
    mock_pipeline.run = AsyncMock()

    watcher = FileWatcher(
        project_root=root,
        config=config,
        pipeline=mock_pipeline,
    )

    await watcher._process_file(broken_file)

    # Pipeline should NOT have been called on unparseable code
    mock_pipeline.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_file_watcher_lifecycle_start_stop(tmp_path: Path) -> None:
    root = tmp_path
    config = GhostConfig(watcher=WatcherConfig(debounce_seconds=1))

    mock_pipeline = MagicMock(spec=TestPipeline)
    mock_pipeline.config = config
    mock_pipeline.project_root = root

    watcher = FileWatcher(
        project_root=root,
        config=config,
        pipeline=mock_pipeline,
    )

    assert not watcher.is_running()
    watcher.start()
    assert watcher.is_running()
    assert watcher.observer is not None

    await watcher.stop(timeout_seconds=1.0)
    assert not watcher.is_running()
    assert watcher.observer is None


@pytest.mark.asyncio
async def test_file_watcher_async_context_manager(tmp_path: Path) -> None:
    root = tmp_path
    config = GhostConfig()
    mock_pipeline = MagicMock(spec=TestPipeline)

    async with FileWatcher(project_root=root, config=config, pipeline=mock_pipeline) as watcher:
        assert watcher.is_running()
        assert watcher.observer is not None

    assert not watcher.is_running()
    assert watcher.observer is None


@pytest.mark.asyncio
async def test_file_watcher_on_file_deleted_cleans_up(tmp_path: Path) -> None:
    root = tmp_path
    src_file = root / "removed.py"

    config = GhostConfig()
    mock_pipeline = MagicMock(spec=TestPipeline)
    mock_pipeline.config = config
    mock_pipeline.project_root = root
    mock_pipeline.tracker = MagicMock()
    mock_pipeline.tracker.aremove = AsyncMock()

    # Pre-populate context.json
    context_file = root / ".ghost" / "context.json"
    context_file.parent.mkdir(parents=True, exist_ok=True)
    context_file.write_text('{"removed.py": "def test(): pass"}', encoding="utf-8")

    watcher = FileWatcher(
        project_root=root,
        config=config,
        pipeline=mock_pipeline,
    )

    watcher._on_file_deleted(src_file)
    # Give the async task a chance to complete
    await asyncio.sleep(0.05)

    mock_pipeline.tracker.aremove.assert_awaited_once_with(src_file)
    assert "removed.py" not in context_file.read_text(encoding="utf-8")
