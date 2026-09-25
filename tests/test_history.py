"""Tests for HistoryTracker, UsageTracker, and rollback behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from ghost.history import (
    HistoryTracker,
    UsageTracker,
    get_history_tracker,
    get_usage_tracker,
)


def test_usage_tracker_initial_and_updates(tmp_path: Path) -> None:
    tracker = get_usage_tracker(tmp_path)
    initial = tracker.load_usage()
    assert initial.total_requests == 0
    assert initial.prompt_tokens == 0
    assert initial.completion_tokens == 0
    assert initial.total_tokens == 0
    assert initial.heal_attempts == 0
    assert initial.healed_tests == 0

    # Record first call
    rec1 = tracker.record_call(
        prompt_tokens=100,
        completion_tokens=50,
        is_heal=False,
    )
    assert rec1.total_requests == 1
    assert rec1.prompt_tokens == 100
    assert rec1.completion_tokens == 50
    assert rec1.total_tokens == 150
    assert rec1.heal_attempts == 0

    # Record second call as heal
    rec2 = tracker.record_call(
        prompt_tokens=200,
        completion_tokens=80,
        is_heal=True,
    )
    assert rec2.total_requests == 2
    assert rec2.prompt_tokens == 300
    assert rec2.completion_tokens == 130
    assert rec2.total_tokens == 430
    assert rec2.heal_attempts == 1
    assert rec2.healed_tests == 0

    # Record heal success
    rec3 = tracker.record_success()
    assert rec3.total_requests == 2
    assert rec3.healed_tests == 1

    # Reload from disk
    reloaded = tracker.load_usage()
    assert reloaded == rec3


@pytest.mark.asyncio
async def test_usage_tracker_async(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path)
    rec = await tracker.arecord_call(prompt_tokens=50, completion_tokens=25, is_heal=True)
    assert rec.total_requests == 1
    assert rec.total_tokens == 75

    rec_succ = await tracker.arecord_success()
    assert rec_succ.healed_tests == 1


def test_history_tracker_snapshots_and_rollback(tmp_path: Path) -> None:
    tracker = get_history_tracker(tmp_path)
    source_file = tmp_path / "src" / "math_mod.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("def add(a, b): return a + b\n")

    test_file = tmp_path / "tests" / "test_math_mod.py"
    test_file.parent.mkdir(parents=True)

    # Initially empty history
    assert tracker.get_history(source_file) == []
    assert tracker.list_tracked_files() == []

    # Record attempt 0
    code_v0 = "def test_add(): assert add(1, 2) == 4\n"
    snap0 = tracker.record_attempt(
        source_path=source_file,
        attempt=0,
        test_code=code_v0,
        status="generated",
    )
    assert snap0.is_file()
    assert snap0.read_text() == code_v0

    # Record attempt 1 (healing)
    code_v1 = "def test_add(): assert add(1, 2) == 3\n"
    snap1 = tracker.record_attempt(
        source_path=source_file,
        attempt=1,
        test_code=code_v1,
        status="healed",
        classification="LOGIC",
    )
    assert snap1.is_file()
    assert snap1.read_text() == code_v1

    # Verify history inspection
    history = tracker.get_history(source_file)
    assert len(history) == 2
    assert history[0].attempt == 0
    assert history[0].status == "generated"
    assert history[1].attempt == 1
    assert history[1].status == "healed"
    assert history[1].classification == "LOGIC"

    # List tracked files
    tracked = tracker.list_tracked_files()
    assert len(tracked) == 1
    assert "math_mod.py" in tracked[0]

    # Test file currently has code_v1
    test_file.write_text(code_v1)

    # Roll back to attempt 0
    tracker.rollback(source_file, test_file, attempt=0)
    assert test_file.read_text() == code_v0

    # Roll back without specifying attempt restores earliest (attempt 0)
    tracker.rollback(source_file, test_file)
    assert test_file.read_text() == code_v0

    # Roll back to attempt 1
    tracker.rollback(source_file, test_file, attempt=1)
    assert test_file.read_text() == code_v1


@pytest.mark.asyncio
async def test_history_tracker_async(tmp_path: Path) -> None:
    tracker = HistoryTracker(tmp_path)
    source_file = tmp_path / "foo.py"
    source_file.write_text("x = 1\n")
    test_file = tmp_path / "test_foo.py"

    snap = await tracker.arecord_attempt(
        source_path=source_file,
        attempt=0,
        test_code="def test_foo(): pass\n",
        status="generated",
    )
    assert snap.is_file()

    await tracker.arollback(source_file, test_file, attempt=0)
    assert test_file.read_text() == "def test_foo(): pass\n"


def test_history_tracker_error_handling(tmp_path: Path) -> None:
    tracker = HistoryTracker(tmp_path)
    source_file = tmp_path / "missing.py"
    test_file = tmp_path / "test_missing.py"

    with pytest.raises(FileNotFoundError, match="No history records found"):
        tracker.rollback(source_file, test_file)

    # Record attempt 0 only
    tracker.record_attempt(
        source_path=source_file,
        attempt=0,
        test_code="assert True\n",
        status="generated",
    )

    with pytest.raises(FileNotFoundError, match="No attempt #5 found"):
        tracker.rollback(source_file, test_file, attempt=5)
