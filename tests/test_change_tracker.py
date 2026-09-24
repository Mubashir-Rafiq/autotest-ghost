"""Tests for SHA-256 change tracking, hash persistence, and --if-changed caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from ghost.change_tracker import ChangeTracker, compute_hash
from ghost.config import GhostConfig
from ghost.pipeline import PipelineStatus, TestPipeline
from ghost.runner import TestRunResult


def test_compute_hash() -> None:
    h1 = compute_hash("def add(a, b): return a + b\n")
    h2 = compute_hash("def add(a, b): return a + b\n")
    h3 = compute_hash("def add(a, b): return a + b # diff\n")

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


def test_normalize_key_relative_path(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    file_path = tmp_path / "src" / "pkg" / "module.py"
    key = tracker.normalize_key(file_path)
    assert key == "src/pkg/module.py"


def test_normalize_key_avoids_basename_collision(tmp_path: Path) -> None:
    """Regression test: identically named files in different subdirectories have distinct keys."""
    tracker = ChangeTracker(tmp_path)
    auth_view = tmp_path / "auth" / "views.py"
    billing_view = tmp_path / "billing" / "views.py"

    auth_view.parent.mkdir(parents=True)
    billing_view.parent.mkdir(parents=True)
    auth_view.write_text("def auth(): pass\n", encoding="utf-8")
    billing_view.write_text("def billing(): pass\n", encoding="utf-8")

    key_auth = tracker.normalize_key(auth_view)
    key_billing = tracker.normalize_key(billing_view)

    assert key_auth == "auth/views.py"
    assert key_billing == "billing/views.py"
    assert key_auth != key_billing

    tracker.mark_processed(auth_view, "def auth(): pass\n")
    tracker.mark_processed(billing_view, "def billing(): pass\n")

    hashes = tracker.load_hashes()
    assert "auth/views.py" in hashes
    assert "billing/views.py" in hashes
    assert hashes["auth/views.py"] != hashes["billing/views.py"]


def test_fail_open_on_missing_hashes_file(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    assert not tracker.hashes_file.exists()
    assert tracker.load_hashes() == {}
    assert tracker.has_changed(tmp_path / "app.py", "content") is True


def test_fail_open_on_corrupted_hashes_file(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    tracker.hashes_file.parent.mkdir(parents=True, exist_ok=True)
    tracker.hashes_file.write_text("{broken json syntax: [invalid", encoding="utf-8")

    # Must fail open (return {}) and treat file as changed without crashing
    assert tracker.load_hashes() == {}
    assert tracker.has_changed(tmp_path / "app.py", "content") is True


def test_mark_processed_persists_atomically(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    app_file = tmp_path / "app.py"
    content_v1 = "x = 1\n"
    content_v2 = "x = 2\n"

    assert tracker.has_changed(app_file, content_v1) is True

    tracker.mark_processed(app_file, content_v1)
    assert tracker.hashes_file.is_file()

    # Content v1 is now recorded as unchanged
    assert tracker.has_changed(app_file, content_v1) is False
    # Content v2 is changed
    assert tracker.has_changed(app_file, content_v2) is True

    # Persist content v2
    tracker.mark_processed(app_file, content_v2)
    assert tracker.has_changed(app_file, content_v1) is True
    assert tracker.has_changed(app_file, content_v2) is False


def test_remove_deletes_entry(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    file_path = tmp_path / "temp.py"
    tracker.mark_processed(file_path, "foo")
    assert tracker.has_changed(file_path, "foo") is False

    tracker.remove(file_path)
    assert tracker.has_changed(file_path, "foo") is True


@pytest.mark.asyncio
async def test_async_methods(tmp_path: Path) -> None:
    tracker = ChangeTracker(tmp_path)
    file_path = tmp_path / "async.py"
    content = "async def f(): pass\n"

    assert await tracker.ahas_changed(file_path, content) is True
    await tracker.amark_processed(file_path, content)
    assert await tracker.ahas_changed(file_path, content) is False
    await tracker.aremove(file_path)
    assert await tracker.ahas_changed(file_path, content) is True


@pytest.mark.asyncio
async def test_pipeline_integration_if_changed(tmp_path: Path) -> None:
    """Test that pipeline skips execution when if_changed=True on unchanged file."""
    src = tmp_path / "maths.py"
    src.write_text("def add(a, b): return a + b\n", encoding="utf-8")

    test_file = tmp_path / "tests" / "test_maths.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("# Generated at: 24-09-2026\ndef test_add(): assert True\n")

    runs = 0

    async def counting_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        nonlocal runs
        runs += 1
        return TestRunResult(test_file=test_file, passed=True, return_code=0, stdout="", stderr="")

    config = GhostConfig()
    pipeline = TestPipeline(config, tmp_path, runner_fn=counting_runner)

    # First run: file has not been tracked yet -> runs and records hash
    res1 = await pipeline.run(src, if_changed=True)
    assert res1.status == PipelineStatus.PASSED
    assert res1.passed is True
    assert runs == 1

    # Second run without content change: should be SKIPPED!
    res2 = await pipeline.run(src, if_changed=True)
    assert res2.status == PipelineStatus.SKIPPED
    assert res2.passed is True
    assert runs == 1  # runner was NOT called again

    # Third run without if_changed: forces execution
    res3 = await pipeline.run(src, if_changed=False)
    assert res3.status == PipelineStatus.PASSED
    assert runs == 2  # runner called

    # Fourth run after content modification: runs again with if_changed=True
    src.write_text("def add(a, b): return a + b + 0\n", encoding="utf-8")
    res4 = await pipeline.run(src, if_changed=True)
    assert res4.status == PipelineStatus.PASSED
    assert runs == 3


@pytest.mark.asyncio
async def test_pipeline_invariant_hash_never_written_on_crash(tmp_path: Path) -> None:
    """Critical invariant: A crash mid-pipeline never records a hash as 'processed'."""
    src = tmp_path / "crash.py"
    content = "raise RuntimeError('boom')\n"
    src.write_text(content, encoding="utf-8")

    test_file = tmp_path / "tests" / "test_crash.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("# Generated at: 24-09-2026\ndef test_crash(): assert True\n")

    async def crashing_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        err_msg = "Subprocess runner crashed unhandled"
        raise RuntimeError(err_msg)

    tracker = ChangeTracker(tmp_path)
    config = GhostConfig()
    pipeline = TestPipeline(config, tmp_path, runner_fn=crashing_runner, tracker=tracker)

    with pytest.raises(RuntimeError, match="Subprocess runner crashed unhandled"):
        await pipeline.run(src, if_changed=True)

    # Invariant: hash was NOT written because pipeline crashed before completing
    assert tracker.has_changed(src, content) is True
    hashes = tracker.load_hashes()
    assert tracker.normalize_key(src) not in hashes
