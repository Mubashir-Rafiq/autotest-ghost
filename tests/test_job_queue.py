"""Tests for asyncio debouncing, per-path concurrency guards, and worker lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ghost.job_queue import AsyncDebouncer, JobQueue


@pytest.mark.asyncio
async def test_debouncer_collapses_rapid_submissions(tmp_path: Path) -> None:
    path = tmp_path / "app.py"
    calls: list[Path] = []

    async def on_ready(p: Path) -> None:
        calls.append(p)

    debouncer = AsyncDebouncer(delay_seconds=0.05, on_ready=on_ready)

    # Submit 5 times in rapid succession
    for _ in range(5):
        debouncer.submit(path)
        await asyncio.sleep(0.01)

    assert len(calls) == 0  # Still in debounce window

    await asyncio.sleep(0.08)
    assert len(calls) == 1
    assert calls[0] == path


@pytest.mark.asyncio
async def test_debouncer_deadline_bumping(tmp_path: Path) -> None:
    path = tmp_path / "calc.py"
    fire_time = 0.0
    loop = asyncio.get_running_loop()
    start_time = loop.time()

    async def on_ready(p: Path) -> None:
        nonlocal fire_time
        fire_time = loop.time()

    debouncer = AsyncDebouncer(delay_seconds=0.06, on_ready=on_ready)
    debouncer.submit(path)

    # At t=0.03, submit again (bumping deadline to t=0.03 + 0.06 = 0.09)
    await asyncio.sleep(0.03)
    debouncer.submit(path)

    # At t=0.07, initial deadline would have fired, but it was bumped!
    await asyncio.sleep(0.04)
    assert fire_time == 0.0

    # At t=0.12, the bumped deadline should have fired
    await asyncio.sleep(0.05)
    assert fire_time > 0.0
    elapsed = fire_time - start_time
    assert elapsed >= 0.08


@pytest.mark.asyncio
async def test_debouncer_independent_paths(tmp_path: Path) -> None:
    p1 = tmp_path / "mod1.py"
    p2 = tmp_path / "mod2.py"
    fired: list[Path] = []

    async def on_ready(p: Path) -> None:
        fired.append(p)

    debouncer = AsyncDebouncer(delay_seconds=0.04, on_ready=on_ready)
    debouncer.submit(p1)
    debouncer.submit(p2)

    await asyncio.sleep(0.06)
    assert set(fired) == {p1, p2}


@pytest.mark.asyncio
async def test_debouncer_cancel(tmp_path: Path) -> None:
    p1 = tmp_path / "f1.py"
    p2 = tmp_path / "f2.py"
    fired: list[Path] = []

    async def on_ready(p: Path) -> None:
        fired.append(p)

    debouncer = AsyncDebouncer(delay_seconds=0.05, on_ready=on_ready)
    debouncer.submit(p1)
    debouncer.submit(p2)

    debouncer.cancel(p1)  # Cancel only p1
    await asyncio.sleep(0.07)

    assert fired == [p2]


@pytest.mark.asyncio
async def test_job_queue_per_path_mutual_exclusion_under_concurrency(tmp_path: Path) -> None:
    """Regression test: multiple workers NEVER process the same path concurrently."""
    path_a = tmp_path / "shared.py"
    concurrent_executions = 0
    max_concurrent_for_path = 0
    total_runs = 0

    async def handler(p: Path) -> None:
        nonlocal concurrent_executions, max_concurrent_for_path, total_runs
        concurrent_executions += 1
        max_concurrent_for_path = max(max_concurrent_for_path, concurrent_executions)
        total_runs += 1

        # Simulate work while in worker
        await asyncio.sleep(0.05)
        concurrent_executions -= 1

    # 4 workers running concurrently!
    queue = JobQueue(handler, debounce_seconds=0.01, max_workers=4)
    queue.start()

    # Submit path_a
    queue.submit(path_a)
    await asyncio.sleep(0.02)  # Let worker 1 pick it up

    # While worker 1 is processing path_a, submit path_a again twice
    queue.submit(path_a)
    await asyncio.sleep(0.01)
    queue.submit(path_a)

    await queue.drain()
    await queue.stop()

    # Critical invariant: max concurrent executions for the same path MUST BE 1
    assert max_concurrent_for_path == 1
    # Both runs (initial + pending rerun) were executed sequentially
    assert total_runs == 2


@pytest.mark.asyncio
async def test_job_queue_different_paths_run_concurrently(tmp_path: Path) -> None:
    p1 = tmp_path / "one.py"
    p2 = tmp_path / "two.py"
    active_now = 0
    max_active = 0

    async def handler(p: Path) -> None:
        nonlocal active_now, max_active
        active_now += 1
        max_active = max(max_active, active_now)
        await asyncio.sleep(0.05)
        active_now -= 1

    queue = JobQueue(handler, debounce_seconds=0.01, max_workers=2)
    queue.start()

    queue.submit(p1)
    queue.submit(p2)

    await queue.drain()
    await queue.stop()

    # Different paths CAN run in parallel on multiple workers
    assert max_active == 2


@pytest.mark.asyncio
async def test_job_queue_drain_completes_all_work(tmp_path: Path) -> None:
    p1 = tmp_path / "p1.py"
    p2 = tmp_path / "p2.py"
    completed: list[Path] = []

    async def handler(p: Path) -> None:
        await asyncio.sleep(0.02)
        completed.append(p)

    queue = JobQueue(handler, debounce_seconds=0.02, max_workers=2)
    queue.start()

    queue.submit(p1)
    queue.submit(p2)

    await queue.drain()
    assert set(completed) == {p1.resolve(), p2.resolve()}
    await queue.stop()


@pytest.mark.asyncio
async def test_job_queue_stop_without_drain_cancels(tmp_path: Path) -> None:
    """Regression test: stop(drain=False) immediately cancels in-flight work and debouncers."""
    p1 = tmp_path / "slow.py"
    started = False
    finished = False

    async def handler(p: Path) -> None:
        nonlocal started, finished
        started = True
        await asyncio.sleep(1.0)
        finished = True

    queue = JobQueue(handler, debounce_seconds=0.01, max_workers=1)
    queue.start()
    queue.submit(p1)

    await asyncio.sleep(0.03)  # Let worker start
    assert started is True
    assert finished is False

    await queue.stop(drain=False, timeout_seconds=0.5)

    assert finished is False
    assert queue.is_running is False
