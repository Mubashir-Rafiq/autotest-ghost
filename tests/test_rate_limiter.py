"""Tests for Ghost's virtual-scheduling rate limiter and retry backoff."""

from __future__ import annotations

import asyncio

import pytest

from ghost.rate_limiter import (
    RateLimiter,
    calculate_backoff_delay,
    is_rate_limit_error,
)


def test_rate_limiter_interval_derived_from_rpm() -> None:
    """Interval is derived correctly from requests-per-minute."""
    limiter_30 = RateLimiter(rate_limit_rpm=30)
    assert limiter_30.interval == 2.0

    limiter_60 = RateLimiter(rate_limit_rpm=60)
    assert limiter_60.interval == 1.0

    # Ensure zero or negative is bounded at 1
    limiter_0 = RateLimiter(rate_limit_rpm=0)
    assert limiter_0.interval == 60.0


@pytest.mark.asyncio
async def test_virtual_scheduling_concurrency() -> None:
    """Concurrent tasks acquire strictly separated slots under virtual scheduling."""
    # Use 1200 RPM -> 0.05s interval for fast testing
    limiter = RateLimiter(rate_limit_rpm=1200)
    interval = limiter.interval
    timestamps: list[float] = []

    async def worker() -> None:
        await limiter.acquire()
        loop = asyncio.get_running_loop()
        timestamps.append(loop.time())

    # Launch 4 concurrent workers
    tasks = [asyncio.create_task(worker()) for _ in range(4)]
    await asyncio.gather(*tasks)

    assert len(timestamps) == 4
    # Timestamps should be ordered and separated by at least interval
    # (allowing a small timing tolerance).
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        assert delta >= interval * 0.8, f"Interval {delta} was smaller than expected {interval}"


@pytest.mark.asyncio
async def test_rate_limiter_context_manager() -> None:
    """RateLimiter functions as an async context manager."""
    limiter = RateLimiter(rate_limit_rpm=6000)  # 0.01s interval
    async with limiter:
        pass


def test_calculate_backoff_delay_exponential_with_jitter() -> None:
    """Backoff delay scales exponentially and adds random jitter."""
    delay_0 = calculate_backoff_delay(attempt=0, base_delay=2.0)
    # base * 2^0 = 2.0, jitter is in [0.1, 1.0] -> [2.1, 3.0]
    assert 2.1 <= delay_0 <= 3.0

    delay_1 = calculate_backoff_delay(attempt=1, base_delay=2.0)
    # base * 2^1 = 4.0, jitter in [0.1, 1.0] -> [4.1, 5.0]
    assert 4.1 <= delay_1 <= 5.0

    # Capped at max_delay
    delay_capped = calculate_backoff_delay(attempt=10, base_delay=2.0, max_delay=10.0)
    assert 10.1 <= delay_capped <= 11.0


def test_is_rate_limit_error_classification() -> None:
    """Classifies rate limit indicators from status codes and error messages."""

    # 1. Status code attribute
    class StatusTestError(Exception):
        def __init__(self, code: int) -> None:
            self.status_code = code

    assert is_rate_limit_error(StatusTestError(429))
    assert not is_rate_limit_error(StatusTestError(500))

    # 2. Response status code attribute
    class ResponseTestError(Exception):
        def __init__(self, code: int) -> None:
            class DummyResponse:
                status_code = code

            self.response = DummyResponse()

    assert is_rate_limit_error(ResponseTestError(429))
    assert not is_rate_limit_error(ResponseTestError(404))

    # 3. Message substrings
    assert is_rate_limit_error(Exception("Rate limit reached for model"))
    assert is_rate_limit_error(Exception("error 429: Too Many Requests"))
    assert is_rate_limit_error(Exception("quota exceeded for current billing cycle"))
    assert is_rate_limit_error(Exception("TPM limit exceeded"))

    # 4. Non-rate-limit errors
    assert not is_rate_limit_error(ValueError("invalid arguments"))
    assert not is_rate_limit_error(RuntimeError("syntax error in prompt"))
