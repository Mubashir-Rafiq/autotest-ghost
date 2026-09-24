"""Ghost's rate limiting and retry coordination.

Owns:
- Asynchronous virtual-scheduling rate limiter (GCRA) that guarantees deterministic
  request spacing under concurrency without blocking the event loop.
- Exponential backoff delay calculation with random jitter to prevent lockstep retries.
- Classification of exceptions as rate-limit vs. non-rate-limit errors.

Does NOT:
- Know about specific LLM providers (that belongs in ``providers.py``).
- Make network calls or format terminal presentations.

Guarantees:
1. Virtual scheduling: competing concurrent coroutines receive strictly spaced
   time slots atomically and sleep outside the critical section.
2. Non-rate-limit exceptions fail immediately without retrying.
"""

from __future__ import annotations

import asyncio
import random
from typing import Final

__all__ = [
    "RateLimiter",
    "calculate_backoff_delay",
    "is_rate_limit_error",
]

_HTTP_TOO_MANY_REQUESTS: Final[int] = 429

_RATE_LIMIT_INDICATORS: Final[tuple[str, ...]] = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota exceeded",
    "resource has been exhausted",
    "tpm",
    "rpm",
)


class RateLimiter:
    """Async virtual-scheduling rate limiter.

    Derives a minimum request interval from requests-per-minute (RPM).
    Under concurrent invocations, each caller receives an atomically
    advanced scheduled timestamp, sleeping outside the lock so other
    tasks can schedule concurrently without head-of-line blocking.
    """

    def __init__(self, rate_limit_rpm: int = 30) -> None:
        self.rate_limit_rpm = max(rate_limit_rpm, 1)
        self.interval: float = 60.0 / self.rate_limit_rpm
        self._lock = asyncio.Lock()
        self._next_schedule: float = 0.0

    async def acquire(self) -> float:
        """Wait until the next virtual schedule slot.

        Returns the delay (in seconds) waited.
        """
        loop = asyncio.get_running_loop()
        now = loop.time()

        async with self._lock:
            scheduled_time = max(now, self._next_schedule)
            self._next_schedule = scheduled_time + self.interval

        delay = scheduled_time - now
        if delay > 0:
            await asyncio.sleep(delay)
            return delay
        return 0.0

    async def __aenter__(self) -> RateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> float:
    """Calculate exponential backoff delay with random jitter.

    Jitter avoids synchronized thundering herds when multiple calls
    experience rate-limit backpressure simultaneously.
    """
    exponential = base_delay * (2.0**attempt)
    capped = min(exponential, max_delay)
    jitter = random.uniform(0.1, 1.0)  # noqa: S311
    return capped + jitter


def is_rate_limit_error(exc: Exception) -> bool:
    """Determine whether an exception indicates an API rate limit or quota exceeded."""
    status_code = getattr(exc, "status_code", None)
    if status_code == _HTTP_TOO_MANY_REQUESTS:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == _HTTP_TOO_MANY_REQUESTS:
        return True

    text = str(exc).lower()
    return any(indicator in text for indicator in _RATE_LIMIT_INDICATORS)
