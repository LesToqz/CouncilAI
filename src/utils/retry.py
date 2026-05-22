from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


async def retry_async(
    func: Callable[[], Awaitable[T]],
    attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff: float = 2.0,
) -> T:
    last_error: Exception | None = None
    sleep_for = delay_seconds
    for _ in range(attempts):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001 - callers need the final original failure.
            last_error = exc
            await asyncio.sleep(sleep_for)
            sleep_for *= backoff
    if last_error is None:
        raise RuntimeError("retry_async called with no attempts")
    raise last_error
