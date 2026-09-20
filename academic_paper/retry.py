"""Retry utility for transient network failures."""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


def _delay_or_raise(attempt: int, attempts: int, base_delay: float, exc: BaseException) -> float:
    """Return the backoff delay for a failed attempt, or re-raise on the final one."""
    if attempt == attempts:
        logger.exception("Final failure after %d attempts: %s", attempts, exc)
        raise exc
    delay = base_delay * (2 ** (attempt - 1))
    logger.warning("Attempt %d/%d failed: %s — retrying in %.1fs", attempt, attempts, exc, delay)
    return delay


def with_retry(fn, *args, attempts: int = 3, base_delay: float = 1.0, exceptions: tuple = (Exception,), **kwargs):
    """Call fn(*args, **kwargs), retrying up to `attempts` times on `exceptions`.

    Delays are exponential: base_delay, base_delay*2, ... On final failure,
    logs with full traceback and re-raises.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:
            time.sleep(_delay_or_raise(attempt, attempts, base_delay, exc))


async def async_with_retry(
    fn, *args, attempts: int = 3, base_delay: float = 1.0, exceptions: tuple = (Exception,), **kwargs
):
    """Async version of with_retry — awaits fn(*args, **kwargs)."""
    for attempt in range(1, attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except exceptions as exc:
            await asyncio.sleep(_delay_or_raise(attempt, attempts, base_delay, exc))
