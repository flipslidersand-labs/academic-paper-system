"""Retry utility for transient network failures."""

import asyncio
import contextvars
import logging
import threading
import time

logger = logging.getLogger(__name__)


# Cooperative cancellation for sync work run via asyncio.to_thread (#313).
# asyncio.to_thread cannot stop a running thread, so cancelling the awaiting task
# (e.g. wait_for timeout) leaves the thread running. to_thread_cancellable() sets
# this event on cancellation; the context is copied into the worker thread, so
# with_retry()/raise_if_cancelled() can see it and stop *between* attempts/batches.
_cancel_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("_cancel_event", default=None)


class OperationCancelledError(Exception):
    """Raised inside a worker thread when its awaiting task was cancelled (#313)."""


def raise_if_cancelled() -> None:
    """Call between steps of a long sync operation to stop after the caller cancelled."""
    ev = _cancel_event.get()
    if ev is not None and ev.is_set():
        raise OperationCancelledError("caller cancelled; abandoning remaining work")


async def to_thread_cancellable(fn, *args, **kwargs):
    """asyncio.to_thread plus a cancellation signal visible to with_retry/raise_if_cancelled.

    Limitation: a call already in flight (HTTP request, SQLite statement) is NOT
    interrupted; it runs until its own client timeout. Only further retries and
    remaining batches are skipped.
    """
    ev = threading.Event()
    token = _cancel_event.set(ev)
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except asyncio.CancelledError:
        ev.set()
        raise
    finally:
        _cancel_event.reset(token)


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
            delay = _delay_or_raise(attempt, attempts, base_delay, exc)
            ev = _cancel_event.get()
            if ev is None:
                time.sleep(delay)
            elif ev.wait(delay):
                # Caller cancelled during backoff: stop retrying (#313).
                raise exc


async def async_with_retry(
    fn, *args, attempts: int = 3, base_delay: float = 1.0, exceptions: tuple = (Exception,), **kwargs
):
    """Async version of with_retry — awaits fn(*args, **kwargs)."""
    for attempt in range(1, attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except exceptions as exc:
            await asyncio.sleep(_delay_or_raise(attempt, attempts, base_delay, exc))
