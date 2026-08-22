"""Tests for retry utility."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from academic_paper.retry import async_with_retry, with_retry


class _TransientError(Exception):
    pass


# ---------------------------------------------------------------------------
# with_retry (sync)
# ---------------------------------------------------------------------------


def test_with_retry_succeeds_first_try():
    fn = MagicMock(return_value=42)
    result = with_retry(fn, attempts=3, exceptions=(_TransientError,))
    assert result == 42
    fn.assert_called_once()


def test_with_retry_two_failures_then_success():
    fn = MagicMock(side_effect=[_TransientError("1st"), _TransientError("2nd"), "ok"])
    result = with_retry(fn, attempts=3, base_delay=0, exceptions=(_TransientError,))
    assert result == "ok"
    assert fn.call_count == 3


def test_with_retry_all_failures_raises():
    fn = MagicMock(side_effect=_TransientError("boom"))
    with pytest.raises(_TransientError):
        with_retry(fn, attempts=3, base_delay=0, exceptions=(_TransientError,))
    assert fn.call_count == 3


def test_with_retry_non_matching_exception_not_retried():
    fn = MagicMock(side_effect=ValueError("not retried"))
    with pytest.raises(ValueError):
        with_retry(fn, attempts=3, base_delay=0, exceptions=(_TransientError,))
    fn.assert_called_once()


# ---------------------------------------------------------------------------
# async_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_with_retry_succeeds_first_try():
    fn = AsyncMock(return_value=99)
    result = await async_with_retry(fn, attempts=3, exceptions=(_TransientError,))
    assert result == 99
    fn.assert_awaited_once()


@pytest.mark.anyio
async def test_async_with_retry_two_failures_then_success():
    fn = AsyncMock(side_effect=[_TransientError("1st"), _TransientError("2nd"), "ok"])
    result = await async_with_retry(fn, attempts=3, base_delay=0, exceptions=(_TransientError,))
    assert result == "ok"
    assert fn.await_count == 3


@pytest.mark.anyio
async def test_async_with_retry_all_failures_raises():
    fn = AsyncMock(side_effect=_TransientError("boom"))
    with pytest.raises(_TransientError):
        await async_with_retry(fn, attempts=3, base_delay=0, exceptions=(_TransientError,))
    assert fn.await_count == 3


@pytest.mark.anyio
async def test_async_with_retry_non_matching_exception_not_retried():
    fn = AsyncMock(side_effect=ValueError("not retried"))
    with pytest.raises(ValueError):
        await async_with_retry(fn, attempts=3, base_delay=0, exceptions=(_TransientError,))
    fn.assert_awaited_once()
