"""Tests for the injected-or-temporary AsyncClient helper."""

from unittest.mock import patch

import httpx
import pytest

from academic_paper.http_client import client_or_temporary


@pytest.mark.anyio
async def test_injected_client_is_yielded_as_is_and_not_closed():
    persistent = httpx.AsyncClient()
    try:
        with patch("academic_paper.http_client.httpx.AsyncClient") as ctor:
            async with client_or_temporary(persistent, timeout=1.0) as client:
                assert client is persistent
        ctor.assert_not_called()
        assert not persistent.is_closed
    finally:
        await persistent.aclose()


@pytest.mark.anyio
async def test_temporary_client_uses_timeout_and_is_closed_on_exit():
    async with client_or_temporary(None, timeout=7.5) as client:
        assert isinstance(client, httpx.AsyncClient)
        assert client.timeout == httpx.Timeout(7.5)
        assert not client.is_closed
    assert client.is_closed
