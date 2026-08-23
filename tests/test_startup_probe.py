"""Tests for _probe_startup_health startup health check."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from academic_paper.server import _probe_startup_health


def _make_app(*, qdrant_ok: bool, embed_ok: bool):
    """Build a minimal fake app with mocked vector_store for probe testing."""
    mock_qdrant = MagicMock()
    if not qdrant_ok:
        mock_qdrant.client.get_collections.side_effect = Exception("qdrant down")

    app = MagicMock()
    app.state.vector_store = mock_qdrant
    return app


@pytest.mark.anyio
async def test_probe_logs_info_when_both_ok(caplog):
    """_probe_startup_health logs INFO for both services when reachable."""
    app = _make_app(qdrant_ok=True, embed_ok=True)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("academic_paper.server.httpx.AsyncClient", return_value=mock_async_cm):
        with caplog.at_level("INFO", logger="academic_paper.server"):
            await _probe_startup_health(app)

    assert "Startup probe OK: Qdrant" in caplog.text
    assert "Startup probe OK: embedding-svc" in caplog.text


@pytest.mark.anyio
async def test_probe_warns_when_qdrant_down(caplog):
    """_probe_startup_health logs WARNING when Qdrant is unreachable."""
    app = _make_app(qdrant_ok=False, embed_ok=True)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("academic_paper.server.httpx.AsyncClient", return_value=mock_async_cm):
        with caplog.at_level("WARNING", logger="academic_paper.server"):
            await _probe_startup_health(app)

    assert "Qdrant unreachable" in caplog.text


@pytest.mark.anyio
async def test_probe_warns_when_embed_svc_down(caplog):
    """_probe_startup_health logs WARNING when embedding-svc is unreachable."""
    app = _make_app(qdrant_ok=True, embed_ok=False)

    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("connection refused")
    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("academic_paper.server.httpx.AsyncClient", return_value=mock_async_cm):
        with caplog.at_level("WARNING", logger="academic_paper.server"):
            await _probe_startup_health(app)

    assert "embedding-svc unreachable" in caplog.text


@pytest.mark.anyio
async def test_probe_warns_when_both_down(caplog):
    """_probe_startup_health logs WARNING for both when both are unreachable."""
    app = _make_app(qdrant_ok=False, embed_ok=False)

    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("connection refused")
    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("academic_paper.server.httpx.AsyncClient", return_value=mock_async_cm):
        with caplog.at_level("WARNING", logger="academic_paper.server"):
            await _probe_startup_health(app)

    assert "Qdrant unreachable" in caplog.text
    assert "embedding-svc unreachable" in caplog.text
