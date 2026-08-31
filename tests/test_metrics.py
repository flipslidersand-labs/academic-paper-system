"""Tests for the Prometheus /metrics endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.server import app


def test_metrics_endpoint_returns_200(temp_db):
    """GET /metrics returns 200."""
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
    mock_qdrant = MagicMock()

    with (
        patch.object(settings, "academic_db", temp_db),
        patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
        patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
    ):
        client = TestClient(app)
        response = client.get("/metrics")

    assert response.status_code == 200


def test_metrics_endpoint_content_type(temp_db):
    """GET /metrics returns text/plain Prometheus format."""
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
    mock_qdrant = MagicMock()

    with (
        patch.object(settings, "academic_db", temp_db),
        patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
        patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
    ):
        client = TestClient(app)
        response = client.get("/metrics")

    assert "text/plain" in response.headers["content-type"]


def test_metrics_contains_http_requests_total(temp_db):
    """GET /metrics body includes http_requests_total after a request."""
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
    mock_qdrant = MagicMock()

    with (
        patch.object(settings, "academic_db", temp_db),
        patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
        patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
    ):
        client = TestClient(app)
        client.get("/health")
        response = client.get("/metrics")

    assert "http_requests_total" in response.text
