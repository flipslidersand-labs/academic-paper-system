"""Tests for the Prometheus /metrics endpoint."""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.db import init_db
from academic_paper.server import app


def test_metrics_endpoint_returns_200():
    """GET /metrics returns 200."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    init_db(db_path)

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
    mock_qdrant = MagicMock()

    with (
        patch.object(settings, "academic_db", db_path),
        patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
        patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
    ):
        client = TestClient(app)
        response = client.get("/metrics")

    assert response.status_code == 200


def test_metrics_endpoint_content_type():
    """GET /metrics returns text/plain Prometheus format."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    init_db(db_path)

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
    mock_qdrant = MagicMock()

    with (
        patch.object(settings, "academic_db", db_path),
        patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
        patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
    ):
        client = TestClient(app)
        response = client.get("/metrics")

    assert "text/plain" in response.headers["content-type"]


def test_metrics_contains_http_requests_total():
    """GET /metrics body includes http_requests_total after a request."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    init_db(db_path)

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
    mock_qdrant = MagicMock()

    with (
        patch.object(settings, "academic_db", db_path),
        patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
        patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
    ):
        client = TestClient(app)
        client.get("/health")
        response = client.get("/metrics")

    assert "http_requests_total" in response.text
