"""Tests for the streaming upload size cap in POST /papers/ingest (#490).

TestClient fills UploadFile.size, so the early file.size rejection would always
fire first. These tests clear size to force the byte-counting path.
"""

import os
import tempfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from academic_paper.config import settings
from academic_paper.server import app


@pytest.fixture
def client(temp_db):
    with (
        patch.object(settings, "academic_db", temp_db),
        patch.object(settings, "api_key", ""),
    ):
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
        mock_qdrant = MagicMock()
        mock_qdrant.aupsert = AsyncMock(return_value=None)
        mock_qdrant.aensure_collection = AsyncMock(return_value=None)
        mock_qdrant.adelete_by_paper_id = AsyncMock(return_value=None)
        with (
            patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
            patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
        ):
            c = TestClient(app)
            c.app.state.embedder = mock_embedder
            c.app.state.vector_store = mock_qdrant
            yield c


@pytest.fixture
def size_unknown():
    """Make every UploadFile report size=None so only the streaming check applies."""
    orig_init = UploadFile.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.size = None

    with patch.object(UploadFile, "__init__", init):
        yield


def _pdf(total: int) -> bytes:
    head = b"%PDF-1.4\n"
    return head + b"0" * (total - len(head))


def _post(client, body: bytes):
    return client.post(
        "/papers/ingest?wait=true",
        files={"file": ("big.pdf", BytesIO(body), "application/pdf")},
    )


def test_exactly_at_limit_is_accepted(client, size_unknown):
    with (
        patch.object(settings, "max_upload_mb", 1),
        patch("academic_paper.services.ingest_service.extract_text", return_value=[{"page": 1, "text": "content"}]),
    ):
        resp = _post(client, _pdf(1024 * 1024))
    assert resp.status_code == 200, resp.text


def test_one_byte_over_limit_returns_413_and_removes_tmpfile(client, size_unknown):
    captured = {}
    orig_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(**kwargs):
        ctx = orig_ntf(**kwargs)
        captured["path"] = ctx.name
        return ctx

    with (
        patch.object(settings, "max_upload_mb", 1),
        patch("academic_paper.services.ingest_service.tempfile.NamedTemporaryFile", side_effect=fake_ntf),
    ):
        resp = _post(client, _pdf(1024 * 1024 + 1))
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()
    assert "path" in captured
    assert not os.path.exists(captured["path"])
