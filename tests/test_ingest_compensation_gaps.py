"""Gap-filling tests for Qdrant compensation on ingest failure (#676).

Complements test_ingest_service.py / test_ingest_compensation.py: exact delete
counts, original-exception precedence + error log when compensation fails, and
wait=true failing before the DB write (embed error).
"""

import asyncio
import logging
import sqlite3
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import anyio.from_thread
import httpx
import pytest
from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.jobs import job_store
from academic_paper.server import app
from academic_paper.services import ingest_service as svc

PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
_PAGES = [{"page": 1, "text": "hello world"}]


def _deps():
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2]])
    store = MagicMock()
    store.aensure_collection = AsyncMock()
    store.aupsert = AsyncMock()
    store.adelete_by_paper_id = AsyncMock()
    return embedder, store


def _run_pipeline(store, embedder):
    async def go():
        return await svc.ingest_pipeline("/x.pdf", 3, "h", "f.pdf", embedder=embedder, vector_store=store)

    return asyncio.run(go())


def test_pipeline_save_chunks_failure_deletes_once_and_reraises_original():
    embedder, store = _deps()
    err = sqlite3.OperationalError("locked")
    with (
        patch.object(svc, "extract_text", return_value=_PAGES),
        patch.object(svc, "chunk_pages", return_value=[{"text": "hello"}]),
        patch.object(svc, "db_connection"),
        patch.object(svc, "save_chunks", side_effect=err),
        pytest.raises(sqlite3.OperationalError) as ei,
    ):
        _run_pipeline(store, embedder)
    assert ei.value is err
    store.adelete_by_paper_id.assert_awaited_once_with(3)


def test_pipeline_update_status_failure_also_compensates():
    embedder, store = _deps()
    with (
        patch.object(svc, "extract_text", return_value=_PAGES),
        patch.object(svc, "chunk_pages", return_value=[{"text": "hello"}]),
        patch.object(svc, "db_connection"),
        patch.object(svc, "save_chunks"),
        patch.object(svc, "update_paper_status", side_effect=sqlite3.OperationalError("locked")),
        pytest.raises(sqlite3.OperationalError),
    ):
        _run_pipeline(store, embedder)
    store.adelete_by_paper_id.assert_awaited_once_with(3)


def test_pipeline_compensation_failure_keeps_original_exception_and_logs(caplog):
    embedder, store = _deps()
    store.adelete_by_paper_id.side_effect = RuntimeError("qdrant down")
    with (
        patch.object(svc, "extract_text", return_value=_PAGES),
        patch.object(svc, "chunk_pages", return_value=[{"text": "hello"}]),
        patch.object(svc, "db_connection"),
        patch.object(svc, "save_chunks", side_effect=sqlite3.OperationalError("locked")),
        caplog.at_level(logging.ERROR, logger=svc.logger.name),
        pytest.raises(sqlite3.OperationalError, match="locked"),
    ):
        _run_pipeline(store, embedder)
    store.adelete_by_paper_id.assert_awaited_once_with(3)
    assert any(
        "compensation delete failed" in r.getMessage()
        and "paper_id=3" in r.getMessage()
        and "qdrant down" in r.getMessage()
        for r in caplog.records
    )


def test_pipeline_embed_failure_does_not_touch_db_or_delete_in_pipeline():
    # Failures before the DB write are compensated by the caller, not the pipeline.
    embedder, store = _deps()
    embedder.embed.side_effect = httpx.ConnectError("down")
    with (
        patch.object(svc, "extract_text", return_value=_PAGES),
        patch.object(svc, "chunk_pages", return_value=[{"text": "hello"}]),
        patch.object(svc, "save_chunks") as save,
        pytest.raises(httpx.ConnectError),
    ):
        _run_pipeline(store, embedder)
    save.assert_not_called()
    store.adelete_by_paper_id.assert_not_called()


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
            anyio.from_thread.start_blocking_portal() as portal,
        ):
            c = TestClient(app)
            c.portal = portal
            c.app.state.embedder = mock_embedder
            c.app.state.vector_store = mock_qdrant
            job_store._db_path = temp_db
            yield c


def _post_wait(client):
    return client.post(
        "/papers/ingest?wait=true",
        files={"file": ("gap.pdf", BytesIO(PDF), "application/pdf")},
    )


def test_wait_true_embed_failure_compensates_marks_failed_and_maps_503(client):
    client.app.state.embedder.embed.side_effect = httpx.ConnectError("embedding-svc down")
    delete = client.app.state.vector_store.adelete_by_paper_id
    with patch("academic_paper.services.ingest_service.extract_text", return_value=_PAGES):
        resp = _post_wait(client)
    assert resp.status_code == 503
    assert "embedding-svc" not in resp.json()["detail"]
    delete.assert_awaited_once()
    paper_id = delete.await_args.args[0]
    assert client.get(f"/papers/{paper_id}").json()["status"] == "failed"
    client.app.state.vector_store.aupsert.assert_not_called()


def test_wait_true_embed_failure_with_delete_failure_keeps_original_status(client, caplog):
    client.app.state.embedder.embed.side_effect = httpx.ConnectError("embedding-svc down")
    client.app.state.vector_store.adelete_by_paper_id.side_effect = RuntimeError("qdrant down")
    with patch("academic_paper.services.ingest_service.extract_text", return_value=_PAGES):
        with caplog.at_level(logging.ERROR):
            resp = _post_wait(client)
    assert resp.status_code == 503  # original error wins, not the delete failure
    assert any("compensation delete failed" in r.getMessage() for r in caplog.records)
