"""Regression tests for the Qdrant compensation path in the ingest pipeline (#489, #145)."""

import sqlite3
import time
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import anyio.from_thread
import pytest
from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.jobs import job_store
from academic_paper.server import app

PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"


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
            # Lifespan (which awaits job_store.init) is patched out (#477).
            job_store._db_path = temp_db
            yield c


def _wait_for_job(client, job_id, timeout=3.0):
    deadline = time.monotonic() + timeout
    job = client.get(f"/jobs/{job_id}").json()
    while job["status"] in ("pending", "running") and time.monotonic() < deadline:
        time.sleep(0.01)
        job = client.get(f"/jobs/{job_id}").json()
    return job


def _post(client, wait):
    return client.post(
        "/papers/ingest" + ("?wait=true" if wait else ""),
        files={"file": ("comp.pdf", BytesIO(PDF), "application/pdf")},
    )


def test_wait_true_db_failure_compensates_and_marks_failed(client):
    delete = client.app.state.vector_store.adelete_by_paper_id
    with (
        patch("academic_paper.server.extract_text", return_value=[{"page": 1, "text": "content"}]),
        patch("academic_paper.server.save_chunks", side_effect=sqlite3.OperationalError("disk I/O error")),
    ):
        resp = _post(client, wait=True)
    assert resp.status_code == 500
    assert "disk I/O error" not in resp.json()["detail"]
    # Pipeline compensation + the wait=true double compensation, both with the paper_id.
    assert delete.await_count >= 1
    paper_id = delete.await_args.args[0]
    assert all(c.args == (paper_id,) for c in delete.await_args_list)
    assert client.get(f"/papers/{paper_id}").json()["status"] == "failed"


def test_wait_true_compensation_delete_failure_keeps_original_error(client):
    delete = client.app.state.vector_store.adelete_by_paper_id
    delete.side_effect = RuntimeError("qdrant down")
    with (
        patch("academic_paper.server.extract_text", return_value=[{"page": 1, "text": "content"}]),
        patch("academic_paper.server.save_chunks", side_effect=sqlite3.OperationalError("locked")),
    ):
        resp = _post(client, wait=True)
    # Status is decided by the original DB error (500), not the delete failure.
    assert resp.status_code == 500
    assert delete.await_count >= 1


def test_async_db_failure_compensates_and_job_failed(client):
    delete = client.app.state.vector_store.adelete_by_paper_id
    with (
        patch("academic_paper.server.extract_text", return_value=[{"page": 1, "text": "content"}]),
        patch("academic_paper.server.save_chunks", side_effect=sqlite3.OperationalError("disk I/O error")),
    ):
        resp = _post(client, wait=False)
        assert resp.status_code == 202
        body = resp.json()
        job = _wait_for_job(client, body["job_id"])
    assert job["status"] == "failed"
    delete.assert_awaited_with(body["paper_id"])
    assert client.get(f"/papers/{body['paper_id']}").json()["status"] == "failed"


def test_async_compensation_delete_failure_still_fails_job(client):
    client.app.state.vector_store.adelete_by_paper_id.side_effect = RuntimeError("qdrant down")
    with (
        patch("academic_paper.server.extract_text", return_value=[{"page": 1, "text": "content"}]),
        patch("academic_paper.server.save_chunks", side_effect=sqlite3.OperationalError("locked")),
    ):
        body = _post(client, wait=False).json()
        job = _wait_for_job(client, body["job_id"])
    assert job["status"] == "failed"
    assert "locked" in job["errors"][0]


def test_reupload_after_compensated_failure_is_accepted(client):
    with (
        patch("academic_paper.server.extract_text", return_value=[{"page": 1, "text": "content"}]),
        patch("academic_paper.server.save_chunks", side_effect=sqlite3.OperationalError("disk I/O error")),
    ):
        assert _post(client, wait=True).status_code == 500
    with patch("academic_paper.server.extract_text", return_value=[{"page": 1, "text": "content"}]):
        resp = _post(client, wait=True)
    assert resp.status_code == 200
    assert resp.json()["status"] == "indexed"
