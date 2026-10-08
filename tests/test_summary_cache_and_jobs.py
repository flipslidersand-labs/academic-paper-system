"""Tests for the summary cache-hit path and _run_summarize_all failure branches (#491)."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.db import get_connection, save_chunks, save_paper
from academic_paper.jobs import job_store
from academic_paper.server import _run_summarize_all, app

SUMMARY = {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}


@pytest.fixture(autouse=True)
def _job_db(temp_db):
    job_store._db_path = temp_db


@pytest.fixture
def summarizer():
    s = MagicMock()
    s.summarize = AsyncMock(return_value=SUMMARY)
    return s


@pytest.fixture
def client(temp_db, summarizer):
    with patch.object(settings, "academic_db", temp_db), patch.object(settings, "api_key", ""):
        with (
            patch("academic_paper.server.EmbedderClient", return_value=MagicMock()),
            patch("academic_paper.server.QdrantStore", return_value=MagicMock()),
        ):
            c = TestClient(app)
            llm = MagicMock()
            llm.display_name = "test-model"
            c.app.state.llm = llm
            c.app.state.summarizer = summarizer
            yield c


def _paper(temp_db, file_hash="h491"):
    conn = get_connection(temp_db)
    pid = save_paper(conn, "p.pdf", file_hash)
    save_chunks(
        conn,
        pid,
        [{"text": "t", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": "q491", "token_count": 1}],
    )
    conn.close()
    return pid


def test_second_post_returns_cached_without_recomputing(client, temp_db, summarizer):
    pid = _paper(temp_db)
    first = client.post(f"/papers/{pid}/summary")
    second = client.post(f"/papers/{pid}/summary")
    assert first.status_code == second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.json()["objective"] == "o"
    assert summarizer.summarize.await_count == 1


def test_post_summary_missing_paper_returns_404(client):
    resp = client.post("/papers/9999/summary")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def _run_job():
    async def go():
        job = await job_store.create(kind="summarize-all")
        await _run_summarize_all(job.id)
        return job

    return asyncio.run(go())


def test_summarize_all_summarizer_none_marks_job_failed(client):
    client.app.state.summarizer = None
    job = _run_job()
    assert job.status == "failed"
    assert job.errors
    assert "summarizer" in job.errors[0].lower()
    assert job.finished_at is not None


def test_summarize_all_db_exception_marks_job_failed(client):
    with patch("academic_paper.server.db_connection", side_effect=sqlite3.OperationalError("db boom")):
        job = _run_job()
    assert job.status == "failed"
    assert any("db boom" in e for e in job.errors)
    assert job.finished_at is not None
