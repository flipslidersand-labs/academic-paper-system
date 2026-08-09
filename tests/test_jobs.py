"""Tests for POST /jobs/summarize-all, GET /jobs, GET /jobs/{job_id}, and job persistence."""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.db import (
    get_connection,
    init_db,
    load_all_jobs,
    save_chunks,
    save_paper,
    save_summary,
    upsert_job,
    update_paper_status,
)
from academic_paper.jobs import Job, JobStore, job_store
from academic_paper.server import app


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    init_db(db_path)
    yield db_path


@pytest.fixture(autouse=True)
def reset_job_store(temp_db):
    """Reset job store state and point it at the test DB between tests."""
    job_store._jobs.clear()
    job_store._db_path = temp_db
    yield
    job_store._jobs.clear()
    job_store._db_path = None


@pytest.fixture
def mock_summarizer():
    s = MagicMock()
    s.summarize = AsyncMock(return_value={
        "objective": "obj", "method": "meth", "results": "res",
        "limitations": "lim", "keywords": ["kw"],
    })
    return s


@pytest.fixture
def client(temp_db, mock_summarizer):
    with patch.object(settings, "academic_db", temp_db):
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        mock_llm.__class__ = type("GeminiClient", (), {})
        with patch("academic_paper.server.EmbedderClient", return_value=mock_embedder), \
             patch("academic_paper.server.QdrantStore", return_value=mock_qdrant), \
             patch("academic_paper.server.get_llm_client", return_value=mock_llm), \
             patch("academic_paper.server.RAGSummarizer", return_value=mock_summarizer):
            c = TestClient(app)
            c.app.state.embedder = mock_embedder
            c.app.state.vector_store = mock_qdrant
            c.app.state.llm = mock_llm
            c.app.state.summarizer = mock_summarizer
            yield c


@pytest.fixture
def client_no_llm(temp_db):
    with patch.object(settings, "academic_db", temp_db):
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()
        with patch("academic_paper.server.EmbedderClient", return_value=mock_embedder), \
             patch("academic_paper.server.QdrantStore", return_value=mock_qdrant), \
             patch("academic_paper.server.get_llm_client", return_value=None):
            c = TestClient(app)
            c.app.state.embedder = mock_embedder
            c.app.state.vector_store = mock_qdrant
            c.app.state.llm = None
            c.app.state.summarizer = None
            yield c


# --- API tests ---

def test_start_summarize_all_no_papers(client):
    """POST /jobs/summarize-all with no papers returns job with total=0 and status=done."""
    resp = client.post("/jobs/summarize-all")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["total"] == 0
    assert job["processed"] == 0
    assert job["failed"] == 0
    assert job["finished_at"] is not None


def test_start_summarize_all_processes_indexed_papers(client, temp_db):
    """POST /jobs/summarize-all summarizes indexed papers without cached summaries."""
    conn = get_connection(temp_db)
    pid = save_paper(conn, "p.pdf", "h_bulk1")
    save_chunks(conn, pid, [{"text": "t", "page_start": 1, "page_end": 1,
                              "chunk_index": 0, "qdrant_id": "qb1", "token_count": 1}])
    update_paper_status(conn, pid, "indexed")
    conn.close()

    resp = client.post("/jobs/summarize-all")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["total"] == 1
    assert job["processed"] == 1
    assert job["failed"] == 0


def test_start_summarize_all_skips_already_summarized(client, temp_db):
    """POST /jobs/summarize-all skips papers that already have a cached summary."""
    conn = get_connection(temp_db)
    pid = save_paper(conn, "p.pdf", "h_bulk2")
    save_chunks(conn, pid, [{"text": "t", "page_start": 1, "page_end": 1,
                              "chunk_index": 0, "qdrant_id": "qb2", "token_count": 1}])
    update_paper_status(conn, pid, "indexed")
    save_summary(conn, pid, "model", {
        "objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": []
    })
    conn.close()

    resp = client.post("/jobs/summarize-all")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["total"] == 0
    assert job["processed"] == 0


def test_start_summarize_all_409_when_running(client):
    """POST /jobs/summarize-all returns 409 if a job is already running."""
    job_store._jobs["running-job"] = Job(id="running-job", status="running")

    resp = client.post("/jobs/summarize-all")
    assert resp.status_code == 409
    assert "already running" in resp.json()["detail"]


def test_start_summarize_all_503_no_llm(client_no_llm):
    """POST /jobs/summarize-all returns 503 when LLM is not configured."""
    resp = client_no_llm.post("/jobs/summarize-all")
    assert resp.status_code == 503
    assert "LLM" in resp.json()["detail"]


def test_get_job_not_found(client):
    """GET /jobs/{job_id} returns 404 for an unknown job ID."""
    resp = client.get("/jobs/nonexistent-uuid")
    assert resp.status_code == 404


def test_list_jobs_empty(client):
    """GET /jobs returns empty list when no jobs have been created."""
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert resp.json()["jobs"] == []


def test_list_jobs_includes_completed(client):
    """GET /jobs returns all jobs including completed ones."""
    client.post("/jobs/summarize-all")

    resp = client.get("/jobs")
    assert resp.status_code == 200
    jobs = resp.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "done"


def test_summarize_all_records_per_paper_errors(client, temp_db, mock_summarizer):
    """POST /jobs/summarize-all records failed papers and still completes the job."""
    conn = get_connection(temp_db)
    for i in range(2):
        pid = save_paper(conn, f"p{i}.pdf", f"h_err_{i}")
        save_chunks(conn, pid, [{"text": "t", "page_start": 1, "page_end": 1,
                                  "chunk_index": 0, "qdrant_id": f"qe{i}", "token_count": 1}])
        update_paper_status(conn, pid, "indexed")
    conn.close()

    mock_summarizer.summarize = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    resp = client.post("/jobs/summarize-all")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["total"] == 2
    assert job["processed"] == 0
    assert job["failed"] == 2
    assert len(job["errors"]) == 2
    assert "LLM timeout" in job["errors"][0]


# --- Persistence tests ---

def test_completed_job_persisted_to_sqlite(client, temp_db):
    """A completed job is written to the jobs table in SQLite."""
    resp = client.post("/jobs/summarize-all")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    conn = get_connection(temp_db)
    rows = load_all_jobs(conn)
    conn.close()

    assert len(rows) == 1
    assert rows[0]["id"] == job_id
    assert rows[0]["status"] == "done"
    assert rows[0]["finished_at"] is not None


def test_job_store_init_loads_existing_jobs(temp_db):
    """JobStore.init() loads previously persisted jobs from SQLite."""
    import time as time_mod

    conn = get_connection(temp_db)
    upsert_job(conn, "job-abc", "done", 3, 3, 0, [], time_mod.time(), time_mod.time())
    conn.close()

    store = JobStore()
    store.init(temp_db)

    job = store.get("job-abc")
    assert job is not None
    assert job.status == "done"
    assert job.total == 3
    assert job.processed == 3


def test_running_job_converted_to_failed_on_init(temp_db):
    """Jobs with status='running' are converted to 'failed' when JobStore reloads."""
    import time as time_mod

    conn = get_connection(temp_db)
    upsert_job(conn, "job-crash", "running", 5, 2, 0, [], time_mod.time(), None)
    conn.close()

    store = JobStore()
    store.init(temp_db)

    job = store.get("job-crash")
    assert job is not None
    assert job.status == "failed"
    assert any("restarted" in e for e in job.errors)

    # Also verify the DB was updated
    conn = get_connection(temp_db)
    rows = load_all_jobs(conn)
    conn.close()
    assert rows[0]["status"] == "failed"
