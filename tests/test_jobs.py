"""Tests for POST /jobs/summarize-all, GET /jobs, GET /jobs/{job_id}, and job persistence."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.db import (
    get_connection,
    load_all_jobs,
    save_chunks,
    save_paper,
    save_summary,
    update_paper_status,
    upsert_job,
)
from academic_paper.jobs import Job, JobStore, job_store
from academic_paper.server import app


@pytest.fixture(autouse=True)
def reset_job_store(temp_db):
    """Point the job store at the test DB (state reset is in conftest.py)."""
    job_store._db_path = temp_db
    yield


@pytest.fixture
def mock_summarizer():
    s = MagicMock()
    s.summarize = AsyncMock(
        return_value={
            "objective": "obj",
            "method": "meth",
            "results": "res",
            "limitations": "lim",
            "keywords": ["kw"],
        }
    )
    return s


@pytest.fixture
def client(temp_db, mock_summarizer):
    with patch.object(settings, "academic_db", temp_db):
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        mock_llm.__class__ = type("GeminiClient", (), {})
        with (
            patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
            patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
            patch("academic_paper.server.get_llm_client", return_value=mock_llm),
            patch("academic_paper.server.RAGSummarizer", return_value=mock_summarizer),
        ):
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
        with (
            patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
            patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
            patch("academic_paper.server.get_llm_client", return_value=None),
        ):
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
    save_chunks(
        conn,
        pid,
        [{"text": "t", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": "qb1", "token_count": 1}],
    )
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
    save_chunks(
        conn,
        pid,
        [{"text": "t", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": "qb2", "token_count": 1}],
    )
    update_paper_status(conn, pid, "indexed")
    save_summary(
        conn, pid, "model", {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": []}
    )
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
    job_store._jobs["running-job"] = Job(id="running-job", status="running", kind="summarize-all")

    resp = client.post("/jobs/summarize-all")
    assert resp.status_code == 409
    assert "already running" in resp.json()["detail"]


def test_start_summarize_all_409_when_pending(client):
    """Regression (#132): a still-pending job must also block a second start (TOCTOU)."""
    job_store._jobs["pending-job"] = Job(id="pending-job", status="pending", kind="summarize-all")

    resp = client.post("/jobs/summarize-all")
    assert resp.status_code == 409


def test_start_summarize_all_not_blocked_by_ingest_job(client):
    """Regression (#132): an in-flight ingest job must not block summarize-all."""
    job_store._jobs["ingest-job"] = Job(id="ingest-job", status="running", kind="ingest")

    resp = client.post("/jobs/summarize-all")
    assert resp.status_code == 202


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
        save_chunks(
            conn,
            pid,
            [{"text": "t", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": f"qe{i}", "token_count": 1}],
        )
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


# --- Sync SQLite writes must not block the event loop (#277) ---


@pytest.mark.anyio
async def test_persist_runs_off_event_loop_thread(temp_db):
    """JobStore.persist() offloads the synchronous sqlite3 write via asyncio.to_thread.

    Regression for #277: a naive `await`-less synchronous call would run
    _persist() directly on the event loop thread, which is exactly the bug
    #277 reports (it blocks the loop for up to busy_timeout=5000ms, delaying
    unrelated asyncio.wait_for timeouts elsewhere in the process). Recording
    the executing thread's identity proves the write runs elsewhere.
    """
    import threading

    store = JobStore()
    store._db_path = temp_db
    job = Job(id="thread-check", status="pending", kind="ingest")

    calling_thread_ids: list[int] = []
    real_persist = store._persist

    def spy_persist(j):
        calling_thread_ids.append(threading.get_ident())
        return real_persist(j)

    with patch.object(store, "_persist", side_effect=spy_persist):
        await store.persist(job)

    assert calling_thread_ids
    assert calling_thread_ids[0] != threading.get_ident()


@pytest.mark.anyio
async def test_create_runs_off_event_loop_thread(temp_db):
    """JobStore.create() also offloads its SQLite persist via asyncio.to_thread (#277)."""
    import threading

    store = JobStore()
    store._db_path = temp_db

    calling_thread_ids: list[int] = []
    real_persist = store._persist

    def spy_persist(j):
        calling_thread_ids.append(threading.get_ident())
        return real_persist(j)

    with patch.object(store, "_persist", side_effect=spy_persist):
        await store.create(kind="ingest")

    assert calling_thread_ids
    assert calling_thread_ids[0] != threading.get_ident()


# --- create_if_not_running atomicity (#235) ---


def test_create_if_not_running_is_atomic_under_concurrency(temp_db):
    """Concurrent create_if_not_running(kind) calls must yield exactly one created Job.

    Regression for #235: has_running() + create() as two separate lock acquisitions let
    two concurrent callers both observe "not running" and both create a job. This drives
    many threads at create_if_not_running() at once and asserts only one succeeds.

    create_if_not_running() is async (#277: its SQLite persist runs via
    asyncio.to_thread so it doesn't block the event loop), so each worker thread
    drives its own event loop with asyncio.run() — the lock-protected check-and-create
    section it guards is still shared, real-OS-thread state, so the race it tests for
    is unaffected.
    """
    import asyncio
    import threading

    store = JobStore()
    store._db_path = temp_db

    results: list[Job | None] = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        results.append(asyncio.run(store.create_if_not_running(kind="ingest")))

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    created = [r for r in results if r is not None]
    assert len(created) == 1
    assert len(store._jobs) == 1


# --- persist() ordering under out-of-order thread completion (#298) ---


@pytest.mark.anyio
async def test_persist_serializes_out_of_order_thread_completion(temp_db):
    """A slow-to-complete older persist() must not overwrite a faster newer one.

    Regression for #298: persist() ran each write via a fresh asyncio.to_thread()
    call with no ordering guarantee, so if the *first* persist() call (older state)
    happened to finish its SQLite write *after* a *second* persist() call (newer
    state), the older state clobbered the newer one. Here the first call is made
    to block until the second call's write has already completed, proving that
    without per-job serialization the write order would invert; with the fix the
    second call must wait for the first lock holder, so the final row reflects the
    call order (newer state) instead of the completion order.
    """
    import asyncio

    store = JobStore()
    store._db_path = temp_db
    job = Job(id="race-job", status="pending", kind="ingest")

    started = asyncio.Event()
    release = asyncio.Event()
    real_persist = store._persist

    def slow_first_persist(j):
        started.set()
        # Block the first call's worker thread until the second persist() call
        # (issued after this one, for the newer state) would have already
        # written to SQLite absent serialization.
        import time as _time

        deadline = _time.time() + 2
        while not release.is_set() and _time.time() < deadline:
            _time.sleep(0.01)
        return real_persist(j)

    with patch.object(store, "_persist", side_effect=slow_first_persist):
        job.status = "running"
        first = asyncio.create_task(store.persist(job))
        await started.wait()

        newer_job = Job(id="race-job", status="done", kind="ingest", processed=1, total=1)
        second = store.persist(newer_job)

        release.set()
        await first
        await second

    conn = get_connection(temp_db)
    rows = {r["id"]: r for r in load_all_jobs(conn)}
    conn.close()
    assert rows["race-job"]["status"] == "done"


@pytest.mark.anyio
async def test_job_store_init_loads_existing_jobs(temp_db):
    """JobStore.init() loads previously persisted jobs from SQLite."""
    import time as time_mod

    conn = get_connection(temp_db)
    upsert_job(conn, "job-abc", "done", 3, 3, 0, [], time_mod.time(), time_mod.time())
    conn.close()

    store = JobStore()
    await store.init(temp_db)

    job = store.get("job-abc")
    assert job is not None
    assert job.status == "done"
    assert job.total == 3
    assert job.processed == 3


@pytest.mark.anyio
async def test_running_job_converted_to_failed_on_init(temp_db):
    """Jobs with status='running' are converted to 'failed' when JobStore reloads."""
    import time as time_mod

    conn = get_connection(temp_db)
    upsert_job(conn, "job-crash", "running", 5, 2, 0, [], time_mod.time(), None)
    conn.close()

    store = JobStore()
    await store.init(temp_db)

    job = store.get("job-crash")
    assert job is not None
    assert job.status == "failed"
    assert any("restarted" in e for e in job.errors)

    # Also verify the DB was updated
    conn = get_connection(temp_db)
    rows = load_all_jobs(conn)
    conn.close()
    assert rows[0]["status"] == "failed"


@pytest.mark.anyio
async def test_create_before_init_is_flushed_not_dropped(temp_db):
    """Jobs created before init() sets _db_path (#302) must survive, not vanish.

    create()/create_if_not_running() can race init() at startup: a job created
    while _db_path is still None used to be silently dropped by _persist()'s
    `if not self._db_path: return` guard, with no log and no error. init()
    completing later must flush that buffered job to SQLite instead of losing it.
    """
    store = JobStore()
    # _db_path is still None here — simulates create() winning the startup race.
    job = await store.create(kind="ingest")

    # The job is in memory but not yet in SQLite.
    conn = get_connection(temp_db)
    assert load_all_jobs(conn) == []
    conn.close()

    await store.init(temp_db)

    # init() flushed the buffered job to SQLite.
    conn = get_connection(temp_db)
    rows = load_all_jobs(conn)
    conn.close()
    assert [r["id"] for r in rows] == [job.id]
    assert not store._pending_writes
