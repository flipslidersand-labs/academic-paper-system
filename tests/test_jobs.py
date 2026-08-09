"""Tests for JobStore persistence."""

import pytest

from academic_paper.db import get_connection, init_db, load_all_jobs, upsert_job
from academic_paper.jobs import JobStore


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def test_create_and_get_job(db_path):
    store = JobStore()
    store.init(db_path)

    job = store.create("summarize-all")

    assert job["id"] is not None
    assert job["status"] == "pending"
    assert job["type"] == "summarize-all"

    fetched = store.get(job["id"])
    assert fetched is not None
    assert fetched["id"] == job["id"]


def test_running_jobs_become_failed_on_init(db_path):
    """Jobs with status=running at shutdown must be marked failed on next init."""
    conn = get_connection(db_path)
    running_job = {
        "id": "test-running-job",
        "type": "summarize-all",
        "status": "running",
        "created_at": "2026-08-10T00:00:00",
        "started_at": "2026-08-10T00:01:00",
        "finished_at": None,
        "result": None,
        "error": None,
    }
    upsert_job(conn, running_job)
    conn.close()

    store = JobStore()
    store.init(db_path)

    job = store.get("test-running-job")
    assert job is not None
    assert job["status"] == "failed"
    assert job["error"] is not None

    # Verify it's also persisted in DB
    conn2 = get_connection(db_path)
    rows = load_all_jobs(conn2)
    conn2.close()
    db_job = next(r for r in rows if r["id"] == "test-running-job")
    assert db_job["status"] == "failed"


def test_persist_updates_db(db_path):
    store = JobStore()
    store.init(db_path)

    job = store.create("summarize-all")
    job["status"] = "completed"
    job["result"] = {"total": 3, "succeeded": 3, "failed": 0}
    store.persist(job)

    conn = get_connection(db_path)
    rows = load_all_jobs(conn)
    conn.close()

    db_job = next(r for r in rows if r["id"] == job["id"])
    assert db_job["status"] == "completed"
    assert db_job["result"]["total"] == 3
