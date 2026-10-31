"""Sync sqlite3 / hash_file / tmp.write must run off the event-loop thread (#742)."""

import asyncio
import threading

from academic_paper import db as db_mod
from academic_paper.services import summary_service


def _record_threads(monkeypatch, module, name, seen):
    orig = getattr(module, name)

    def wrapper(*a, **kw):
        seen.append(threading.get_ident())
        return orig(*a, **kw)

    monkeypatch.setattr(module, name, wrapper)


def test_generate_summary_db_calls_run_off_loop(temp_db, override_settings, monkeypatch):
    override_settings(academic_db=temp_db)
    with db_mod.db_connection(temp_db) as conn:
        pid = db_mod.save_paper(conn, "a.pdf", "h1")

    seen: list[int] = []
    _record_threads(monkeypatch, summary_service, "get_paper", seen)
    _record_threads(monkeypatch, summary_service, "save_summary", seen)

    class FakeSummarizer:
        async def summarize(self, *a, **kw):
            return {}

    class FakeLLM:
        display_name = "m"

    async def run():
        loop_tid = threading.get_ident()
        await summary_service.generate_summary(pid, True, FakeLLM(), FakeSummarizer())
        return loop_tid

    loop_tid = asyncio.run(run())
    assert len(seen) == 2
    assert loop_tid not in seen


def test_cached_summary_endpoint_runs_off_loop(temp_db, override_settings, monkeypatch):
    from academic_paper.routers import summary as summary_router

    override_settings(academic_db=temp_db)
    seen: list[int] = []
    orig = summary_router.get_cached_summary

    def wrapper(pid):
        seen.append(threading.get_ident())
        return orig(pid)

    monkeypatch.setattr(summary_router, "get_cached_summary", wrapper)

    async def run():
        loop_tid = threading.get_ident()
        try:
            await summary_router.get_summary_endpoint(999)
        except Exception:
            pass  # 404 (paper missing) is fine; only the thread matters
        return loop_tid

    loop_tid = asyncio.run(run())
    assert seen and loop_tid not in seen


# --- remaining #742 sites (#783) ---------------------------------------------

_PDF = b"%PDF-1.4 minimal"


class _FakeUpload:
    """Minimal UploadFile stand-in: size unknown, content served in one chunk."""

    size = None
    filename = "a.pdf"

    def __init__(self, data=_PDF):
        self._chunks = [data, b""]

    async def read(self, n=-1):
        return self._chunks.pop(0)


def test_stream_upload_tmp_write_runs_off_loop(monkeypatch):
    import os
    import tempfile

    from academic_paper.services import ingest_service

    seen: list[int] = []

    class _Proxy:
        def __init__(self, real):
            self._real = real

        def write(self, data):
            seen.append(threading.get_ident())
            return self._real.write(data)

        def __getattr__(self, name):
            return getattr(self._real, name)

        def __enter__(self):
            self._real.__enter__()
            return self

        def __exit__(self, *exc):
            return self._real.__exit__(*exc)

    real_ntf = tempfile.NamedTemporaryFile
    monkeypatch.setattr(ingest_service.tempfile, "NamedTemporaryFile", lambda **kw: _Proxy(real_ntf(**kw)))

    async def run():
        loop_tid = threading.get_ident()
        path = await ingest_service.stream_upload_to_tmp(_FakeUpload(), 1024)
        return loop_tid, path

    loop_tid, path = asyncio.run(run())
    try:
        assert seen and loop_tid not in seen
    finally:
        os.unlink(path)


def test_ingest_hash_file_and_register_paper_run_off_loop(temp_db, override_settings, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from academic_paper.routers import papers as papers_router
    from academic_paper.services import ingest_service

    override_settings(academic_db=temp_db)
    hash_tids: list[int] = []
    register_tids: list[int] = []
    orig_hash = papers_router.hash_file
    orig_register = ingest_service.register_paper

    def hash_wrapper(path):
        hash_tids.append(threading.get_ident())
        return orig_hash(path)

    def register_wrapper(*a, **kw):
        register_tids.append(threading.get_ident())
        return orig_register(*a, **kw)

    monkeypatch.setattr(papers_router, "hash_file", hash_wrapper)
    monkeypatch.setattr(ingest_service, "register_paper", register_wrapper)
    # wait=True path: stub the pipeline so no embedder / Qdrant / PDF parsing is touched.
    monkeypatch.setattr(ingest_service, "ingest_pipeline", AsyncMock(return_value=3))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(embedder=None, vector_store=None)))

    async def run():
        loop_tid = threading.get_ident()
        result = await papers_router.ingest_paper(request, _FakeUpload(), None, None, None, None, None, wait=True)
        return loop_tid, result

    loop_tid, result = asyncio.run(run())
    assert result["status"] == "indexed" and result["chunks"] == 3
    assert hash_tids and loop_tid not in hash_tids
    assert register_tids and loop_tid not in register_tids


def test_health_responds_while_db_call_is_blocked(monkeypatch):
    """#742/#783: a blocked sync DB call must not stall the event loop, so /health still answers.

    Verified by ordering, not elapsed time: the DB stand-in blocks on a threading.Event that is
    only released *after* /health has returned. If the DB call ran on the loop thread, /health
    could not be served and the test would hang on the safety timeout (and fail), never flake
    on a timing threshold. No lifespan is run (as in test_server's `client` fixture), so no real
    embedder / Qdrant / DB is touched; the API key is a test-only dummy.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    import httpx

    from academic_paper.config import settings
    from academic_paper.routers import summary as summary_router
    from academic_paper.server import app

    release = threading.Event()
    db_entered = threading.Event()
    events: list[str] = []

    def blocking_get_cached_summary(paper_id):
        db_entered.set()
        assert release.wait(timeout=10), "DB stand-in was never released"
        events.append("db_done")
        return {"paper_id": paper_id}

    monkeypatch.setattr(summary_router, "get_cached_summary", blocking_get_cached_summary)
    app.state.vector_store = MagicMock(aping=AsyncMock(return_value=None))
    app.state.embedder = MagicMock(health=AsyncMock(return_value=None))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            db_task = asyncio.create_task(http.get("/papers/1/summary", headers={"X-API-Key": "test-dummy-key"}))
            # Wait (without blocking the loop) until the DB stand-in is really inside its blocking call.
            while not db_entered.is_set():
                await asyncio.sleep(0.01)
            health = await asyncio.wait_for(http.get("/health"), timeout=5)
            events.append("health_done")
            assert not db_task.done()  # DB call is still blocked when /health has already answered
            release.set()
            db_resp = await asyncio.wait_for(db_task, timeout=5)
            return health, db_resp

    with patch.object(settings, "api_key", "test-dummy-key"), patch.object(settings, "auth_disabled", False):
        health, db_resp = asyncio.run(run())

    assert health.status_code == 200
    assert db_resp.status_code == 200
    assert events == ["health_done", "db_done"]


def test_run_summarize_all_db_calls_run_off_loop(temp_db, override_settings, monkeypatch):
    from academic_paper.db import update_paper_status
    from academic_paper.jobs import job_store

    override_settings(academic_db=temp_db)
    job_store._db_path = temp_db
    with db_mod.db_connection(temp_db) as conn:
        pid = db_mod.save_paper(conn, "s.pdf", "hs")
        update_paper_status(conn, pid, "indexed")

    seen: dict[str, list[int]] = {"select": [], "save": []}
    _record_threads(monkeypatch, summary_service, "_select_unsummarized", seen["select"])
    _record_threads(monkeypatch, summary_service, "_save_summary_row", seen["save"])

    class FakeSummarizer:
        async def summarize(self, *a, **kw):
            return {}

    class FakeLLM:
        display_name = "m"

    async def run():
        loop_tid = threading.get_ident()
        job = await job_store.create(kind="summarize-all")
        await summary_service.run_summarize_all(job.id, FakeSummarizer(), FakeLLM(), job_store)
        return loop_tid, job

    loop_tid, job = asyncio.run(run())
    assert job.status == "done" and job.processed == 1
    assert seen["select"] and seen["save"]
    assert loop_tid not in seen["select"] + seen["save"]
