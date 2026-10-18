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
