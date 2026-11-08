"""wait=true ingest cancelled mid-pipeline must fail the paper and compensate (#852)."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from academic_paper.routers import papers as papers_router
from academic_paper.services import ingest_service

PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"


class _Upload:
    size = None
    filename = "a.pdf"

    def __init__(self):
        self._data = PDF

    async def read(self, n=-1):
        data, self._data = self._data, b""
        return data


def _status(db, paper_id):
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT status FROM papers WHERE id = ?", (paper_id,)).fetchone()[0]
    finally:
        conn.close()


def _call(vector_store):
    return papers_router.ingest_paper(
        _Upload(), None, None, None, None, None, wait=True, embedder=None, vector_store=vector_store, active=None
    )


def test_cancelled_wait_ingest_marks_failed_compensates_and_allows_reupload(temp_db, override_settings, monkeypatch):
    override_settings(academic_db=temp_db)
    started = asyncio.Event
    store = MagicMock()
    store.adelete_by_paper_id = AsyncMock()
    seen = {}

    async def hang(tmp_path, paper_id, *a, **kw):
        seen["paper_id"] = paper_id
        seen["started"].set()
        await asyncio.sleep(60)

    monkeypatch.setattr(ingest_service, "ingest_pipeline", hang)

    async def run():
        seen["started"] = started()
        task = asyncio.create_task(_call(store))
        waiter = asyncio.create_task(seen["started"].wait())
        await asyncio.wait({task, waiter}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        assert seen["started"].is_set(), "pipeline never started"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert _status(temp_db, seen["paper_id"]) == "failed"
    store.adelete_by_paper_id.assert_awaited_once_with(seen["paper_id"])

    # Re-upload of the same PDF is no longer a 409 duplicate.
    monkeypatch.setattr(ingest_service, "ingest_pipeline", AsyncMock(return_value=1))
    result = asyncio.run(_call(store))
    assert result["status"] == "indexed"


def test_cancel_still_propagates_when_compensation_fails(temp_db, override_settings, monkeypatch):
    override_settings(academic_db=temp_db)
    started = asyncio.Event
    store = MagicMock()
    store.adelete_by_paper_id = AsyncMock(side_effect=RuntimeError("qdrant down"))
    seen = {}

    async def hang(*a, **kw):
        seen["started"].set()
        await asyncio.sleep(60)

    monkeypatch.setattr(ingest_service, "ingest_pipeline", hang)
    monkeypatch.setattr(ingest_service, "mark_paper_failed", AsyncMock(side_effect=RuntimeError("db down")))

    async def run():
        seen["started"] = started()
        task = asyncio.create_task(_call(store))
        waiter = asyncio.create_task(seen["started"].wait())
        await asyncio.wait({task, waiter}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        assert seen["started"].is_set(), "pipeline never started"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
