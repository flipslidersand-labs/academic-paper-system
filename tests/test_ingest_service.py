"""Unit tests for academic_paper/services/ingest_service.py (no FastAPI app booted, #615)."""

import asyncio
import functools
import os
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from academic_paper.jobs import Job
from academic_paper.services import ingest_service as svc


def _sync(fn):
    """Run an async test body with asyncio.run (repo has no pytest-asyncio auto mode)."""

    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))

    return wrapper


_PAGES = [{"page": 1, "text": "hello world"}]


def _deps():
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2]])
    store = MagicMock()
    store.aensure_collection = AsyncMock()
    store.aupsert = AsyncMock()
    store.adelete_by_paper_id = AsyncMock()
    return embedder, store


@_sync
async def test_pipeline_success_upserts_and_marks_indexed():
    embedder, store = _deps()
    with (
        patch.object(svc, "extract_text", return_value=_PAGES),
        patch.object(svc, "chunk_pages", return_value=[{"text": "hello"}]),
        patch.object(svc, "db_connection"),
        patch.object(svc, "save_chunks") as save,
        patch.object(svc, "update_paper_status") as status,
    ):
        n = await svc.ingest_pipeline("/x.pdf", 7, "h", "f.pdf", embedder=embedder, vector_store=store)
    assert n == 1
    store.aupsert.assert_awaited_once()
    assert store.aupsert.await_args.args[0][0]["payload"]["paper_id"] == 7
    save.assert_called_once()
    assert status.call_args.args[1:] == (7, "indexed")
    store.adelete_by_paper_id.assert_not_called()


@_sync
async def test_pipeline_no_text_raises_value_error():
    embedder, store = _deps()
    with patch.object(svc, "extract_text", return_value=[]), pytest.raises(ValueError):
        await svc.ingest_pipeline("/x.pdf", 1, "h", "f.pdf", embedder=embedder, vector_store=store)
    embedder.embed.assert_not_called()


@_sync
async def test_pipeline_db_failure_compensates_qdrant():
    embedder, store = _deps()
    with (
        patch.object(svc, "extract_text", return_value=_PAGES),
        patch.object(svc, "chunk_pages", return_value=[{"text": "hello"}]),
        patch.object(svc, "db_connection"),
        patch.object(svc, "save_chunks", side_effect=sqlite3.OperationalError("locked")),
        pytest.raises(sqlite3.OperationalError),
    ):
        await svc.ingest_pipeline("/x.pdf", 3, "h", "f.pdf", embedder=embedder, vector_store=store)
    store.adelete_by_paper_id.assert_awaited_once_with(3)


@_sync
async def test_compensate_qdrant_swallows_errors():
    _, store = _deps()
    store.adelete_by_paper_id.side_effect = RuntimeError("down")
    await svc.compensate_qdrant(store, 1)


@_sync
async def test_mark_paper_failed_sets_status_and_swallows_errors():
    with patch.object(svc, "db_connection"), patch.object(svc, "update_paper_status") as status:
        await svc.mark_paper_failed(5)
    assert status.call_args.args[1:] == (5, "failed")
    with patch.object(svc, "db_connection", side_effect=RuntimeError("locked")):
        await svc.mark_paper_failed(5)


def test_unlink_quiet(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"x")
    svc.unlink_quiet(str(f))
    assert not f.exists()
    svc.unlink_quiet(str(f))  # missing file: no error


def _job_store(job):
    js = MagicMock()
    js.get = MagicMock(return_value=job)
    js.persist = AsyncMock()
    return js


@_sync
async def test_run_ingest_failure_marks_paper_and_job_failed(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-")
    job = Job(id="j", status="pending", kind="ingest")
    embedder, store = _deps()
    with (
        patch.object(svc, "ingest_pipeline", AsyncMock(side_effect=ValueError("bad"))),
        patch.object(svc, "mark_paper_failed", AsyncMock()) as mark,
    ):
        await svc.run_ingest(
            "j", str(pdf), 9, "h", "f.pdf", embedder=embedder, vector_store=store, job_store=_job_store(job)
        )
    assert job.status == "failed" and job.failed == 1 and job.finished_at is not None
    mark.assert_awaited_once_with(9)
    store.adelete_by_paper_id.assert_awaited_once_with(9)
    assert not os.path.exists(pdf)


@_sync
async def test_run_ingest_success_sets_result(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-")
    job = Job(id="j", status="pending", kind="ingest")
    embedder, store = _deps()
    with patch.object(svc, "ingest_pipeline", AsyncMock(return_value=4)):
        await svc.run_ingest(
            "j", str(pdf), 9, "h", "f.pdf", embedder=embedder, vector_store=store, job_store=_job_store(job)
        )
    assert job.status == "done" and job.result["chunks"] == 4
    assert not os.path.exists(pdf)


class _FakeUpload:
    def __init__(self, data: bytes, size=None, filename="a.pdf"):
        self._data = data
        self._pos = 0
        self.size = size
        self.filename = filename

    async def read(self, n):
        out = self._data[self._pos : self._pos + n]
        self._pos += n
        return out


@_sync
async def test_stream_upload_ok_returns_path_with_content():
    path = await svc.stream_upload_to_tmp(_FakeUpload(b"%PDF-1.4 body"), 1024)
    try:
        assert open(path, "rb").read() == b"%PDF-1.4 body"
    finally:
        os.unlink(path)


@_sync
async def test_stream_upload_not_pdf_removes_tmp():
    created = []
    orig = svc.tempfile.NamedTemporaryFile

    def spy(**kw):
        ctx = orig(**kw)
        created.append(ctx.name)
        return ctx

    with patch.object(svc.tempfile, "NamedTemporaryFile", side_effect=spy), pytest.raises(svc.NotAPdfError):
        await svc.stream_upload_to_tmp(_FakeUpload(b"hello world"), 1024)
    assert created and not os.path.exists(created[0])


@_sync
async def test_stream_upload_declared_size_too_large_creates_no_file():
    with patch.object(svc.tempfile, "NamedTemporaryFile") as ntf, pytest.raises(svc.UploadTooLargeError):
        await svc.stream_upload_to_tmp(_FakeUpload(b"%PDF-", size=2048), 1024)
    ntf.assert_not_called()


@_sync
async def test_stream_upload_streamed_size_too_large_removes_tmp():
    created = []
    orig = svc.tempfile.NamedTemporaryFile

    def spy(**kw):
        ctx = orig(**kw)
        created.append(ctx.name)
        return ctx

    with patch.object(svc.tempfile, "NamedTemporaryFile", side_effect=spy), pytest.raises(svc.UploadTooLargeError):
        await svc.stream_upload_to_tmp(_FakeUpload(b"%PDF-" + b"x" * 2000), 1024)
    assert created and not os.path.exists(created[0])


def test_sanitize_file_name():
    assert svc.sanitize_file_name(None) == "unknown.pdf"
    assert svc.sanitize_file_name("a<b>/c.pdf") == "a_b__c.pdf"
    assert len(svc.sanitize_file_name("x" * 400)) == 255


def _row(status):
    return {"id": 1, "status": status}


@pytest.mark.parametrize("status", ["indexed", "pending"])
def test_register_paper_duplicate_raises(status):
    with patch.object(svc, "db_connection") as dbc, patch.object(svc, "save_paper") as save:
        dbc.return_value.__enter__.return_value.cursor.return_value.fetchone.return_value = _row(status)
        with pytest.raises(svc.DuplicatePaperError) as ei:
            svc.register_paper("f.pdf", "h", lambda: {})
    assert ei.value.status == status
    save.assert_not_called()


def test_register_paper_new_saves_with_metadata():
    with patch.object(svc, "db_connection") as dbc, patch.object(svc, "save_paper", return_value=11) as save:
        dbc.return_value.__enter__.return_value.cursor.return_value.fetchone.return_value = None
        assert svc.register_paper("f.pdf", "h", lambda: {"title": "T"}) == 11
    assert save.call_args.kwargs == {"title": "T"}


def test_register_paper_failed_row_is_purged():
    with (
        patch.object(svc, "db_connection") as dbc,
        patch.object(svc, "save_paper", return_value=2),
        patch.object(svc, "delete_paper") as delete,
    ):
        dbc.return_value.__enter__.return_value.cursor.return_value.fetchone.return_value = _row("failed")
        assert svc.register_paper("f.pdf", "h", lambda: {}) == 2
    assert delete.call_args.args[1] == 1
