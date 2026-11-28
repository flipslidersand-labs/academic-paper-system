"""Ingest processing (extract -> chunk -> embed -> upsert) extracted from server.py (#358, #615).

No dependency on the FastAPI ``app`` / ``app.state``: embedder, vector_store and
job_store are passed in, so this module is testable without booting the app.
server.py keeps thin ``_``-prefixed wrappers that supply ``app.state`` objects.
"""

import asyncio
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable

from academic_paper.chunker import chunk_pages
from academic_paper.config import get_settings
from academic_paper.db import (
    db_connection,
    delete_paper,
    get_paper_by_hash,
    save_chunks,
    save_paper,
    update_paper_status,
)
from academic_paper.embedder import EmbedderClient
from academic_paper.errors import PdfExtractionError, _safe_error_message
from academic_paper.extractor import extract_text
from academic_paper.jobs import Job, JobStore
from academic_paper.telemetry import get_tracer
from academic_paper.vector_store import QdrantStore, make_qdrant_id

logger = logging.getLogger(__name__)
tracer = get_tracer()

_PDF_MAGIC = b"%PDF-"
_READ_CHUNK = 1 << 20


class UploadTooLargeError(Exception):
    """Upload exceeds the configured max size (route maps to HTTP 413)."""


class NotAPdfError(Exception):
    """Upload lacks the %PDF- magic bytes (route maps to HTTP 415)."""


class DuplicatePaperError(Exception):
    """A paper with the same file_hash is indexed or being ingested (route maps to HTTP 409)."""

    def __init__(self, status: str):
        super().__init__(f"duplicate paper (status={status})")
        self.status = status


async def stream_upload_to_tmp(file, max_bytes: int) -> str:
    """Stream an upload to a temp .pdf file, enforcing size cap and PDF magic bytes.

    Returns the temp path (the caller owns cleanup). On any failure the temp file
    is removed before the exception propagates. Raises UploadTooLargeError /
    NotAPdfError (HTTP-independent).
    """
    # Reject early when the multipart part already declares an oversized length,
    # before buffering anything.
    if file.size is not None and file.size > max_bytes:
        raise UploadTooLargeError
    tmp_path: str | None = None
    try:
        # Stream to disk in 1 MiB chunks, enforcing the size cap as bytes arrive
        # so an oversized body is cut off mid-transfer instead of being fully
        # buffered in memory first.
        header = b""
        received = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(_READ_CHUNK):
                received += len(chunk)
                if received > max_bytes:
                    raise UploadTooLargeError
                if len(header) < len(_PDF_MAGIC):
                    header += chunk[: len(_PDF_MAGIC) - len(header)]
                await asyncio.to_thread(tmp.write, chunk)
        if not header.startswith(_PDF_MAGIC):
            raise NotAPdfError
        return tmp_path
    except BaseException:
        if tmp_path is not None:
            unlink_quiet(tmp_path)
        raise


def sanitize_file_name(raw_name: str | None) -> str:
    """Sanitize filename to prevent log injection and stored XSS (#189).

    Allow only word chars, dots, hyphens, spaces; replace everything else with '_'.
    """
    return re.sub(r"[^\w.\- ]", "_", raw_name or "unknown.pdf")[:255]


def register_paper(file_name: str, file_hash: str, metadata_factory: Callable[[], dict]) -> int:
    """Duplicate-check by file_hash, then save a 'pending' paper row. Returns paper_id.

    Raises DuplicatePaperError('indexed' | 'pending'). Failed rows are purged so the
    same PDF can be re-uploaded after a partial failure (#145). Deleting a 'pending'
    row would pull it out from under the running job and its compensating Qdrant
    delete could wipe the new paper's shared points (#496); stale 'pending' rows from
    a killed server are turned 'failed' at startup. ``metadata_factory`` is called
    only after the duplicate check so metadata validation errors keep their order.
    """
    with db_connection(get_settings().academic_db) as conn:
        existing = get_paper_by_hash(conn, file_hash)
        if existing:
            if existing["status"] in ("indexed", "pending"):
                raise DuplicatePaperError(existing["status"])
            # 'failed' row — purge and re-ingest
            delete_paper(conn, existing["id"])
        return save_paper(conn, file_name, file_hash, **metadata_factory())


class CancelledAfterIndexed(asyncio.CancelledError):
    """ingest_pipeline was cancelled, but its DB write had already committed (paper is 'indexed') (#878)."""


async def _await_settled[T](fut: asyncio.Future[T], what: str) -> T:
    """Await ``fut`` shielded; if cancelled meanwhile, wait for it to settle, then re-raise (#855, #872).

    Cancelling the awaiting task does not stop the underlying work, so callers that must
    observe its final state (a committed DB write, a persisted job) wait for it to finish,
    even through repeated cancellation, before the CancelledError propagates. A failure of
    ``fut`` during that wait is logged; the CancelledError is still what propagates.
    """
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        while not fut.done():
            try:
                await asyncio.shield(fut)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not fut.cancelled() and fut.exception() is not None:
            logger.error("%s failed while cancellation was pending", what, exc_info=fut.exception())
        raise


async def ingest_pipeline(
    tmp_path: str,
    paper_id: int,
    file_hash: str,
    file_name: str,
    *,
    embedder: EmbedderClient,
    vector_store: QdrantStore,
) -> int:
    """Extract → chunk → embed → Qdrant upsert for a saved paper. Returns chunk count.

    Raises on extraction/chunking/embedding failure; the caller is responsible
    for updating the paper status to 'failed'.
    """
    with tracer.start_as_current_span("pdf.extract"):
        # extract_text is CPU-bound / sync I/O — run in thread pool (#149).
        # extract_text() has no page/time limit of its own, so a malformed or
        # huge PDF can otherwise hang the job forever (#238); bound the wait here.
        try:
            pages = await asyncio.wait_for(
                asyncio.to_thread(extract_text, tmp_path), timeout=get_settings().pdf_extract_timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"PDF extraction exceeded {get_settings().pdf_extract_timeout}s timeout (#238)") from exc
        except Exception as exc:
            # Corrupt/encrypted PDFs raise pdfminer errors (not ValueError): classify as 400 (#729).
            raise PdfExtractionError("Failed to extract text from PDF") from exc
    if not pages:
        raise ValueError("No text extracted from PDF")

    chunks_list = chunk_pages(pages, chunk_size=get_settings().chunk_size, overlap=get_settings().chunk_overlap)
    if not chunks_list:
        raise ValueError("No chunks generated")

    chunk_texts = [chunk["text"] for chunk in chunks_list]
    with tracer.start_as_current_span("embed.batch"):
        embeddings = await embedder.embed(chunk_texts, mode="index")

    await vector_store.aensure_collection()

    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks_list, embeddings)):
        qdrant_id = make_qdrant_id(file_hash, idx)
        chunk["qdrant_id"] = qdrant_id
        points.append(
            {
                "id": qdrant_id,
                "vector": embedding,
                "payload": {
                    "paper_id": paper_id,
                    "chunk_index": idx,
                    "text": chunk["text"],
                    "file_name": file_name,
                },
            }
        )

    with tracer.start_as_current_span("qdrant.upsert"):
        await vector_store.aupsert(points)

    def _write() -> None:
        # Connection lifecycle stays inside the worker thread (sqlite3 check_same_thread) (#855).
        with db_connection(get_settings().academic_db) as conn:
            save_chunks(conn, paper_id, chunks_list)
            update_paper_status(conn, paper_id, "indexed")

    try:
        # Thousands of chunk INSERTs (+ FTS) must not block the event loop (#855).
        # Keep a reference and shield it: cancelling the await does not stop the worker
        # thread, so on cancellation wait for the write to settle before re-raising. That
        # way compensation (mark_paper_failed / compensate_qdrant) always runs after the
        # 'indexed' write is committed, never before it.
        fut = asyncio.ensure_future(asyncio.to_thread(_write))
        try:
            await _await_settled(fut, f"DB write for paper_id={paper_id}")
        except asyncio.CancelledError as exc:
            if not fut.cancelled() and fut.exception() is None:
                # The write committed before the cancel surfaced: tell run_ingest so the
                # job's error can say the paper is indexed (#878).
                raise CancelledAfterIndexed() from exc
            raise
    except Exception:
        # Qdrant upsert succeeded but DB write failed — compensate by deleting
        # the orphaned vectors so the paper can be re-ingested (#145).
        try:
            await vector_store.adelete_by_paper_id(paper_id)
        except Exception as qdrant_exc:
            logger.error("Qdrant compensation delete failed for paper_id=%s: %s", paper_id, qdrant_exc)
        raise

    return len(chunks_list)


async def compensate_qdrant(vector_store: QdrantStore, paper_id: int) -> None:
    """Best-effort: delete a paper's Qdrant vectors after a failed ingest (#471).

    Upsert is sent in batches, so a mid-way failure leaves earlier batches behind.
    A failure here is logged and swallowed so it never masks the original error.
    """
    try:
        await vector_store.adelete_by_paper_id(paper_id)
    except Exception as exc:
        logger.error("Qdrant compensation delete failed for paper_id=%s: %s", paper_id, exc)


async def mark_paper_failed(paper_id: int) -> None:
    """Best-effort: set a paper's status to 'failed' without blocking the event loop.

    update_paper_status is synchronous sqlite3 I/O with a 5s busy_timeout, so it
    is run in a thread (#277-style). A failure here (e.g. 'database is locked')
    is logged and swallowed rather than raised, so callers can update job status
    first and unconditionally — a paper-status write failure must never prevent
    the job from being marked 'failed' (#421).
    """
    try:

        def _update() -> None:
            with db_connection(get_settings().academic_db) as conn:
                update_paper_status(conn, paper_id, "failed")

        await asyncio.to_thread(_update)
    except Exception:
        logger.error("Failed to mark paper_id=%s as failed (job status still updated)", paper_id, exc_info=True)


def unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


async def _persist_shielded(job_store: JobStore, job: Job) -> None:
    """persist(job) that survives repeated task cancellation (#872)."""
    await _await_settled(asyncio.ensure_future(job_store.persist(job)), f"persist of final state of job={job.id}")


async def run_ingest(
    job_id: str,
    tmp_path: str,
    paper_id: int,
    file_hash: str,
    file_name: str,
    *,
    embedder: EmbedderClient,
    vector_store: QdrantStore,
    job_store: JobStore,
) -> None:
    """Background task: run the ingest pipeline, updating job + paper status.

    Owns ``tmp_path`` cleanup on every path, including the early return when
    the job is gone and a failure in the initial ``persist`` (#454).
    """
    try:
        job = job_store.get(job_id)
        if job is None:
            return
        try:
            # running 化の persist もここで包む: 失敗しても except で paper/job を
            # 終端状態にし、paper が pending・job が running のまま固着しない (#725)。
            job.status = "running"
            job.total = 1
            await job_store.persist(job)
            chunks = await ingest_pipeline(
                tmp_path, paper_id, file_hash, file_name, embedder=embedder, vector_store=vector_store
            )
            job.result = {"paper_id": paper_id, "file_name": file_name, "chunks": chunks, "status": "indexed"}
            job.processed = 1
            job.status = "done"
        except Exception as e:
            logger.exception("Background ingest failed for paper_id=%s", paper_id)
            # mark_paper_failed never raises (#421): it swallows and logs its own
            # errors, so the job attributes below always run afterward instead of
            # being skipped by an exception from the paper-status write.
            # Terminalize the job before the compensation awaits so a cancel (or failure)
            # during compensation still leaves a failed job with the reason (#872).
            job.failed = 1
            job.errors.append(f"paper_id={paper_id}: {_safe_error_message(e)}")
            job.status = "failed"
            # status と同時に finished_at を入れる: 補償の await 中に GET /jobs が来ても、
            # failed かつ finished_at=None の中間状態を見せない(finally は後から上書きしない)。
            job.finished_at = time.time()
            await mark_paper_failed(paper_id)
            # Upsert may have partially succeeded (batched); remove leftover vectors
            # so a failed paper is not returned by vector search (#471).
            await compensate_qdrant(vector_store, paper_id)
        except asyncio.CancelledError as exc:
            # CancelledError is a BaseException so `except Exception` misses it; without
            # this the job stays 'running' with finished_at set (#872). Terminalize the
            # job in memory first (the finally below persists it), then re-raise. Paper
            # state is left to ingest_pipeline's shielded write / startup cleanup so we
            # don't double-compensate.
            logger.warning("Background ingest cancelled for paper_id=%s job=%s", paper_id, job_id)
            job.failed = 1
            # The write may have committed just before the cancel surfaced (#878): the
            # paper is then 'indexed' while the job is 'failed'. Say so in the error
            # (the job stays failed; the API shape is unchanged).
            note = ""
            if isinstance(exc, CancelledAfterIndexed):
                note = " (paper is indexed: DB write completed before cancellation)"
            job.errors.append(f"paper_id={paper_id}: cancelled{note}")
            job.status = "failed"
            raise
        finally:
            if job.finished_at is None:
                job.finished_at = time.time()
            try:
                await _persist_shielded(job_store, job)
            except Exception:
                # in-memory の job は既に終端状態。ここで投げるとタスク例外が未取得で落ちるだけ (#725)。
                logger.error("Failed to persist final state of job=%s", job_id, exc_info=True)
    finally:
        unlink_quiet(tmp_path)
