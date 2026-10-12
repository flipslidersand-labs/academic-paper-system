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
from academic_paper.config import settings
from academic_paper.db import db_connection, delete_paper, save_chunks, save_paper, update_paper_status
from academic_paper.embedder import EmbedderClient
from academic_paper.errors import _safe_error_message
from academic_paper.extractor import extract_text
from academic_paper.jobs import JobStore
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
                tmp.write(chunk)
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
    with db_connection(settings.academic_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, status FROM papers WHERE file_hash = ?", (file_hash,))
        existing = cursor.fetchone()
        if existing:
            if existing["status"] in ("indexed", "pending"):
                raise DuplicatePaperError(existing["status"])
            # 'failed' row — purge and re-ingest
            delete_paper(conn, existing["id"])
        return save_paper(conn, file_name, file_hash, **metadata_factory())


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
                asyncio.to_thread(extract_text, tmp_path), timeout=settings.pdf_extract_timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"PDF extraction exceeded {settings.pdf_extract_timeout}s timeout (#238)") from exc
    if not pages:
        raise ValueError("No text extracted from PDF")

    chunks_list = chunk_pages(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
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

    try:
        with db_connection(settings.academic_db) as conn:
            save_chunks(conn, paper_id, chunks_list)
            update_paper_status(conn, paper_id, "indexed")
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
            with db_connection(settings.academic_db) as conn:
                update_paper_status(conn, paper_id, "failed")

        await asyncio.to_thread(_update)
    except Exception:
        logger.error("Failed to mark paper_id=%s as failed (job status still updated)", paper_id, exc_info=True)


def unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


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
        job.status = "running"
        job.total = 1
        await job_store.persist(job)
        try:
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
            await mark_paper_failed(paper_id)
            # Upsert may have partially succeeded (batched); remove leftover vectors
            # so a failed paper is not returned by vector search (#471).
            await compensate_qdrant(vector_store, paper_id)
            job.failed = 1
            job.errors.append(f"paper_id={paper_id}: {_safe_error_message(e)}")
            job.status = "failed"
        finally:
            job.finished_at = time.time()
            await job_store.persist(job)
    finally:
        unlink_quiet(tmp_path)
