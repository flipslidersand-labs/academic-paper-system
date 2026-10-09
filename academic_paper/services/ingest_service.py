"""Ingest processing (extract -> chunk -> embed -> upsert) extracted from server.py (#358, #615).

No dependency on the FastAPI ``app`` / ``app.state``: embedder, vector_store and
job_store are passed in, so this module is testable without booting the app.
server.py keeps thin ``_``-prefixed wrappers that supply ``app.state`` objects.
"""

import asyncio
import logging
import os
import time

from academic_paper.chunker import chunk_pages
from academic_paper.config import settings
from academic_paper.db import db_connection, save_chunks, update_paper_status
from academic_paper.embedder import EmbedderClient
from academic_paper.errors import _safe_error_message
from academic_paper.extractor import extract_text
from academic_paper.jobs import JobStore
from academic_paper.telemetry import get_tracer
from academic_paper.vector_store import QdrantStore, make_qdrant_id

logger = logging.getLogger(__name__)
tracer = get_tracer()


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
