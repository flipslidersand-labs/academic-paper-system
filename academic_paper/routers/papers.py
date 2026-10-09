"""Papers routes (ingest / list / get), extracted from server.py (#358, #617).

Path, response and OpenAPI are unchanged. verify_api_key is applied at the router
level. App-level dependencies (embedder, vector_store, active ingest tasks) are
read from ``request.app.state`` so this module does not import server.py.
"""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from academic_paper.auth import verify_api_key
from academic_paper.config import settings
from academic_paper.db import db_connection, get_paper, list_papers_filtered
from academic_paper.errors import _http_exc_for
from academic_paper.extractor import hash_file
from academic_paper.jobs import job_store
from academic_paper.services import ingest_service
from academic_paper.validators import _parse_list_field, _sanitize_text, _validate_published_date

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])

_DUPLICATE_DETAIL = {"indexed": "File already ingested", "pending": "File is already being ingested"}


@router.post("/papers/ingest")
async def ingest_paper(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None, max_length=1000),
    authors: str | None = Form(None, max_length=10000),
    categories: str | None = Form(None, max_length=2000),
    published_date: str | None = Form(None, max_length=10),
    source: str | None = Form(None, max_length=100),
    wait: bool = Query(False, description="Process synchronously and return the indexed result (200)"),
):
    """Ingest a PDF paper with optional metadata.

    Accepts the upload, deduplicates by file hash, saves a 'pending' paper row,
    then processes (extract → chunk → embed → Qdrant upsert) in the background,
    returning **202** with a `job_id`. Poll `GET /jobs/{job_id}` for completion;
    on `done` the job's `result` holds `paper_id`/`chunks`.

    Set `wait=true` to process synchronously and receive the indexed result (200).

    Raises:
        HTTPException 409: File already ingested, or the same file is currently
            being ingested (existing row is 'indexed' or 'pending').
        HTTPException 413: File exceeds the max upload size.
        HTTPException 415: File is not a PDF (missing %PDF- magic bytes).
        HTTPException 422: Invalid metadata (non-ISO published_date, non-string list elements).
        HTTPException 400: Extraction / chunking / embedding error (wait=true only).
    """
    state = request.app.state
    tmp_path: str | None = None
    keep_tmp = False
    try:
        _validate_published_date(published_date)
        max_bytes = settings.max_upload_mb * 1024 * 1024
        try:
            tmp_path = await ingest_service.stream_upload_to_tmp(file, max_bytes)
        except ingest_service.UploadTooLargeError:
            raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_mb} MB)")
        except ingest_service.NotAPdfError:
            raise HTTPException(status_code=415, detail="Not a PDF file (missing %PDF- header)")

        file_hash = hash_file(tmp_path)
        file_name = ingest_service.sanitize_file_name(file.filename)

        try:
            paper_id = ingest_service.register_paper(
                file_name,
                file_hash,
                lambda: {
                    "title": _sanitize_text(title),
                    "authors": _parse_list_field(authors, "authors"),
                    "categories": _parse_list_field(categories, "categories"),
                    "published_date": published_date or None,
                    "source": _sanitize_text(source),
                },
            )
        except ingest_service.DuplicatePaperError as dup:
            raise HTTPException(status_code=409, detail=_DUPLICATE_DETAIL[dup.status])

        if wait:
            try:
                chunks = await ingest_service.ingest_pipeline(
                    tmp_path, paper_id, file_hash, file_name, embedder=state.embedder, vector_store=state.vector_store
                )
            except Exception as e:
                logger.exception("Synchronous ingest failed for paper_id=%s", paper_id)
                await ingest_service.mark_paper_failed(paper_id)
                # _ingest_pipeline already compensates Qdrant on save_chunks failure;
                # compensate here for embed/upsert errors that leave no Qdrant data.
                await ingest_service.compensate_qdrant(state.vector_store, paper_id)
                raise _http_exc_for(e, "Ingest failed: check PDF content and try again")
            return {
                "paper_id": paper_id,
                "file_name": file_name,
                "chunks": chunks,
                "status": "indexed",
            }

        job = await job_store.create(kind="ingest")
        keep_tmp = True  # background task now owns tmp cleanup
        task = asyncio.create_task(
            ingest_service.run_ingest(
                job.id,
                tmp_path,
                paper_id,
                file_hash,
                file_name,
                embedder=state.embedder,
                vector_store=state.vector_store,
                job_store=job_store,
            )
        )
        # A task cancelled before its first step (shutdown) never enters
        # _run_ingest's try/finally, so also unlink when the task completes (#454).
        task.add_done_callback(lambda _t, p=tmp_path: ingest_service.unlink_quiet(p))
        active = getattr(state, "active_ingest_tasks", None)
        if active is not None:
            active.add(task)
            task.add_done_callback(active.discard)
        return JSONResponse(
            status_code=202,
            content={"job_id": job.id, "paper_id": paper_id, "status": "pending"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during paper ingest")
        raise _http_exc_for(e, "Ingest failed unexpectedly")
    finally:
        if tmp_path is not None and not keep_tmp:
            ingest_service.unlink_quiet(tmp_path)


@router.get("/papers")
def list_papers_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    author: str | None = Query(None, description="Filter by author name (substring match)"),
    category: str | None = Query(None, description="Filter by category code e.g. cs.AI"),
    sort: Literal["ingested_at", "score"] = Query("ingested_at", description="Sort order"),
):
    """List papers with pagination, optional filters, and sort."""
    try:
        with db_connection(settings.academic_db) as conn:
            total, papers = list_papers_filtered(
                conn, limit=limit, offset=offset, author=author, category=category, sort=sort
            )
        return {"total": total, "papers": papers}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list papers")
        raise _http_exc_for(e, "Failed to list papers")


@router.get("/papers/{paper_id}")
def get_paper_endpoint(paper_id: int):
    """Get paper details by ID."""
    try:
        with db_connection(settings.academic_db) as conn:
            paper = get_paper(conn, paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get paper_id=%s", paper_id)
        raise _http_exc_for(e, "Failed to get paper")
