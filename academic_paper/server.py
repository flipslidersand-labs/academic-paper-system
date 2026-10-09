"""FastAPI server for academic paper ingestion and retrieval."""

import asyncio
import logging
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from academic_paper.auth import require_scope
from academic_paper.config import settings
from academic_paper.db import (
    db_connection,
    delete_paper,
    get_paper,
    init_db,
    list_papers_filtered,
    save_paper,
    update_paper_status,
)
from academic_paper.embedder import EmbedderClient
from academic_paper.errors import _http_exc_for, _safe_error_message  # noqa: F401  (re-exported, #614/#615)
from academic_paper.extractor import hash_file
from academic_paper.jobs import job_store
from academic_paper.llm import get_llm_client
from academic_paper.logging_config import configure_logging
from academic_paper.services import ingest_service
from academic_paper.services.search_service import run_search
from academic_paper.services.summary_service import (  # noqa: F401  (_summary_response re-exported, #618)
    _summary_response,
    generate_summary,
    get_cached_summary,
    list_all_summaries,
    run_summarize_all,
    score_all,
    score_one,
)
from academic_paper.summarizer import RAGSummarizer
from academic_paper.telemetry import get_tracer, setup_telemetry
from academic_paper.validators import (  # noqa: F401  (re-exported for backward compat, #614)
    _parse_list_field,
    _sanitize_text,
    _validate_published_date,
)
from academic_paper.vector_store import QdrantStore

logger = logging.getLogger(__name__)

# Max seconds shutdown waits for in-flight ingest tasks before cancelling them (#194).
INGEST_SHUTDOWN_TIMEOUT_S = 30.0
tracer = get_tracer()


async def _probe_startup_health(app: FastAPI) -> None:
    """Probe Qdrant and embedding-svc at startup; log warnings on failure."""
    try:
        await app.state.vector_store.aping()
        logger.info("Startup probe OK: Qdrant")
    except Exception:
        logger.warning(
            "Startup probe: Qdrant unreachable at %s — ingest/search will fail until available",
            settings.qdrant_url,
        )

    try:
        await app.state.embedder.health(timeout=2.0)
        logger.info("Startup probe OK: embedding-svc")
    except Exception:
        logger.warning(
            "Startup probe: embedding-svc unreachable at %s — ingest/search will fail until available",
            app.state.embedder.base_url,
        )


async def _cleanup_orphaned_ingests(app: FastAPI) -> None:
    """On startup, mark papers stuck in 'pending' as 'failed' and remove orphaned Qdrant vectors.

    A paper stays 'pending' when the server is killed mid-ingest (after Qdrant
    upsert but before save_chunks commits).  The JobStore already marks the
    corresponding job 'failed'; this function ensures the paper row and any
    partially-uploaded vectors are also cleaned up so the file can be re-ingested.
    """
    try:
        with db_connection(settings.academic_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM papers WHERE status = 'pending'")
            stuck_ids = [row[0] for row in cursor.fetchall()]
        if not stuck_ids:
            return
        logger.warning("Startup: found %d papers stuck in 'pending'; cleaning up Qdrant vectors", len(stuck_ids))
        for paper_id in stuck_ids:
            try:
                await app.state.vector_store.adelete_by_paper_id(paper_id)
            except Exception as exc:
                logger.warning("Startup cleanup: Qdrant delete failed for paper_id=%s: %s", paper_id, exc)
            with db_connection(settings.academic_db) as conn:
                update_paper_status(conn, paper_id, "failed")
    except Exception as exc:
        logger.warning("Startup cleanup failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and services on startup."""
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    setup_telemetry(app, settings.otel_endpoint)
    init_db(settings.academic_db)
    await job_store.init(settings.academic_db)
    # Separate clients: embedding batches can take 60–120 s for large PDFs
    # (EMBEDDING_TIMEOUT); Qdrant calls are fast point operations (QDRANT_TIMEOUT).
    # Using qdrant_timeout for both caused ReadTimeout on big ingest batches (#153).
    embed_client = httpx.AsyncClient(timeout=settings.embedding_timeout)
    app.state.embedder = EmbedderClient(client=embed_client)
    app.state.vector_store = QdrantStore()
    # Ollama gets a lifespan-managed persistent AsyncClient so TCP connections are
    # reused across summarize-all iterations (#192). The client is injected (not
    # owned by the LLM client), so this lifespan closes it.
    ollama_http_client: httpx.AsyncClient | None = None
    provider = settings.llm_provider
    if provider == "ollama" or (provider == "auto" and not settings.google_api_key and settings.ollama_url):
        ollama_http_client = httpx.AsyncClient(timeout=settings.ollama_timeout)
    llm_client = get_llm_client(http_client=ollama_http_client)
    app.state.llm = llm_client
    if llm_client is not None:
        app.state.summarizer = RAGSummarizer(llm_client, app.state.vector_store, app.state.embedder)
    else:
        app.state.summarizer = None
    # Track active background ingest tasks so shutdown can wait for them (#194).
    app.state.active_ingest_tasks: set[asyncio.Task] = set()
    # Keep a reference so the task isn't garbage-collected mid-flight
    # (documented asyncio pitfall), and cancel it on shutdown.
    app.state.probe_task = asyncio.create_task(_probe_startup_health(app))
    await _cleanup_orphaned_ingests(app)
    if not settings.auth_enabled:
        logger.warning(
            "API_KEY is not set (nor API_KEYS / INGEST_API_KEY) — all endpoints (including write endpoints) are unauthenticated (#241)"
        )
    yield
    # Graceful shutdown: wait up to 30 s for in-flight ingest tasks (#194).
    active = list(app.state.active_ingest_tasks)
    if active:
        logger.info("Shutdown: waiting for %d active ingest task(s) (timeout 30s)", len(active))
        try:
            await asyncio.wait_for(asyncio.gather(*active, return_exceptions=True), timeout=INGEST_SHUTDOWN_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning("Shutdown: ingest tasks did not finish in 30s; cancelling")
            for t in active:
                t.cancel()
    app.state.probe_task.cancel()
    await embed_client.aclose()
    if app.state.llm is not None:
        await app.state.llm.aclose()
    if ollama_http_client is not None:
        await ollama_http_client.aclose()
    await app.state.vector_store.aclose()


app = FastAPI(title="Academic Paper System", lifespan=lifespan)


_require_authenticated = require_scope()  # any valid key, no scope required (#602)


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Require X-API-Key on write and read endpoints when API_KEY env var is set (#146).

    Applied to read endpoints too (#241): paper text/search snippets are at least
    as sensitive as the job metadata already gated behind auth (#190), so the
    boundary must not be asymmetric.

    Uses hmac.compare_digest for constant-time comparison to prevent
    timing attacks that could leak key length / prefix (#190).
    """
    await _require_authenticated(x_api_key)


# /metrics is added by Instrumentator.expose() rather than manually registered, so it
# must be gated the same way as every other route: pass verify_api_key through as a
# dependency (#299 — request-path counters and latency histograms are otherwise
# readable without auth whenever API_KEY is set).
Instrumentator().instrument(app).expose(app, dependencies=[Depends(verify_api_key)])


async def _ingest_pipeline(tmp_path: str, paper_id: int, file_hash: str, file_name: str) -> int:
    """Thin wrapper: supplies app.state deps to ingest_service.ingest_pipeline (#615)."""
    return await ingest_service.ingest_pipeline(
        tmp_path,
        paper_id,
        file_hash,
        file_name,
        embedder=app.state.embedder,
        vector_store=app.state.vector_store,
    )


async def _compensate_qdrant(paper_id: int) -> None:
    await ingest_service.compensate_qdrant(app.state.vector_store, paper_id)


async def _mark_paper_failed(paper_id: int) -> None:
    await ingest_service.mark_paper_failed(paper_id)


_unlink_quiet = ingest_service.unlink_quiet


async def _run_ingest(job_id: str, tmp_path: str, paper_id: int, file_hash: str, file_name: str) -> None:
    await ingest_service.run_ingest(
        job_id,
        tmp_path,
        paper_id,
        file_hash,
        file_name,
        embedder=app.state.embedder,
        vector_store=app.state.vector_store,
        job_store=job_store,
    )


@app.post("/papers/ingest", dependencies=[Depends(verify_api_key)])
async def ingest_paper(
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
    tmp_path: str | None = None
    keep_tmp = False
    try:
        _validate_published_date(published_date)
        max_bytes = settings.max_upload_mb * 1024 * 1024
        size_detail = f"File too large (max {settings.max_upload_mb} MB)"
        # Reject early when the multipart part already declares an oversized length,
        # before buffering anything.
        if file.size is not None and file.size > max_bytes:
            raise HTTPException(status_code=413, detail=size_detail)

        # Stream to disk in 1 MiB chunks, enforcing the size cap as bytes arrive
        # so an oversized body is cut off mid-transfer instead of being fully
        # buffered in memory first.
        header = b""
        received = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(1 << 20):
                received += len(chunk)
                if received > max_bytes:
                    raise HTTPException(status_code=413, detail=size_detail)
                if len(header) < 5:
                    header += chunk[: 5 - len(header)]
                tmp.write(chunk)

        if not header.startswith(b"%PDF-"):
            raise HTTPException(status_code=415, detail="Not a PDF file (missing %PDF- header)")

        file_hash = hash_file(tmp_path)
        # Sanitize filename to prevent log injection and stored XSS (#189).
        # Allow only word chars, dots, hyphens, spaces; replace everything else with '_'.
        raw_name = file.filename or "unknown.pdf"
        file_name = re.sub(r"[^\w.\- ]", "_", raw_name)[:255]

        with db_connection(settings.academic_db) as conn:
            cursor = conn.cursor()
            # 409 when already indexed or still being ingested ('pending'); failed rows
            # are removed so the same PDF can be re-uploaded after a partial failure (#145).
            # Deleting a 'pending' row would pull it out from under the running job and
            # its compensating Qdrant delete could wipe the new paper's shared points (#496).
            # Stale 'pending' rows from a killed server are turned 'failed' at startup.
            cursor.execute("SELECT id, status FROM papers WHERE file_hash = ?", (file_hash,))
            existing = cursor.fetchone()
            if existing:
                if existing["status"] == "indexed":
                    raise HTTPException(status_code=409, detail="File already ingested")
                if existing["status"] == "pending":
                    raise HTTPException(status_code=409, detail="File is already being ingested")
                # 'failed' row — purge and re-ingest
                delete_paper(conn, existing["id"])

            paper_id = save_paper(
                conn,
                file_name,
                file_hash,
                title=_sanitize_text(title),
                authors=_parse_list_field(authors, "authors"),
                categories=_parse_list_field(categories, "categories"),
                published_date=published_date or None,
                source=_sanitize_text(source),
            )

        if wait:
            try:
                chunks = await _ingest_pipeline(tmp_path, paper_id, file_hash, file_name)
            except Exception as e:
                logger.exception("Synchronous ingest failed for paper_id=%s", paper_id)
                await _mark_paper_failed(paper_id)
                # _ingest_pipeline already compensates Qdrant on save_chunks failure;
                # compensate here for embed/upsert errors that leave no Qdrant data.
                await _compensate_qdrant(paper_id)
                raise _http_exc_for(e, "Ingest failed: check PDF content and try again")
            return {
                "paper_id": paper_id,
                "file_name": file_name,
                "chunks": chunks,
                "status": "indexed",
            }

        job = await job_store.create(kind="ingest")
        keep_tmp = True  # background task now owns tmp cleanup
        task = asyncio.create_task(_run_ingest(job.id, tmp_path, paper_id, file_hash, file_name))
        # A task cancelled before its first step (shutdown) never enters
        # _run_ingest's try/finally, so also unlink when the task completes (#454).
        task.add_done_callback(lambda _t, p=tmp_path: _unlink_quiet(p))
        active = getattr(app.state, "active_ingest_tasks", None)
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
            _unlink_quiet(tmp_path)


@app.get("/papers", dependencies=[Depends(verify_api_key)])
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


@app.get("/papers/{paper_id}", dependencies=[Depends(verify_api_key)])
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


@app.get("/papers/{paper_id}/summary", dependencies=[Depends(verify_api_key)])
async def get_summary_endpoint(paper_id: int):
    """Return the cached summary only (GET is safe/idempotent, #140). Generation lives in POST."""
    return get_cached_summary(paper_id)


@app.post("/papers/{paper_id}/summary", dependencies=[Depends(verify_api_key)])
async def generate_summary_endpoint(paper_id: int, force: bool = Query(False)):
    """Generate the summary (cached result is returned unless force=true)."""
    return await generate_summary(paper_id, force, app.state.llm, app.state.summarizer)


@app.post("/papers/score-all", dependencies=[Depends(verify_api_key)])
def score_all_papers():
    """Compute and store relevance scores for all papers.

    Score = freshness (30-day half-life, 0–0.5) + category match (0–0.5).
    Preferred categories are configured via PREFERRED_CATEGORIES env var.
    A per-paper failure is counted in `failed`/`errors` and does not abort the run (#276).

    Returns:
        JSON with total, scored, failed counts and per-paper errors.
    """
    return score_all()


@app.post("/papers/{paper_id}/score", dependencies=[Depends(verify_api_key)])
def score_paper(paper_id: int):
    """Compute and store relevance score for a single paper."""
    return score_one(paper_id)


@app.get("/summaries", dependencies=[Depends(verify_api_key)])
def list_summaries_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all paper summaries with associated paper metadata."""
    return list_all_summaries(limit, offset)


async def _run_summarize_all(job_id: str) -> None:
    """Background task: summarize all indexed papers without a cached summary (#620)."""
    await run_summarize_all(job_id, app.state.summarizer, app.state.llm, job_store)


@app.post("/jobs/summarize-all", status_code=202, dependencies=[Depends(verify_api_key)])
async def start_summarize_all(background_tasks: BackgroundTasks):
    """Start a background job to summarize all papers without a cached summary."""
    if app.state.llm is None or app.state.summarizer is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    job = await job_store.create_if_not_running(kind="summarize-all")
    if job is None:
        raise HTTPException(status_code=409, detail="A summarize-all job is already running")
    background_tasks.add_task(_run_summarize_all, job.id)
    return {"job_id": job.id, "status": job.status}


@app.get("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
def get_job_endpoint(job_id: str):
    """Get status of a background job by ID."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/jobs", dependencies=[Depends(verify_api_key)])
def list_jobs_endpoint():
    """List all background jobs (#190: require auth — exposes all job IDs / paper metadata)."""
    return {"jobs": [j.to_dict() for j in job_store.list_all()]}


@app.get("/search", dependencies=[Depends(verify_api_key)])
async def search(
    q: str = Query(..., min_length=1, max_length=1000),
    mode: str = Query("hybrid", pattern="^(vector|keyword|hybrid|nugget)$"),
    limit: int = Query(10, ge=1, le=100),
    paper_id: int | None = Query(None),
    snippet_length: int = Query(
        200, ge=0, description="Max snippet length (0=full text); affects keyword/vector/hybrid only"
    ),
    nuggets_per_chunk: int = Query(3, ge=1, le=10, description="Sentences per chunk (nugget mode only)"),
    nugget_embed_weight: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="Embedding weight in nugget hybrid scoring: 0=BM25-only, 1=embed-only "
        "(nugget mode only; nugget-rag-eval #11 found 0.7 optimal)",
    ),
):
    """Search papers using vector, keyword, hybrid, or nugget mode.

    nugget mode: runs hybrid search then extracts the top-N most query-relevant
    sentences from each chunk instead of returning the full snippet. Reduces
    context length by ~68% while maintaining Recall@5.
    """
    try:
        with db_connection(settings.academic_db) as conn:
            return await run_search(
                conn,
                app.state.embedder,
                app.state.vector_store,
                q=q,
                mode=mode,
                limit=limit,
                paper_id=paper_id,
                snippet_length=snippet_length,
                nuggets_per_chunk=nuggets_per_chunk,
                nugget_embed_weight=nugget_embed_weight,
            )
    except Exception as e:
        logger.exception("Search error for query=%r mode=%s", q, mode)
        raise _http_exc_for(e, "Search failed: check query and try again")


@app.get("/health")
async def health():
    """Qdrant / embedding-svc 疎通チェック。全アップストリーム障害時は HTTP 503 を返す (#195)。"""
    status = {"qdrant": "ok", "embedding_svc": "ok"}
    overall = "ok"

    try:
        await app.state.vector_store.aping()
    except Exception:
        status["qdrant"] = "error"
        overall = "degraded"

    try:
        await app.state.embedder.health()
    except Exception:
        status["embedding_svc"] = "error"
        overall = "degraded"

    http_status = 503 if overall == "degraded" else 200
    return JSONResponse(content={"status": overall, **status}, status_code=http_status)


@app.get("/stats", dependencies=[Depends(verify_api_key)])
def stats():
    """DB統計情報"""
    try:
        with db_connection(settings.academic_db) as conn:
            papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            try:
                qdrant_points = app.state.vector_store.count_points()
            except Exception:
                qdrant_points = -1
        return {"papers": papers, "chunks": chunks, "qdrant_points": qdrant_points}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get stats")
        raise _http_exc_for(e, "Failed to get stats")


# Serve frontend at /ui — must be mounted after all API routes
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
