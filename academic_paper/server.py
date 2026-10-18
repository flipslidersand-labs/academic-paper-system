"""FastAPI server for academic paper ingestion and retrieval."""

import asyncio
import base64
import hashlib
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from academic_paper.auth import Scope, require_scope, verify_api_key  # noqa: F401  (verify_api_key re-exported, #617)
from academic_paper.config import settings
from academic_paper.db import (
    count_rows,
    db_connection,
    init_db,
    list_paper_ids_by_status,
    update_paper_status,
)
from academic_paper.embedder import EmbedderClient
from academic_paper.errors import _http_exc_for, _safe_error_message  # noqa: F401  (re-exported, #614/#615)
from academic_paper.jobs import job_store
from academic_paper.llm import get_llm_client
from academic_paper.logging_config import configure_logging
from academic_paper.routers.jobs import router as jobs_router
from academic_paper.routers.papers import router as papers_router
from academic_paper.routers.summary import router as summary_router
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

        def _list_stuck() -> list:
            with db_connection(settings.academic_db) as conn:
                return list_paper_ids_by_status(conn, "pending")

        def _mark_failed(pid: int) -> None:
            with db_connection(settings.academic_db) as conn:
                update_paper_status(conn, pid, "failed")

        stuck_ids = await asyncio.to_thread(_list_stuck)
        if not stuck_ids:
            return
        logger.warning("Startup: found %d papers stuck in 'pending'; cleaning up Qdrant vectors", len(stuck_ids))
        for paper_id in stuck_ids:
            try:
                await app.state.vector_store.adelete_by_paper_id(paper_id)
            except Exception as exc:
                logger.warning("Startup cleanup: Qdrant delete failed for paper_id=%s: %s", paper_id, exc)
            await asyncio.to_thread(_mark_failed, paper_id)
    except Exception as exc:
        logger.warning("Startup cleanup failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and services on startup."""
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    # Fail closed: no API key configured and auth not explicitly disabled (#646).
    if not settings.auth_enabled and not settings.auth_disabled:
        raise RuntimeError(
            "API_KEY / API_KEYS is not set. Refusing to start with unauthenticated endpoints; "
            "set API_KEY (or API_KEYS), or set AUTH_DISABLED=true to run without authentication explicitly (#646)."
        )
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
        logger.warning("AUTH_DISABLED=true — all endpoints (including write endpoints) are unauthenticated (#241)")
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


# /docs, /redoc and /openapi.json are unauthenticated and enumerate every endpoint, so they
# are off unless ENABLE_DOCS=true is set explicitly (#648).
def _docs_kwargs(enabled: bool) -> dict[str, None]:
    return {} if enabled else {"docs_url": None, "redoc_url": None, "openapi_url": None}


app = FastAPI(title="Academic Paper System", lifespan=lifespan, **_docs_kwargs(settings.enable_docs))


# /metrics is added by Instrumentator.expose() rather than manually registered, so it
# must be gated the same way as every other route: pass require_scope(Scope.ADMIN) as a
# dependency (#299 — request-path counters and latency histograms are otherwise
# readable without auth whenever API_KEY is set).
Instrumentator().instrument(app).expose(app, dependencies=[Depends(require_scope(Scope.ADMIN))])


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


app.include_router(papers_router)
app.include_router(jobs_router)
app.include_router(summary_router)


async def _run_summarize_all(job_id: str) -> None:
    """Background task: summarize all indexed papers without a cached summary (#620)."""
    await run_summarize_all(job_id, app.state.summarizer, app.state.llm, job_store)


@app.get("/search", dependencies=[Depends(require_scope(Scope.READ))])
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


@app.get("/stats", dependencies=[Depends(require_scope(Scope.READ))])
def stats():
    """DB統計情報"""
    try:
        with db_connection(settings.academic_db) as conn:
            counts = count_rows(conn)
            try:
                qdrant_points = app.state.vector_store.count_points()
            except Exception:
                qdrant_points = -1
        return {**counts, "qdrant_points": qdrant_points}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get stats")
        raise _http_exc_for(e, "Failed to get stats")


# Serve frontend at /ui — must be mounted after all API routes
_frontend_dir = Path(__file__).parent.parent / "frontend"


def _inline_script_hashes(html: str) -> list[str]:
    """CSP 'sha256-...' sources for each inline <script> block, so no 'unsafe-inline' is needed."""
    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.S)
    return [f"'sha256-{base64.b64encode(hashlib.sha256(b.encode()).digest()).decode()}'" for b in blocks]


def _ui_csp(index_html: str) -> str:
    # Alpine's CDN build evaluates x-* expressions via new Function, hence 'unsafe-eval' (#734);
    # migrating to @alpinejs/csp would remove it. style-src keeps 'unsafe-inline' for the <style> block.
    script_src = " ".join(["'self'", "https://cdn.jsdelivr.net", "'unsafe-eval'", *_inline_script_hashes(index_html)])
    return (
        f"default-src 'self'; script-src {script_src}; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    )


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
_ui_csp_value = (
    _ui_csp(_index.read_text(encoding="utf-8")) if (_index := _frontend_dir / "index.html").exists() else None
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """nosniff / frame / referrer headers on every response; CSP on the /ui shell (#734)."""
    response = await call_next(request)
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    path = request.url.path
    if _ui_csp_value and (path == "/ui" or path.startswith("/ui/")):
        response.headers.setdefault("Content-Security-Policy", _ui_csp_value)
    return response


if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
