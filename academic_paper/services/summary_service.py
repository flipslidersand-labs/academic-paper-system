"""Summary retrieval/generation logic for the summary endpoints, extracted from server.py (#359 1/5, #618).

server.py's summary handlers stay thin FastAPI shims: they pass ``app.state.llm`` /
``app.state.summarizer`` in as arguments and call these functions, which raise
``HTTPException`` exactly as the inline handlers did, so the response shape and the
error mapping are unchanged. This module has no dependency on the FastAPI app.
"""

import logging

from fastapi import HTTPException

from academic_paper.config import settings
from academic_paper.db import (
    db_connection,
    get_all_papers_for_scoring,
    get_paper,
    get_summary,
    list_summaries,
    save_summary,
    update_paper_score,
)
from academic_paper.errors import _http_exc_for, _safe_error_message
from academic_paper.models import PaperSummary
from academic_paper.scorer import compute_score
from academic_paper.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()


def _summary_response(paper_id: int, model: str, summary: dict, cached: bool) -> dict:
    """Build the summary response; summary fields come from PaperSummary.model_fields."""
    defaults = PaperSummary().model_dump()
    return {
        "paper_id": paper_id,
        "model": model,
        **{f: summary.get(f, defaults[f]) for f in PaperSummary.model_fields},
        "cached": cached,
    }


def get_cached_summary(paper_id: int) -> dict:
    """Return the cached summary only. GET is safe/idempotent (#140):
    crawler or monitoring access must never trigger LLM generation or DB
    writes. Generation lives in generate_summary().
    """
    try:
        with db_connection(settings.academic_db) as conn:
            paper = get_paper(conn, paper_id)
            if paper is None:
                raise HTTPException(status_code=404, detail="Paper not found")

            cached_summary = get_summary(conn, paper_id)
            if cached_summary is None:
                raise HTTPException(
                    status_code=404,
                    detail="Summary not generated yet — POST /papers/{paper_id}/summary to generate",
                )
            return _summary_response(paper_id, cached_summary["model"], cached_summary, True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get summary for paper_id=%s", paper_id)
        raise _http_exc_for(e, "Failed to get summary")


async def generate_summary(paper_id: int, force: bool, llm, summarizer) -> dict:
    """Generate the summary (cached result is returned unless force=True)."""
    # Read paper and cache in a short-lived connection — close before LLM await
    # to avoid holding a WAL write-lock for the full LLM timeout (#191).
    try:
        with db_connection(settings.academic_db) as conn:
            paper = get_paper(conn, paper_id)

            if paper is None:
                raise HTTPException(status_code=404, detail="Paper not found")

            if not force:
                cached_summary = get_summary(conn, paper_id)
                if cached_summary is not None:
                    return _summary_response(paper_id, cached_summary["model"], cached_summary, True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to check cached summary for paper_id=%s", paper_id)
        raise _http_exc_for(e, "Failed to check cached summary")

    # Connection closed — safe to await LLM for up to 300 s without blocking writers.
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    if summarizer is None:
        raise HTTPException(status_code=503, detail="Summarizer not initialized")

    try:
        with tracer.start_as_current_span("summarize") as span:
            span.set_attribute("paper_id", paper_id)
            summary = await summarizer.summarize(
                paper_id, paper["file_hash"], title=paper.get("title"), file_name=paper.get("file_name")
            )

        model = llm.display_name

        # Open a fresh short-lived connection just for the write.
        with db_connection(settings.academic_db) as conn:
            save_summary(conn, paper_id, model, summary)

        return _summary_response(paper_id, model, summary, False)

    except Exception as e:
        logger.exception("Summarization error for paper_id=%s", paper_id)
        raise _http_exc_for(e, "Summarization failed: check LLM availability")


def score_all() -> dict:
    """Compute and store relevance scores for all papers (POST /papers/score-all).

    A per-paper failure (e.g. compute_score/update_paper_score raising) does not
    abort the whole run — it is counted in `failed`/`errors` and the loop moves
    on to the next paper (#276).
    """
    preferred = settings.preferred_categories_list
    try:
        with db_connection(settings.academic_db) as conn:
            papers = get_all_papers_for_scoring(conn)
            scored = 0
            failed = 0
            errors: list[str] = []
            for paper in papers:
                try:
                    score = compute_score(paper, preferred)
                    update_paper_score(conn, paper["id"], score)
                    scored += 1
                except Exception as e:
                    logger.exception("Scoring failed for paper_id=%s", paper.get("id"))
                    failed += 1
                    errors.append(f"paper_id={paper.get('id')}: {_safe_error_message(e)}")
        return {
            "total": len(papers),
            "scored": scored,
            "failed": failed,
            "errors": errors,
            "preferred_categories": preferred,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("score_all_papers failed unexpectedly")
        raise _http_exc_for(e, "Scoring failed unexpectedly")


def score_one(paper_id: int) -> dict:
    """Compute and store relevance score for a single paper (POST /papers/{id}/score)."""
    try:
        with db_connection(settings.academic_db) as conn:
            paper = get_paper(conn, paper_id)
            if paper is None:
                raise HTTPException(status_code=404, detail="Paper not found")
            preferred = settings.preferred_categories_list
            score = compute_score(paper, preferred)
            update_paper_score(conn, paper_id, score)
        return {"paper_id": paper_id, "score": score}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to score paper_id=%s", paper_id)
        raise _http_exc_for(e, "Scoring failed")


def list_all_summaries(limit: int, offset: int) -> dict:
    """List all paper summaries with associated paper metadata (GET /summaries)."""
    try:
        with db_connection(settings.academic_db) as conn:
            total, summaries = list_summaries(conn, limit=limit, offset=offset)
        return {"total": total, "summaries": summaries}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list summaries")
        raise _http_exc_for(e, "Failed to list summaries")
