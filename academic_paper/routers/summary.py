"""Summary / score routes, extracted from server.py (#359 5/5, #622).

Path, response and OpenAPI are unchanged. require_scope (#627) is applied per route. llm / summarizer are read from ``request.app.state`` so this module does not
import server.py. Declaration order is preserved: ``/papers/score-all`` stays before
the ``/papers/{paper_id}/...`` routes.
"""

import asyncio

from fastapi import APIRouter, Depends, Query, Request

from academic_paper.auth import Scope, require_scope
from academic_paper.services.summary_service import (
    generate_summary,
    get_cached_summary,
    list_all_summaries,
    score_all,
    score_one,
)

router = APIRouter()


@router.get("/papers/{paper_id}/summary", dependencies=[Depends(require_scope(Scope.READ))])
async def get_summary_endpoint(paper_id: int):
    """Return the cached summary only (GET is safe/idempotent, #140). Generation lives in POST."""
    return await asyncio.to_thread(get_cached_summary, paper_id)


@router.post("/papers/{paper_id}/summary", dependencies=[Depends(require_scope(Scope.ADMIN))])
async def generate_summary_endpoint(request: Request, paper_id: int, force: bool = Query(False)):
    """Generate the summary (cached result is returned unless force=true)."""
    state = request.app.state
    return await generate_summary(paper_id, force, state.llm, state.summarizer)


@router.post("/papers/score-all", dependencies=[Depends(require_scope(Scope.ADMIN))])
def score_all_papers():
    """Compute and store relevance scores for all papers.

    Score = freshness (30-day half-life, 0–0.5) + category match (0–0.5).
    Preferred categories are configured via PREFERRED_CATEGORIES env var.
    A per-paper failure is counted in `failed`/`errors` and does not abort the run (#276).

    Returns:
        JSON with total, scored, failed counts and per-paper errors.
    """
    return score_all()


@router.post("/papers/{paper_id}/score", dependencies=[Depends(require_scope(Scope.ADMIN))])
def score_paper(paper_id: int):
    """Compute and store relevance score for a single paper."""
    return score_one(paper_id)


@router.get("/summaries", dependencies=[Depends(require_scope(Scope.READ))])
def list_summaries_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all paper summaries with associated paper metadata."""
    return list_all_summaries(limit, offset)
