"""Jobs routes (summarize-all / get / list), extracted from server.py (#359 4/5, #621).

Path, status code and OpenAPI are unchanged. require_scope (#627) is applied per route. llm / summarizer are injected via academic_paper.dependencies (#360).
 ``/jobs/summarize-all`` is declared before ``/jobs/{job_id}``.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from academic_paper.auth import Scope, require_scope
from academic_paper.dependencies import get_llm, get_summarizer
from academic_paper.jobs import job_store
from academic_paper.services.summary_service import run_summarize_all

router = APIRouter()


@router.post("/jobs/summarize-all", status_code=202, dependencies=[Depends(require_scope(Scope.ADMIN))])
async def start_summarize_all(
    background_tasks: BackgroundTasks, llm=Depends(get_llm), summarizer=Depends(get_summarizer)
):
    """Start a background job to summarize all papers without a cached summary."""
    if llm is None or summarizer is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    job = await job_store.create_if_not_running(kind="summarize-all")
    if job is None:
        raise HTTPException(status_code=409, detail="A summarize-all job is already running")
    background_tasks.add_task(run_summarize_all, job.id, summarizer, llm, job_store)
    return {"job_id": job.id, "status": job.status}


@router.get("/jobs/{job_id}", dependencies=[Depends(require_scope(Scope.READ))])
async def get_job_endpoint(job_id: str):
    """Get status of a background job by ID (evicted finished jobs are read from SQLite, #774)."""
    job = await job_store.get_or_load(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/jobs", dependencies=[Depends(require_scope(Scope.READ))])
def list_jobs_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List background jobs, newest first, with pagination (#190 auth, #752 paging).

    Covers in-memory jobs only; finished jobs evicted to SQLite are out of scope (fetch via GET /jobs/{id}).
    """
    # reversed() first so jobs with equal started_at keep newest-created first (sorted is stable).
    jobs = sorted(reversed(job_store.list_all()), key=lambda j: j.started_at, reverse=True)
    return {"total": len(jobs), "jobs": [j.to_dict() for j in jobs[offset : offset + limit]]}
