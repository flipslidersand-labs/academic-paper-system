"""Jobs routes (summarize-all / get / list), extracted from server.py (#359 4/5, #621).

Path, status code and OpenAPI are unchanged. verify_api_key is applied at the router
level. llm / summarizer are read from ``request.app.state`` so this module does not
import server.py. ``/jobs/summarize-all`` is declared before ``/jobs/{job_id}``.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from academic_paper.auth import verify_api_key
from academic_paper.jobs import job_store
from academic_paper.services.summary_service import run_summarize_all

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/jobs/summarize-all", status_code=202)
async def start_summarize_all(request: Request, background_tasks: BackgroundTasks):
    """Start a background job to summarize all papers without a cached summary."""
    state = request.app.state
    if state.llm is None or state.summarizer is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    job = await job_store.create_if_not_running(kind="summarize-all")
    if job is None:
        raise HTTPException(status_code=409, detail="A summarize-all job is already running")
    background_tasks.add_task(run_summarize_all, job.id, state.summarizer, state.llm, job_store)
    return {"job_id": job.id, "status": job.status}


@router.get("/jobs/{job_id}")
def get_job_endpoint(job_id: str):
    """Get status of a background job by ID."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/jobs")
def list_jobs_endpoint():
    """List all background jobs (#190: require auth — exposes all job IDs / paper metadata)."""
    return {"jobs": [j.to_dict() for j in job_store.list_all()]}
