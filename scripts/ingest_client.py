"""Shared client helper for the async /papers/ingest flow.

`/papers/ingest` returns 202 + a job_id and processes in the background.
`submit_and_wait` posts the PDF and polls `GET /jobs/{job_id}` until the job
reaches `done` or `failed`, so collectors count ingested/failed accurately
without hitting a client-side timeout mid-processing.

A backward-compatible path is kept: if the server returns 200 with the final
result (e.g. `wait=true`), the result is returned directly without polling.
"""

import io
import os
import time

import httpx


def _auth_headers() -> dict:
    """X-API-Key header from PAPER_API_KEY, or {} when auth is not configured (#183)."""
    api_key = os.environ.get("PAPER_API_KEY", "")
    return {"X-API-Key": api_key} if api_key else {}


def submit_and_wait(
    client: httpx.Client,
    api_url: str,
    file_name: str,
    pdf_bytes: bytes,
    metadata: dict,
    *,
    submit_timeout: int = 30,
    poll_timeout: int = 300,
    poll_interval: float = 2.0,
) -> dict:
    """POST a PDF to /papers/ingest and poll the job to completion.

    Args:
        client: A reusable httpx.Client.
        api_url: academic-paper-system base URL.
        file_name: Upload filename.
        pdf_bytes: Raw PDF content.
        metadata: Form fields (title/authors/categories/published_date/source).
        submit_timeout: Timeout (s) for the POST and each poll request.
        poll_timeout: Max seconds to wait for the job to finish.
        poll_interval: Seconds between polls.

    Returns:
        dict with "status" in {"ingested", "duplicate"} plus paper_id/chunks
        when available.

    Raises:
        httpx.HTTPError: Transport error or non-2xx from POST/poll.
        RuntimeError: The ingest job reported status "failed".
        TimeoutError: The job did not finish within poll_timeout.
    """
    resp = client.post(
        f"{api_url}/papers/ingest",
        files={"file": (file_name, io.BytesIO(pdf_bytes), "application/pdf")},
        data=metadata,
        headers=_auth_headers(),
        timeout=submit_timeout,
    )
    if resp.status_code == 409:
        return {"status": "duplicate"}
    resp.raise_for_status()
    body = resp.json()

    # Synchronous server response (wait=true / legacy): final result, no job to poll.
    if "job_id" not in body:
        return {**body, "status": "ingested"}

    job_id = body["job_id"]
    paper_id = body.get("paper_id")
    deadline = time.monotonic() + poll_timeout
    while True:
        jresp = client.get(f"{api_url}/jobs/{job_id}", timeout=submit_timeout)
        jresp.raise_for_status()
        job = jresp.json()
        status = job["status"]
        if status == "done":
            result = job.get("result") or {}
            # result carries status="indexed"; spread first so our "ingested" wins.
            return {**result, "paper_id": result.get("paper_id", paper_id), "status": "ingested"}
        if status == "failed":
            errors = job.get("errors") or []
            raise RuntimeError(errors[0] if errors else "ingest job failed")
        if time.monotonic() > deadline:
            raise TimeoutError(f"ingest job {job_id} did not finish within {poll_timeout}s")
        time.sleep(poll_interval)
