"""Shared client helper for the async /papers/ingest flow.

`/papers/ingest` returns 202 + a job_id and processes in the background.
`submit_and_wait` posts the PDF and polls `GET /jobs/{job_id}` until the job
reaches `done` or `failed`, so collectors count ingested/failed accurately
without hitting a client-side timeout mid-processing.
"""

import time
from typing import BinaryIO

import httpx
from cli_utils import auth_headers


def submit_and_wait(
    client: httpx.Client,
    api_url: str,
    file_name: str,
    pdf_data: BinaryIO,
    metadata: dict,
    *,
    submit_timeout: int = 30,
    poll_timeout: int = 300,
    poll_interval: float = 2.0,
    max_poll_errors: int = 5,
) -> dict:
    """POST a PDF to /papers/ingest and poll the job to completion.

    Args:
        client: A reusable httpx.Client.
        api_url: academic-paper-system base URL.
        file_name: Upload filename.
        pdf_data: An open binary file handle (avoids the extra in-memory copy
            that ``bytes`` would require when content was already streamed to
            a temp file, #194).
        metadata: Form fields (title/authors/categories/published_date/source).
        submit_timeout: Timeout (s) for the POST and each poll request.
        poll_timeout: Max seconds to wait for the job to finish.
        poll_interval: Seconds between polls.
        max_poll_errors: Consecutive transient poll errors (transport / 5xx)
            tolerated before the last one is raised.

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
        files={"file": (file_name, pdf_data, "application/pdf")},
        data=metadata,
        headers=auth_headers(),
        timeout=submit_timeout,
    )
    if resp.status_code == 409:
        return {"status": "duplicate"}
    resp.raise_for_status()
    body = resp.json()

    job_id = body["job_id"]
    paper_id = body.get("paper_id")
    deadline = time.monotonic() + poll_timeout
    poll_errors = 0
    while True:
        try:
            jresp = client.get(f"{api_url}/jobs/{job_id}", headers=auth_headers(), timeout=submit_timeout)
            jresp.raise_for_status()
            job = jresp.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            # The server keeps ingesting regardless of a failed poll, so a transient
            # error (timeout, connection reset, 5xx) must not be counted as failed (#825).
            # Permanent errors (4xx: auth, 404) and exhausted budgets propagate as before.
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            poll_errors += 1
            if poll_errors >= max_poll_errors:
                raise
            if time.monotonic() > deadline:
                raise TimeoutError(f"ingest job {job_id} did not finish within {poll_timeout}s") from exc
            time.sleep(poll_interval)
            continue
        poll_errors = 0
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
