"""Shared helpers for paper collector scripts.

Provides:
  - download_pdf   : stream a PDF URL into a temp file (avoid full-content memory)
  - ingest_pdf     : call submit_and_wait and surface server detail on HTTP error
  - run_collect    : canonical ingest loop — counts, summary print, JSON write, exit
"""

import contextlib
import json
import sys
import tempfile
from pathlib import Path

import httpx
from ingest_client import submit_and_wait


@contextlib.contextmanager
def download_pdf(client: httpx.Client, url: str, timeout: int = 60):
    """Stream a PDF from *url* using *client* into a named temp file.

    Yields the temp-file path.  The file is deleted on exit.
    Raises ValueError when the response content-type does not contain "pdf".
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        with client.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "").lower()
            if "pdf" not in ct and not url.lower().endswith(".pdf"):
                raise ValueError(f"Not a PDF (content-type: {ct})")
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    fh.write(chunk)
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def ingest_pdf(
    client: httpx.Client,
    api_url: str,
    file_name: str,
    tmp_path: str,
    metadata: dict,
    poll_timeout: int = 300,
) -> dict:
    """Read *tmp_path* and submit it via submit_and_wait.

    On HTTP error, extracts ``{"detail": ...}`` from the response body so the
    error message shown in cron logs is the server reason, not just the status.
    """
    pdf_bytes = Path(tmp_path).read_bytes()
    try:
        return submit_and_wait(client, api_url, file_name, pdf_bytes, metadata, poll_timeout=poll_timeout)
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text[:200])
        except Exception:
            detail = exc.response.text[:200]
        raise RuntimeError(f"HTTP {exc.response.status_code}: {detail}") from exc


def run_collect(
    source_label: str,
    papers: list[dict],
    ingest_fn,
    summary_file: str | None,
    *,
    fetch_error: str | None = None,
) -> None:
    """Run the ingest loop for *papers* and handle summary / exit-code.

    ``ingest_fn(client, paper) -> dict`` must return a dict containing at
    minimum ``"status"`` (``"ingested"`` or ``"duplicate"``) and ``"label"``
    (string used for per-paper log output).  Any additional keys are stored
    verbatim in the detail list inside the summary JSON.

    Exits with code 1 when any paper failed.
    """
    counts: dict[str, int] = {"ingested": 0, "duplicate": 0, "failed": 0}
    detail: list[dict] = []

    with httpx.Client() as client:
        for paper in papers:
            try:
                result = ingest_fn(client, paper)
                label = result.pop("label", "?")
                status = result["status"]
                counts[status] = counts.get(status, 0) + 1
                tag = "OK  " if status == "ingested" else "SKIP"
                title = (paper.get("title") or "")[:50] if isinstance(paper, dict) else ""
                print(f"  {tag} [{label}] {title}")
                detail.append({"label": label, "status": status, **result})
            except Exception as exc:
                label = _paper_label(paper)
                counts["failed"] += 1
                print(f"  ERR  [{label}] {exc}", file=sys.stderr)
                detail.append({"label": label, "status": "failed", "error": str(exc)})

    tag_lower = source_label.lower().replace(" ", "-")
    print(f"\n## {source_label} Summary")
    print(f"- Found    : {len(papers)}")
    print(f"- Ingested : {counts['ingested']}")
    print(f"- Duplicate: {counts['duplicate']}")
    print(f"- Failed   : {counts['failed']}")

    if summary_file:
        payload: dict = {**counts, "fetched": len(papers), "detail": detail}
        if fetch_error is not None:
            payload["fetch_error"] = fetch_error
        with open(summary_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[{tag_lower}] summary written to {summary_file}")

    if counts["failed"] > 0:
        sys.exit(1)


def _paper_label(paper: dict) -> str:
    """Best-effort one-line identifier for a paper dict, used in error logs."""
    for key in ("arxiv_id", "paperId", "pmc_id", "id"):
        val = paper.get(key)
        if val:
            return str(val)[:16]
    return "?"
