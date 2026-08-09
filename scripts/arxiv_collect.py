#!/usr/bin/env python3
"""arXiv daily paper collector.

Fetches recent papers from arXiv Atom API by category and ingests them
into the academic-paper-system API via POST /papers/ingest.

Usage:
    python scripts/arxiv_collect.py
    python scripts/arxiv_collect.py --categories cs.AI cs.LG --max 30
    python scripts/arxiv_collect.py --api-url http://localhost:8020

Exit codes:
    0 — all papers processed (new + duplicate)
    1 — one or more papers failed
"""

import argparse
import io
import json
import sys
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"


def fetch_recent_papers(
    categories: list[str],
    max_results: int,
    timeout: int = 30,
) -> list[dict]:
    """Fetch recent papers from arXiv Atom API.

    Args:
        categories: List of arXiv category codes (e.g. ["cs.AI", "cs.LG"]).
        max_results: Maximum number of papers to return.
        timeout: HTTP timeout in seconds.

    Returns:
        List of dicts with arxiv_id, title, pdf_url, file_name.
    """
    query = " OR ".join(f"cat:{c}" for c in categories)
    url = (
        f"{ARXIV_API}"
        f"?search_query={quote(query)}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={max_results}"
    )
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        id_elem = entry.find("atom:id", ns)
        title_elem = entry.find("atom:title", ns)
        if id_elem is None or title_elem is None:
            continue
        arxiv_id = id_elem.text.strip().split("/abs/")[-1]
        title = title_elem.text.strip().replace("\n", " ")
        safe_id = arxiv_id.replace("/", "_")
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "pdf_url": f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf",
            "file_name": f"arxiv_{safe_id}.pdf",
        })
    return papers


def ingest_paper(
    client: httpx.Client,
    paper: dict,
    api_url: str,
    pdf_timeout: int = 60,
) -> dict:
    """Download PDF and POST to /papers/ingest.

    Args:
        client: Shared httpx.Client for ingest requests.
        paper: Paper dict with pdf_url and file_name.
        api_url: Base URL of the academic-paper-system API.
        pdf_timeout: Timeout for PDF download in seconds.

    Returns:
        Dict with status key: "ingested", "duplicate", or raises on error.
    """
    pdf_resp = httpx.get(paper["pdf_url"], timeout=pdf_timeout, follow_redirects=True)
    pdf_resp.raise_for_status()

    resp = client.post(
        f"{api_url}/papers/ingest",
        files={"file": (paper["file_name"], io.BytesIO(pdf_resp.content), "application/pdf")},
        timeout=30,
    )
    if resp.status_code == 409:
        return {"status": "duplicate"}
    resp.raise_for_status()
    data = resp.json()
    data["status"] = "ingested"
    return data


def build_summary(results: dict, papers: list[dict]) -> str:
    """Build a human-readable and machine-parseable run summary."""
    lines = [
        "## arXiv Daily Collect Summary",
        f"- Fetched : {len(papers)}",
        f"- Ingested: {results['ingested']}",
        f"- Skipped (duplicate): {results['duplicate']}",
        f"- Failed  : {results['failed']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv daily paper collector")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cs.AI", "cs.LG", "cs.CL"],
        metavar="CAT",
        help="arXiv category codes to collect (default: cs.AI cs.LG cs.CL)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        dest="max_results",
        metavar="N",
        help="Maximum papers to fetch per run (default: 20)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8020",
        help="academic-paper-system API base URL (default: http://localhost:8020)",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Write run summary JSON to this path (optional)",
    )
    args = parser.parse_args()

    print(f"[arxiv-collect] categories={args.categories} max={args.max_results} api={args.api_url}")

    try:
        papers = fetch_recent_papers(args.categories, args.max_results)
    except Exception as exc:
        print(f"[arxiv-collect] ERROR fetching arXiv: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[arxiv-collect] fetched {len(papers)} papers")

    counts = {"ingested": 0, "duplicate": 0, "failed": 0}
    detail: list[dict] = []

    with httpx.Client() as client:
        for paper in papers:
            arxiv_id = paper["arxiv_id"]
            try:
                result = ingest_paper(client, paper, args.api_url)
                status = result["status"]
                counts[status] = counts.get(status, 0) + 1
                label = "OK  " if status == "ingested" else "SKIP"
                pid = result.get("paper_id", "-")
                title_short = paper["title"][:70]
                print(f"  {label} [{arxiv_id}] paper_id={pid} {title_short}")
                detail.append({"arxiv_id": arxiv_id, "status": status, "paper_id": pid})
            except Exception as exc:
                counts["failed"] += 1
                print(f"  ERR  [{arxiv_id}] {exc}", file=sys.stderr)
                detail.append({"arxiv_id": arxiv_id, "status": "failed", "error": str(exc)})

    summary = build_summary(counts, papers)
    print(f"\n{summary}")

    if args.summary_file:
        payload = {**counts, "fetched": len(papers), "detail": detail}
        with open(args.summary_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[arxiv-collect] summary written to {args.summary_file}")

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
