#!/usr/bin/env python3
"""Semantic Scholar paper collector.

Searches Semantic Scholar for open-access papers by keyword and optional
date range, then ingests their PDFs into the academic-paper-system API.

Usage:
    python scripts/semantic_scholar_collect.py
    python scripts/semantic_scholar_collect.py --query "RAG nugget retrieval" --max 10
    python scripts/semantic_scholar_collect.py --query "jailbreak" --from-date 2025-08-01 --until-date 2026-02-28
    python scripts/semantic_scholar_collect.py --api-key YOUR_KEY

API key:
    Without a key, rate limit is ~1 req/s (429 likely under heavy use).
    Free key via https://www.semanticscholar.org/product/api raises it to 100 req/s.
    Pass via --api-key or env var SEMANTIC_SCHOLAR_API_KEY.

Exit codes:
    0 — all papers processed (new + duplicate)
    1 — one or more papers failed
"""

import argparse
import io
import json
import os
import sys
import time

import httpx

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,authors,year,publicationDate,openAccessPdf,fieldsOfStudy"


def fetch_papers(
    query: str,
    from_date: str = "",
    until_date: str = "",
    max_results: int = 10,
    api_key: str = "",
    timeout: int = 30,
) -> list[dict]:
    """Search Semantic Scholar for open-access papers.

    Uses the `publicationDateOrYear` filter (format: YYYY-MM-DD:YYYY-MM-DD)
    when date range is provided. Only returns papers with an openAccessPdf.
    """
    headers = {"x-api-key": api_key} if api_key else {}
    papers: list[dict] = []
    seen: set[str] = set()
    offset = 0
    limit = min(100, max_results * 3)  # fetch more since many lack open-access PDFs

    params: dict = {
        "query": query,
        "fields": S2_FIELDS,
        "limit": limit,
        "offset": offset,
    }
    if from_date or until_date:
        lo = from_date or "2000-01-01"
        hi = until_date or "2099-12-31"
        params["publicationDateOrYear"] = f"{lo}:{hi}"

    while len(papers) < max_results:
        params["offset"] = offset
        try:
            resp = httpx.get(S2_API, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", "15"))
                print(f"[s2] rate limited, sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                resp = httpx.get(S2_API, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:
            print(f"[s2] ERROR fetching offset={offset}: {exc}", file=sys.stderr)
            break

        data = resp.json()
        batch = data.get("data") or []
        if not batch:
            break

        for paper in batch:
            pid = paper.get("paperId", "")
            if not pid or pid in seen:
                continue
            if not paper.get("openAccessPdf"):
                continue
            seen.add(pid)
            papers.append(paper)
            if len(papers) >= max_results:
                break

        total = data.get("total", 0)
        offset += len(batch)
        if offset >= total or len(batch) < limit:
            break

        time.sleep(1 if not api_key else 0.05)

    return papers


def ingest_paper(
    client: httpx.Client,
    paper: dict,
    api_url: str,
    pdf_timeout: int = 60,
) -> dict:
    """Download PDF and POST to /papers/ingest."""
    pdf_url = paper["openAccessPdf"]["url"]
    pdf_resp = httpx.get(pdf_url, timeout=pdf_timeout, follow_redirects=True)
    pdf_resp.raise_for_status()
    if "pdf" not in pdf_resp.headers.get("content-type", "").lower():
        raise ValueError(f"Not a PDF (content-type: {pdf_resp.headers.get('content-type')})")

    s2_id = paper["paperId"]
    title = (paper.get("title") or "").replace("\n", " ").strip()
    authors = [a.get("name", "") for a in (paper.get("authors") or [])]
    categories = paper.get("fieldsOfStudy") or []
    pub_date = (paper.get("publicationDate") or "")[:10] or None

    resp = client.post(
        f"{api_url}/papers/ingest",
        files={"file": (f"s2_{s2_id[:12]}.pdf", io.BytesIO(pdf_resp.content), "application/pdf")},
        data={
            "title": title,
            "authors": json.dumps(authors),
            "categories": json.dumps(categories),
            "published_date": pub_date or "",
            "source": "semantic_scholar",
        },
        timeout=30,
    )
    if resp.status_code == 409:
        return {"status": "duplicate"}
    resp.raise_for_status()
    data = resp.json()
    data["status"] = "ingested"
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Scholar paper collector")
    parser.add_argument(
        "--query",
        default="machine learning",
        help="Search keyword (default: 'machine learning')",
    )
    # Legacy alias kept for backward compatibility
    parser.add_argument(
        "--topics",
        nargs="+",
        dest="topics",
        default=None,
        help="[deprecated] Use --query instead. If given, joined as single query.",
    )
    parser.add_argument(
        "--from-date",
        default="",
        metavar="YYYY-MM-DD",
        help="Filter papers published on or after this date (inclusive)",
    )
    parser.add_argument(
        "--until-date",
        default="",
        metavar="YYYY-MM-DD",
        help="Filter papers published on or before this date (inclusive)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=10,
        dest="max_results",
        help="Max open-access papers to ingest (default: 10)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8020",
        help="academic-paper-system API base URL",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Semantic Scholar API key (or set SEMANTIC_SCHOLAR_API_KEY env var)",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Write run summary JSON to this path",
    )
    args = parser.parse_args()

    # --topics is deprecated; merge into --query if provided
    if args.topics:
        query = " ".join(args.topics)
        print("[s2] WARNING: --topics is deprecated; use --query", file=sys.stderr)
    else:
        query = args.query

    api_key = args.api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    date_range = ""
    if args.from_date or args.until_date:
        date_range = f" [{args.from_date or '*'} → {args.until_date or '*'}]"
    print(f"[s2] query='{query}'{date_range} max={args.max_results} api={args.api_url}")
    if api_key:
        print("[s2] using API key")

    papers = fetch_papers(
        query=query,
        from_date=args.from_date,
        until_date=args.until_date,
        max_results=args.max_results,
        api_key=api_key,
    )
    print(f"[s2] found {len(papers)} open-access papers")

    counts = {"ingested": 0, "duplicate": 0, "failed": 0}
    detail: list[dict] = []

    with httpx.Client() as client:
        for paper in papers:
            s2_id = paper["paperId"]
            try:
                result = ingest_paper(client, paper, args.api_url)
                status = result["status"]
                counts[status] = counts.get(status, 0) + 1
                label = "OK  " if status == "ingested" else "SKIP"
                print(f"  {label} [{s2_id[:8]}] {(paper.get('title') or '')[:50]}")
                detail.append({"s2_id": s2_id, "status": status})
            except Exception as exc:
                counts["failed"] += 1
                print(f"  ERR  [{s2_id[:8]}] {exc}", file=sys.stderr)
                detail.append({"s2_id": s2_id, "status": "failed", "error": str(exc)})

    print("\n## Semantic Scholar Summary")
    print(f"- Found    : {len(papers)}")
    print(f"- Ingested : {counts['ingested']}")
    print(f"- Duplicate: {counts['duplicate']}")
    print(f"- Failed   : {counts['failed']}")

    if args.summary_file:
        with open(args.summary_file, "w") as f:
            json.dump({**counts, "fetched": len(papers), "detail": detail}, f, indent=2)
        print(f"[s2] summary written to {args.summary_file}")

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
