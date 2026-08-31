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
import json
import os
import sys
import time

import httpx
from _collect_common import download_pdf, ingest_pdf, run_collect
from cli_utils import check_date_order, iso_date, positive_int

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,authors,year,publicationDate,openAccessPdf,fieldsOfStudy"


def fetch_papers(
    query: str,
    from_date: str = "",
    until_date: str = "",
    max_results: int = 10,
    api_key: str = "",
    timeout: int = 30,
) -> tuple[list[dict], str | None]:
    """Search Semantic Scholar for open-access papers.

    Uses the `publicationDateOrYear` filter (format: YYYY-MM-DD:YYYY-MM-DD)
    when date range is provided. Only returns papers with an openAccessPdf.
    Returns (papers, fetch error or None); a page-fetch failure aborts
    pagination and is reported so callers can distinguish "no results" from
    "search API down" (exit-code false success, #133).
    """
    headers = {"x-api-key": api_key} if api_key else {}
    papers: list[dict] = []
    seen: set[str] = set()
    fetch_error: str | None = None
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
            fetch_error = str(exc) or type(exc).__name__
            print(f"[s2] ERROR fetching offset={offset}: {fetch_error}", file=sys.stderr)
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

    return papers, fetch_error


def ingest_paper(
    client: httpx.Client,
    paper: dict,
    api_url: str,
    pdf_timeout: int = 60,
    poll_timeout: int = 300,
) -> dict:
    """Stream the PDF from Semantic Scholar and submit it via the ingest API."""
    pdf_url = paper["openAccessPdf"]["url"]
    s2_id = paper["paperId"]
    title = (paper.get("title") or "").replace("\n", " ").strip()
    authors = [a.get("name", "") for a in (paper.get("authors") or [])]
    categories = paper.get("fieldsOfStudy") or []
    pub_date = (paper.get("publicationDate") or "")[:10] or None

    with download_pdf(client, pdf_url, pdf_timeout) as tmp_path:
        result = ingest_pdf(
            client,
            api_url,
            f"s2_{s2_id[:12]}.pdf",
            tmp_path,
            {
                "title": title,
                "authors": json.dumps(authors),
                "categories": json.dumps(categories),
                "published_date": pub_date or "",
                "source": "semantic_scholar",
            },
            poll_timeout,
        )
    return {**result, "label": s2_id[:8], "s2_id": s2_id}


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
        type=iso_date,
        default="",
        metavar="YYYY-MM-DD",
        help="Filter papers published on or after this date (inclusive)",
    )
    parser.add_argument(
        "--until-date",
        type=iso_date,
        default="",
        metavar="YYYY-MM-DD",
        help="Filter papers published on or before this date (inclusive)",
    )
    parser.add_argument(
        "--max",
        type=positive_int,
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
        "--poll-timeout",
        type=int,
        default=300,
        metavar="SEC",
        help="Max seconds to wait for each ingest job to finish (default: 300)",
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
    check_date_order(parser, args.from_date, args.until_date)

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

    papers, fetch_error = fetch_papers(
        query=query,
        from_date=args.from_date,
        until_date=args.until_date,
        max_results=args.max_results,
        api_key=api_key,
    )
    print(f"[s2] found {len(papers)} open-access papers")
    if fetch_error and not papers:
        print(f"[s2] FATAL: search API failed with no results: {fetch_error}", file=sys.stderr)
        if args.summary_file:
            with open(args.summary_file, "w") as f:
                json.dump({"fetched": 0, "fetch_error": fetch_error}, f, indent=2)
        sys.exit(1)

    run_collect(
        "Semantic Scholar",
        papers,
        lambda client, paper: ingest_paper(client, paper, args.api_url, poll_timeout=args.poll_timeout),
        args.summary_file,
        fetch_error=fetch_error,
    )


if __name__ == "__main__":
    main()
