#!/usr/bin/env python3
"""arXiv daily paper collector.

Fetches recent papers from arXiv Atom API by category and ingests them
into the academic-paper-system API via POST /papers/ingest.
Authors, categories, and publication dates are parsed from the feed and
saved as structured metadata.

Usage:
    python scripts/arxiv_collect.py
    python scripts/arxiv_collect.py --categories cs.AI cs.LG --max 30
    python scripts/arxiv_collect.py --api-url http://localhost:8020
    python scripts/arxiv_collect.py --from-date 2025-02-01 --until-date 2025-08-01

Date filtering notes:
    arXiv's submittedDate filter can be unreliable under rate limits.
    This script adds the date range to the query AND post-filters results
    by published_date as a fallback. When a date range is given, --max is
    the target *after* filtering; the raw fetch is multiplied by FETCH_FACTOR
    (default 5) to compensate for filtered-out entries.

Exit codes:
    0 — all papers processed (new + duplicate)
    1 — one or more papers failed
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx
from cli_utils import check_date_order, iso_date, positive_int
from ingest_client import submit_and_wait

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
FETCH_FACTOR = 5  # multiply max_results when date-filtering to fill the target


def _date_to_arxiv(date_str: str, end_of_day: bool = False) -> str:
    """Convert YYYY-MM-DD to arXiv submittedDate format YYYYMMDDHHII."""
    d = date_str.replace("-", "")
    suffix = "2359" if end_of_day else "0000"
    return d + suffix


def fetch_papers(
    categories: list[str],
    max_results: int,
    from_date: str = "",
    until_date: str = "",
    timeout: int = 30,
) -> list[dict]:
    """Fetch papers from arXiv Atom API.

    When from_date / until_date are provided:
    - Adds submittedDate filter to the arXiv query (best-effort).
    - Fetches max_results * FETCH_FACTOR raw entries.
    - Post-filters by published_date and returns up to max_results papers.

    Returns dicts with: arxiv_id, title, authors, categories,
    published_date, pdf_url, file_name.
    """
    cat_query = " OR ".join(f"cat:{c}" for c in categories)

    if from_date or until_date:
        lo = _date_to_arxiv(from_date) if from_date else "000000000000"
        hi = _date_to_arxiv(until_date, end_of_day=True) if until_date else "999999992359"
        date_filter = f"submittedDate:[{lo} TO {hi}]"
        search_query = f"({cat_query}) AND {date_filter}"
        fetch_max = max_results * FETCH_FACTOR
    else:
        search_query = cat_query
        fetch_max = max_results

    url = (
        f"{ARXIV_API}"
        f"?search_query={quote(search_query)}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={fetch_max}"
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

        authors = []
        for author_elem in entry.findall("atom:author", ns):
            name_elem = author_elem.find("atom:name", ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        cats = []
        for cat_elem in entry.findall("atom:category", ns):
            term = cat_elem.get("term", "")
            if term:
                cats.append(term)

        published_elem = entry.find("atom:published", ns)
        published_date = None
        if published_elem is not None and published_elem.text:
            published_date = published_elem.text.strip()[:10]

        # Post-filter by date range (fallback when API-side filter is unreliable)
        if published_date:
            if from_date and published_date < from_date:
                continue
            if until_date and published_date > until_date:
                continue

        safe_id = arxiv_id.replace("/", "_")
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "categories": cats,
                "published_date": published_date,
                "pdf_url": f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf",
                "file_name": f"arxiv_{safe_id}.pdf",
            }
        )

        if len(papers) >= max_results:
            break

    return papers


def ingest_paper(
    client: httpx.Client,
    paper: dict,
    api_url: str,
    pdf_timeout: int = 60,
    poll_timeout: int = 300,
) -> dict:
    """Download the PDF and submit it to the async /papers/ingest job."""
    pdf_resp = httpx.get(paper["pdf_url"], timeout=pdf_timeout, follow_redirects=True)
    pdf_resp.raise_for_status()
    if "pdf" not in pdf_resp.headers.get("content-type", "").lower():
        raise ValueError(f"Not a PDF (content-type: {pdf_resp.headers.get('content-type')})")

    return submit_and_wait(
        client,
        api_url,
        paper["file_name"],
        pdf_resp.content,
        {
            "title": paper["title"],
            "authors": json.dumps(paper["authors"]),
            "categories": json.dumps(paper["categories"]),
            "published_date": paper["published_date"] or "",
            "source": "arxiv",
        },
        poll_timeout=poll_timeout,
    )


def build_summary(counts: dict, papers: list[dict]) -> str:
    """Build a human-readable run summary."""
    lines = [
        "## arXiv Daily Collect Summary",
        f"- Fetched : {len(papers)}",
        f"- Ingested: {counts['ingested']}",
        f"- Skipped (duplicate): {counts['duplicate']}",
        f"- Failed  : {counts['failed']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv daily paper collector")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cs.AI", "cs.LG", "cs.CL"],
        metavar="CAT",
        help="arXiv category codes (default: cs.AI cs.LG cs.CL)",
    )
    parser.add_argument(
        "--max",
        type=positive_int,
        default=20,
        dest="max_results",
        metavar="N",
        help="Maximum papers to fetch per run (default: 20)",
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
        "--api-url",
        default="http://localhost:8020",
        help="academic-paper-system API base URL (default: http://localhost:8020)",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=300,
        metavar="SEC",
        help="Max seconds to wait for each ingest job to finish (default: 300)",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Write run summary JSON to this path (optional)",
    )
    args = parser.parse_args()
    check_date_order(parser, args.from_date, args.until_date)

    date_range = ""
    if args.from_date or args.until_date:
        date_range = f" [{args.from_date or '*'} → {args.until_date or '*'}]"
    print(f"[arxiv-collect] categories={args.categories} max={args.max_results}{date_range} api={args.api_url}")

    try:
        papers = fetch_papers(
            args.categories,
            args.max_results,
            from_date=args.from_date,
            until_date=args.until_date,
        )
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
                result = ingest_paper(client, paper, args.api_url, poll_timeout=args.poll_timeout)
                status = result["status"]
                counts[status] = counts.get(status, 0) + 1
                label = "OK  " if status == "ingested" else "SKIP"
                pid = result.get("paper_id", "-")
                authors_short = ", ".join(paper["authors"][:2])
                if len(paper["authors"]) > 2:
                    authors_short += " et al."
                print(f"  {label} [{arxiv_id}] id={pid} | {paper['title'][:50]} | {authors_short}")
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
