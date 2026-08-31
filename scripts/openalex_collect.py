#!/usr/bin/env python3
"""OpenAlex paper collector.

Searches OpenAlex for papers by keyword and date range, then ingests
their PDFs into the academic-paper-system API.

OpenAlex is completely free, requires no authentication, and supports
flexible date-range filtering — ideal for finding papers from 6–12 months ago.

Usage:
    python scripts/openalex_collect.py --query "retrieval augmented generation"
    python scripts/openalex_collect.py --query "multi-agent LLM" --from-date 2025-02-01 --until-date 2025-08-01
    python scripts/openalex_collect.py --query "jailbreak safety" --max 20

Exit codes:
    0 — all papers processed (new + duplicate)
    1 — one or more papers failed
"""

import argparse
import json
import re
import sys
import time

import httpx
from _collect_common import download_pdf, ingest_pdf, run_collect
from cli_utils import check_date_order, iso_date, positive_int

OPENALEX_API = "https://api.openalex.org/works"
FIELDS = ",".join(
    [
        "id",
        "title",
        "publication_date",
        "authorships",
        "topics",
        "abstract_inverted_index",
        "ids",
        "open_access",
        "primary_location",
    ]
)


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format.

    The inverted index maps each word to a list of positions:
    {"word": [pos1, pos2, ...], ...}
    We sort all (position, word) pairs and join them.
    """
    if not inverted_index:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(word for _, word in pairs)


def extract_arxiv_id(work: dict) -> str:
    """Extract arXiv ID from OpenAlex work, or empty string."""
    ids = work.get("ids") or {}
    arxiv_url = ids.get("arxiv", "")
    if arxiv_url:
        m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", arxiv_url)
        if m:
            return m.group(1)
    return ""


def get_pdf_url(work: dict) -> str | None:
    """Return the best available PDF URL for a work.

    Priority:
    1. arXiv PDF (most reliable)
    2. open_access.oa_url (may be a landing page, not a direct PDF)
    """
    arxiv_id = extract_arxiv_id(work)
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    oa = work.get("open_access") or {}
    oa_url = oa.get("oa_url") or ""
    if oa_url and oa_url.endswith(".pdf"):
        return oa_url

    return None


def fetch_papers(
    query: str,
    from_date: str = "",
    until_date: str = "",
    max_results: int = 20,
    timeout: int = 30,
) -> tuple[list[dict], str | None]:
    """Search OpenAlex and return (works with a PDF URL, fetch error or None).

    Uses cursor-based pagination. Stops when max_results is reached or
    all pages are exhausted. A page-fetch failure aborts pagination and is
    reported via the second tuple element so callers can distinguish
    "no results" from "search API down" (exit-code false success, #133).
    """
    filter_parts = [f"title.search:{query}"]
    if from_date:
        filter_parts.append(f"from_publication_date:{from_date}")
    if until_date:
        filter_parts.append(f"to_publication_date:{until_date}")

    params: dict = {
        "filter": ",".join(filter_parts),
        "select": FIELDS,
        "sort": "publication_date:desc",
        "per-page": min(200, max_results * 3),  # fetch more than needed; many lack PDFs
        "cursor": "*",
        "mailto": "noreply@example.com",  # polite pool: faster rate limits
    }

    papers: list[dict] = []
    seen: set[str] = set()
    fetch_error: str | None = None

    while len(papers) < max_results:
        try:
            resp = httpx.get(OPENALEX_API, params=params, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:
            fetch_error = str(exc) or type(exc).__name__
            print(f"[openalex] ERROR fetching page: {fetch_error}", file=sys.stderr)
            break

        data = resp.json()
        results = data.get("results") or []
        if not results:
            break

        for work in results:
            work_id = work.get("id", "")
            if not work_id or work_id in seen:
                continue
            pdf_url = get_pdf_url(work)
            if not pdf_url:
                continue
            seen.add(work_id)
            work["_pdf_url"] = pdf_url
            papers.append(work)
            if len(papers) >= max_results:
                break

        next_cursor = (data.get("meta") or {}).get("next_cursor")
        if not next_cursor or len(papers) >= max_results:
            break

        params["cursor"] = next_cursor
        time.sleep(0.5)  # OpenAlex polite pool: 10 req/s; 0.5s is comfortable

    return papers, fetch_error


def ingest_paper(
    client: httpx.Client,
    work: dict,
    api_url: str,
    pdf_timeout: int = 60,
    poll_timeout: int = 300,
) -> dict:
    """Stream the PDF from OpenAlex and submit it via the ingest API."""
    pdf_url = work["_pdf_url"]
    arxiv_id = extract_arxiv_id(work)
    work_id_short = (work.get("id") or "").split("/")[-1][:12]
    file_name = f"oa_{arxiv_id or work_id_short}.pdf"
    label = f"arXiv:{arxiv_id}" if arxiv_id else work_id_short

    title = (work.get("title") or "").replace("\n", " ").strip()
    authors = [(a.get("author") or {}).get("display_name", "") for a in (work.get("authorships") or [])]
    topics = list(
        {
            (t.get("field") or {}).get("display_name", "")
            for t in (work.get("topics") or [])
            if (t.get("field") or {}).get("display_name")
        }
    )
    pub_date = (work.get("publication_date") or "")[:10] or None

    with download_pdf(client, pdf_url, pdf_timeout) as tmp_path:
        result = ingest_pdf(
            client,
            api_url,
            file_name,
            tmp_path,
            {
                "title": title,
                "authors": json.dumps([a for a in authors if a]),
                "categories": json.dumps(topics),
                "published_date": pub_date or "",
                "source": "openalex",
            },
            poll_timeout,
        )
    work_id = (work.get("id") or "").split("/")[-1]
    return {**result, "label": label, "id": work_id, "arxiv_id": arxiv_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAlex paper collector")
    parser.add_argument(
        "--query",
        default="large language model",
        help="Title search keyword (default: 'large language model')",
    )
    parser.add_argument(
        "--from-date",
        type=iso_date,
        default="",
        help="Start date YYYY-MM-DD (inclusive). Example: 2025-02-01",
    )
    parser.add_argument(
        "--until-date",
        type=iso_date,
        default="",
        help="End date YYYY-MM-DD (inclusive). Example: 2025-08-01",
    )
    parser.add_argument(
        "--max",
        type=positive_int,
        default=10,
        dest="max_results",
        help="Max papers to ingest (default: 10)",
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
        "--summary-file",
        default=None,
        help="Write run summary JSON to this path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and list papers without ingesting",
    )
    args = parser.parse_args()
    check_date_order(parser, args.from_date, args.until_date)

    date_range = ""
    if args.from_date or args.until_date:
        date_range = f" [{args.from_date or '*'} → {args.until_date or '*'}]"
    print(f"[openalex] query='{args.query}'{date_range} max={args.max_results} api={args.api_url}")

    papers, fetch_error = fetch_papers(
        query=args.query,
        from_date=args.from_date,
        until_date=args.until_date,
        max_results=args.max_results,
    )
    print(f"[openalex] found {len(papers)} papers with PDF URLs")
    if fetch_error and not papers:
        print(f"[openalex] FATAL: search API failed with no results: {fetch_error}", file=sys.stderr)
        if args.summary_file:
            with open(args.summary_file, "w") as f:
                json.dump({"fetched": 0, "fetch_error": fetch_error}, f, indent=2)
        sys.exit(1)

    if args.dry_run:
        for p in papers:
            arxiv_id = extract_arxiv_id(p)
            label = f"arXiv:{arxiv_id}" if arxiv_id else p.get("id", "").split("/")[-1]
            print(f"  {p.get('publication_date', '?')}  {label}  {(p.get('title') or '')[:60]}")
        return

    run_collect(
        "OpenAlex",
        papers,
        lambda client, work: ingest_paper(client, work, args.api_url, poll_timeout=args.poll_timeout),
        args.summary_file,
        fetch_error=fetch_error,
    )


if __name__ == "__main__":
    main()
