#!/usr/bin/env python3
"""Semantic Scholar paper collector.

Searches Semantic Scholar for recent open-access papers by topic and ingests
their PDFs into the academic-paper-system API.

Usage:
    python scripts/semantic_scholar_collect.py
    python scripts/semantic_scholar_collect.py --topics "machine learning" --max 10
    python scripts/semantic_scholar_collect.py --api-key YOUR_KEY

Exit codes:
    0 — all papers processed (new + duplicate)
    1 — one or more papers failed
"""

import argparse
import io
import json
import sys
import time

import httpx

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,authors,year,publicationDate,openAccessPdf,fieldsOfStudy"


def fetch_papers(
    topics: list[str],
    max_results: int,
    api_key: str = "",
    timeout: int = 30,
) -> list[dict]:
    """Search Semantic Scholar for open-access papers on given topics."""
    headers = {"x-api-key": api_key} if api_key else {}
    papers: list[dict] = []
    seen: set[str] = set()

    for topic in topics:
        if len(papers) >= max_results:
            break
        params = {"query": topic, "fields": S2_FIELDS, "limit": 100}
        try:
            resp = httpx.get(S2_API, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                print(f"[s2] rate limited on '{topic}', sleeping 15s", file=sys.stderr)
                time.sleep(15)
                resp = httpx.get(S2_API, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:
            print(f"[s2] ERROR fetching '{topic}': {exc}", file=sys.stderr)
            continue

        for paper in resp.json().get("data", []):
            pid = paper.get("paperId", "")
            if not pid or pid in seen:
                continue
            if not paper.get("openAccessPdf"):
                continue
            seen.add(pid)
            papers.append(paper)
            if len(papers) >= max_results:
                break

        time.sleep(1)  # 1 req/sec to stay within rate limits

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
        "--topics", nargs="+",
        default=["machine learning", "deep learning", "natural language processing"],
        help="Search topics (default: machine learning, deep learning, NLP)",
    )
    parser.add_argument(
        "--max", type=int, default=10, dest="max_results",
        help="Max open-access papers to ingest (default: 10)",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8020",
        help="academic-paper-system API base URL",
    )
    parser.add_argument(
        "--api-key", default="",
        help="Semantic Scholar API key (optional; raises rate limit)",
    )
    parser.add_argument(
        "--summary-file", default=None,
        help="Write run summary JSON to this path",
    )
    args = parser.parse_args()

    print(f"[s2] topics={args.topics} max={args.max_results} api={args.api_url}")

    papers = fetch_papers(args.topics, args.max_results, args.api_key)
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
