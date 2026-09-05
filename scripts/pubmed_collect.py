#!/usr/bin/env python3
"""PubMed Central paper collector.

Searches PubMed Central (PMC) for open-access papers and ingests their PDFs
into the academic-paper-system API via the NCBI E-utilities API.

Usage:
    python scripts/pubmed_collect.py
    python scripts/pubmed_collect.py --terms "machine learning" --max 5
    python scripts/pubmed_collect.py --api-key YOUR_NCBI_KEY

Exit codes:
    0 — all papers processed (new + duplicate)
    1 — one or more papers failed
"""

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET

import httpx
from _collect_common import download_pdf, ingest_pdf, run_collect
from cli_utils import check_date_order, iso_date, positive_int

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_PDF_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"


def fetch_pmc_ids(
    terms: list[str],
    max_results: int,
    api_key: str = "",
    timeout: int = 30,
    from_date: str = "",
    until_date: str = "",
) -> list[str]:
    """Search PMC for open-access paper IDs."""
    query = " OR ".join(f'"{t}"' for t in terms) + " AND open access[filter]"
    params: dict = {
        "db": "pmc",
        "term": query,
        "retmax": max_results,
        "sort": "pub date",
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if from_date or until_date:
        params["datetype"] = "pdat"
        if from_date:
            params["mindate"] = from_date
        if until_date:
            params["maxdate"] = until_date

    resp = httpx.get(ESEARCH_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_paper_metadata(
    pmc_ids: list[str],
    api_key: str = "",
    timeout: int = 60,
) -> list[dict]:
    """Fetch XML metadata for a batch of PMC IDs."""
    if not pmc_ids:
        return []
    params: dict = {
        "db": "pmc",
        "id": ",".join(pmc_ids),
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    resp = httpx.get(EFETCH_URL, params=params, timeout=timeout)
    resp.raise_for_status()

    papers: list[dict] = []
    try:
        root = ET.fromstring(resp.content)
        for article in root.findall(".//article"):
            meta = _parse_article(article)
            if meta:
                papers.append(meta)
    except ET.ParseError as exc:
        print(f"[pubmed] XML parse error: {exc}", file=sys.stderr)
    return papers


def _parse_article(article) -> dict | None:
    """Extract metadata from a PMC article XML element."""
    pmc_elem = article.find(".//article-id[@pub-id-type='pmc']")
    if pmc_elem is None or not pmc_elem.text:
        return None
    pmc_id = pmc_elem.text.strip()

    title_elem = article.find(".//article-title")
    title = "".join(title_elem.itertext()) if title_elem is not None else ""
    title = title.replace("\n", " ").strip()

    authors: list[str] = []
    for contrib in article.findall(".//contrib[@contrib-type='author']"):
        surname = contrib.findtext("name/surname", "")
        given = contrib.findtext("name/given-names", "")
        name = f"{given} {surname}".strip()
        if name:
            authors.append(name)

    # Prefer epub date, fall back to any pub-date. Element truthiness is
    # deprecated (childless elements are falsy), so compare against None.
    pub_date_elem = next(
        (
            elem
            for elem in (
                article.find(".//pub-date[@pub-type='epub']"),
                article.find(".//pub-date[@date-type='pub']"),
                article.find(".//pub-date"),
            )
            if elem is not None
        ),
        None,
    )
    pub_date = None
    if pub_date_elem is not None:
        year = pub_date_elem.findtext("year", "")
        month = pub_date_elem.findtext("month", "01").zfill(2)
        day = pub_date_elem.findtext("day", "01").zfill(2)
        if year:
            pub_date = f"{year}-{month}-{day}"

    categories: list[str] = []
    for subj in article.findall(".//subject"):
        text = (subj.text or "").strip()
        if text:
            categories.append(text)

    return {
        "pmc_id": pmc_id,
        "title": title,
        "authors": authors,
        "pub_date": pub_date,
        "categories": categories,
    }


def ingest_paper(
    client: httpx.Client,
    paper: dict,
    api_url: str,
    pdf_timeout: int = 60,
    poll_timeout: int = 300,
) -> dict:
    """Stream the PDF from PMC and submit it via the ingest API."""
    pmc_id = paper["pmc_id"]
    pdf_url = PMC_PDF_URL.format(pmc_id=pmc_id)
    with download_pdf(client, pdf_url, pdf_timeout) as tmp_path:
        result = ingest_pdf(
            client,
            api_url,
            f"pmc_{pmc_id}.pdf",
            tmp_path,
            {
                "title": paper["title"],
                "authors": json.dumps(paper["authors"]),
                "categories": json.dumps(paper["categories"]),
                "published_date": paper["pub_date"] or "",
                "source": "pubmed",
            },
            poll_timeout,
        )
    return {**result, "label": f"PMC{pmc_id}", "pmc_id": pmc_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="PubMed Central paper collector")
    parser.add_argument(
        "--terms",
        nargs="+",
        default=["artificial intelligence", "machine learning"],
        help="Search terms (default: artificial intelligence, machine learning)",
    )
    parser.add_argument(
        "--max",
        type=positive_int,
        default=5,
        dest="max_results",
        help="Max papers to ingest (default: 5)",
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
        help="NCBI API key (optional; raises rate limit to 10 req/sec)",
    )
    parser.add_argument(
        "--from-date",
        type=iso_date,
        default="",
        metavar="YYYY-MM-DD",
        help="Only return papers published on or after this date (pdat filter)",
    )
    parser.add_argument(
        "--until-date",
        type=iso_date,
        default="",
        metavar="YYYY-MM-DD",
        help="Only return papers published on or before this date (pdat filter)",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Write run summary JSON to this path",
    )
    args = parser.parse_args()
    check_date_order(parser, args.from_date, args.until_date)

    print(f"[pubmed] terms={args.terms} max={args.max_results} api={args.api_url}")

    try:
        pmc_ids = fetch_pmc_ids(
            args.terms, args.max_results, args.api_key,
            from_date=args.from_date, until_date=args.until_date,
        )
    except Exception as exc:
        print(f"[pubmed] ERROR fetching PMC IDs: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[pubmed] found {len(pmc_ids)} PMC IDs")
    if not pmc_ids:
        if args.summary_file:
            with open(args.summary_file, "w") as f:
                json.dump({"fetched": 0, "ingested": 0, "duplicate": 0, "failed": 0, "detail": []}, f)
        sys.exit(0)

    time.sleep(0.34)  # respect 3 req/sec default rate limit
    try:
        papers = fetch_paper_metadata(pmc_ids, args.api_key)
    except Exception as exc:
        print(f"[pubmed] ERROR fetching metadata: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[pubmed] parsed {len(papers)} articles")

    def _ingest(client, paper):
        time.sleep(0.34)  # respect PMC rate limit (3 req/sec default)
        return ingest_paper(client, paper, args.api_url, poll_timeout=args.poll_timeout)

    run_collect("PubMed", papers, _ingest, args.summary_file)


if __name__ == "__main__":
    main()
