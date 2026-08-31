"""Unit tests for collector scripts' fetch/parse logic (#151).

These cover the code paths the nightly cron exercises before hitting the
ingest API: search-API response parsing, filtering, and pure formatting
helpers. HTTP calls are mocked with respx.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import respx

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import generate_portfolio  # noqa: E402
import openalex_collect  # noqa: E402
import pubmed_collect  # noqa: E402
import semantic_scholar_collect  # noqa: E402
from arxiv_collect import fetch_papers as arxiv_fetch_papers  # noqa: E402

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2410.10071v1</id>
    <title>Content Caching-Assisted Vehicular Edge Computing</title>
    <author><name>Jinjin Shen</name></author>
    <author><name>Yan Lin</name></author>
    <category term="cs.MA"/>
    <published>2024-10-14T00:00:00Z</published>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00001v2</id>
    <title>Old Paper</title>
    <author><name>Someone</name></author>
    <category term="cs.AI"/>
    <published>2023-01-01T00:00:00Z</published>
  </entry>
</feed>
"""


@respx.mock
def test_arxiv_fetch_papers_parses_atom():
    respx.get(url__startswith="https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=ARXIV_ATOM)
    )
    papers = arxiv_fetch_papers(["cs.AI"], max_results=10)

    assert len(papers) == 2
    p = papers[0]
    assert p["arxiv_id"] == "2410.10071v1"
    assert p["title"] == "Content Caching-Assisted Vehicular Edge Computing"
    assert p["authors"] == ["Jinjin Shen", "Yan Lin"]
    assert p["categories"] == ["cs.MA"]
    assert p["published_date"] == "2024-10-14"
    assert p["file_name"] == "arxiv_2410.10071v1.pdf"
    assert p["pdf_url"].endswith("2410.10071v1.pdf")


@respx.mock
def test_arxiv_fetch_papers_date_post_filter():
    respx.get(url__startswith="https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=ARXIV_ATOM)
    )
    papers = arxiv_fetch_papers(["cs.AI"], max_results=10, from_date="2024-01-01")

    assert [p["arxiv_id"] for p in papers] == ["2410.10071v1"]


def _s2_paper(pid: str, with_pdf: bool = True) -> dict:
    return {
        "paperId": pid,
        "title": f"Paper {pid}",
        "authors": [{"name": "A"}],
        "publicationDate": "2026-01-01",
        "fieldsOfStudy": ["Computer Science"],
        "openAccessPdf": {"url": f"https://example.org/{pid}.pdf"} if with_pdf else None,
    }


@respx.mock
def test_s2_fetch_papers_filters_and_dedupes():
    respx.get(url__startswith=semantic_scholar_collect.S2_API).mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 4,
                "data": [_s2_paper("aaa"), _s2_paper("aaa"), _s2_paper("bbb", with_pdf=False), _s2_paper("ccc")],
            },
        )
    )
    papers, fetch_error = semantic_scholar_collect.fetch_papers("query", max_results=10)

    assert fetch_error is None
    assert [p["paperId"] for p in papers] == ["aaa", "ccc"]  # dedup + no-PDF filtered


@respx.mock
def test_s2_fetch_papers_reports_fetch_error():
    respx.get(url__startswith=semantic_scholar_collect.S2_API).mock(return_value=httpx.Response(500, text="boom"))
    papers, fetch_error = semantic_scholar_collect.fetch_papers("query", max_results=10)

    assert papers == []
    assert fetch_error is not None


@respx.mock
def test_pubmed_fetch_pmc_ids():
    respx.get(url__startswith=pubmed_collect.ESEARCH_URL).mock(
        return_value=httpx.Response(200, json={"esearchresult": {"idlist": ["111", "222"]}})
    )
    assert pubmed_collect.fetch_pmc_ids(["ai"], max_results=5) == ["111", "222"]


def test_pubmed_parse_article():
    xml = """
    <article>
      <article-id pub-id-type="pmc">123456</article-id>
      <article-title>Deep Learning in Medicine</article-title>
      <contrib-group>
        <contrib contrib-type="author">
          <name><surname>Tanaka</surname><given-names>Yuki</given-names></name>
        </contrib>
      </contrib-group>
      <pub-date pub-type="epub"><year>2026</year><month>3</month><day>5</day></pub-date>
      <subject>Oncology</subject>
    </article>
    """
    meta = pubmed_collect._parse_article(ET.fromstring(xml))
    assert meta == {
        "pmc_id": "123456",
        "title": "Deep Learning in Medicine",
        "authors": ["Yuki Tanaka"],
        "pub_date": "2026-03-05",
        "categories": ["Oncology"],
    }


def test_pubmed_parse_article_without_pmc_id_returns_none():
    assert pubmed_collect._parse_article(ET.fromstring("<article/>")) is None


def test_openalex_reconstruct_abstract():
    inverted = {"deep": [0], "learning": [1], "wins": [2]}
    assert openalex_collect.reconstruct_abstract(inverted) == "deep learning wins"
    assert openalex_collect.reconstruct_abstract(None) == ""


def test_openalex_pdf_url_prefers_arxiv():
    work = {
        "ids": {"arxiv": "https://arxiv.org/abs/2608.27417"},
        "open_access": {"oa_url": "https://example.org/landing"},
    }
    assert openalex_collect.get_pdf_url(work) == "https://arxiv.org/pdf/2608.27417.pdf"


def test_openalex_pdf_url_requires_direct_pdf():
    assert openalex_collect.get_pdf_url({"open_access": {"oa_url": "https://example.org/landing"}}) is None
    assert (
        openalex_collect.get_pdf_url({"open_access": {"oa_url": "https://example.org/x.pdf"}})
        == "https://example.org/x.pdf"
    )


def test_portfolio_score_badge_tiers():
    assert "—" in generate_portfolio.score_badge(None)
    assert "bg-green-100" in generate_portfolio.score_badge(0.9)
    assert "bg-yellow-100" in generate_portfolio.score_badge(0.5)
    assert "bg-gray-100" in generate_portfolio.score_badge(0.1)


def test_portfolio_source_link_escapes_and_links():
    paper = {"source": "arxiv", "file_name": "arxiv_2410.10071v1.pdf", "title": "<b>Title</b>"}
    link = generate_portfolio.source_link(paper)
    assert "&lt;b&gt;Title&lt;/b&gt;" in link
    assert "arxiv.org/abs/" in link


def test_portfolio_build_html_smoke():
    papers = [{"id": 1, "title": "T", "authors": ["A"], "score": 0.8, "source": "arxiv", "file_name": "x.pdf"}]
    html_out = generate_portfolio.build_html(papers, {1: {"objective": "obj"}}, "2026-08-31")
    assert "1 papers" in html_out
    assert "obj" in html_out
