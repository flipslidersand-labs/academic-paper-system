"""Unit tests for collector scripts' fetch/parse logic (#151).

These cover the code paths the nightly cron exercises before hitting the
ingest API: search-API response parsing, filtering, and pure formatting
helpers. HTTP calls are mocked with respx.
"""

import json
import xml.etree.ElementTree as ET
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

# scripts/ is added to sys.path in tests/conftest.py (#272).
import generate_portfolio  # noqa: E402
import httpx
import openalex_collect  # noqa: E402
import pubmed_collect  # noqa: E402
import pytest
import respx
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


def _fake_urlopen(pages, key="papers"):
    """Build a urlopen stand-in that pages through `pages` by the offset query param."""

    def _urlopen(url, timeout=15):
        offset = int(parse_qs(urlparse(url).query).get("offset", ["0"])[0])
        page = pages[offset // 100] if offset // 100 < len(pages) else []

        class _Resp:
            def read(self):
                return json.dumps({key: page}).encode()

        return _Resp()

    return _urlopen


def test_portfolio_fetch_all_stops_on_partial_batch(monkeypatch):
    pages = [[{"id": i} for i in range(100)], [{"id": i} for i in range(100, 130)]]
    monkeypatch.setattr(generate_portfolio, "urlopen", _fake_urlopen(pages))

    items = generate_portfolio.fetch_all("http://api.example/papers?sort=score")

    assert len(items) == 130
    assert items[0]["id"] == 0
    assert items[-1]["id"] == 129


def test_portfolio_fetch_all_stops_on_empty_batch(monkeypatch):
    monkeypatch.setattr(generate_portfolio, "urlopen", _fake_urlopen([[]]))

    assert generate_portfolio.fetch_all("http://api.example/summaries?") == []


def test_portfolio_fetch_all_returns_partial_results_on_url_error(monkeypatch, capsys):
    def _raise(url, timeout=15):
        raise URLError("connection refused")

    monkeypatch.setattr(generate_portfolio, "urlopen", _raise)

    items = generate_portfolio.fetch_all("http://api.example/papers?sort=score")

    assert items == []
    assert "[warn] fetch failed" in capsys.readouterr().err


def test_portfolio_category_badges_escapes_and_limits_to_five():
    cats = ["<script>", "b", "c", "d", "e", "f"]
    out = generate_portfolio.category_badges(cats)

    assert "&lt;script&gt;" in out
    for c in ["b", "c", "d", "e"]:
        assert f">{c}<" in out
    assert ">f<" not in out


def test_portfolio_category_badges_empty_input():
    assert generate_portfolio.category_badges([]) == ""
    assert generate_portfolio.category_badges(None) == ""


def test_portfolio_main_writes_index_and_json(monkeypatch, tmp_path):
    papers = [{"id": 1, "title": "T", "authors": ["A"], "score": 0.8, "source": "arxiv", "file_name": "x.pdf"}]
    summaries = [{"paper_id": 1, "objective": "obj"}]

    def _fake_fetch_all(url):
        return summaries if "summaries" in url else papers

    monkeypatch.setattr(generate_portfolio, "fetch_all", _fake_fetch_all)
    out_dir = tmp_path / "docs"
    monkeypatch.setattr(
        "sys.argv", ["generate_portfolio.py", "--api-url", "http://api.example", "--output-dir", str(out_dir)]
    )

    generate_portfolio.main()

    index_html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "1 papers" in index_html
    assert "obj" in index_html
    papers_json = json.loads((out_dir / "papers.json").read_text(encoding="utf-8"))
    assert papers_json["count"] == 1
    assert papers_json["papers"] == papers


def test_portfolio_main_aborts_when_no_papers_fetched(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_portfolio, "fetch_all", lambda url: [])
    out_dir = tmp_path / "docs"
    monkeypatch.setattr("sys.argv", ["generate_portfolio.py", "--output-dir", str(out_dir)])

    with pytest.raises(SystemExit) as exc_info:
        generate_portfolio.main()

    assert exc_info.value.code == 1
    assert not (out_dir / "index.html").exists()
