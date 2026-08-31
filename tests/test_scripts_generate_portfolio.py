"""Unit tests for scripts/generate_portfolio.py."""

import json
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_portfolio  # noqa: E402

# ---------------------------------------------------------------------------
# score_badge
# ---------------------------------------------------------------------------


def test_score_badge_high():
    html = generate_portfolio.score_badge(0.9)
    assert "green" in html
    assert "90" in html


def test_score_badge_medium():
    html = generate_portfolio.score_badge(0.5)
    assert "yellow" in html


def test_score_badge_low():
    html = generate_portfolio.score_badge(0.2)
    assert "gray" in html


def test_score_badge_none():
    html = generate_portfolio.score_badge(None)
    assert "—" in html


# ---------------------------------------------------------------------------
# source_link
# ---------------------------------------------------------------------------


def test_source_link_arxiv():
    paper = {"source": "arxiv", "file_name": "arxiv_2401.00001.pdf", "title": "ML Paper"}
    html = generate_portfolio.source_link(paper)
    assert "arxiv.org/abs/" in html
    assert "ML Paper" in html


def test_source_link_non_arxiv():
    paper = {"source": "pubmed", "title": "PMC Paper"}
    html = generate_portfolio.source_link(paper)
    assert "<span>" in html
    assert "PMC Paper" in html


def test_source_link_xss_escape():
    paper = {"source": "arxiv", "file_name": "arxiv_2401.pdf", "title": "<script>alert(1)</script>"}
    html = generate_portfolio.source_link(paper)
    assert "<script>" not in html


# ---------------------------------------------------------------------------
# category_badges
# ---------------------------------------------------------------------------


def test_category_badges_empty():
    assert generate_portfolio.category_badges([]) == ""


def test_category_badges_renders_up_to_5():
    cats = ["A", "B", "C", "D", "E", "F"]
    html = generate_portfolio.category_badges(cats)
    assert "F" not in html  # 6th category truncated
    assert "A" in html and "E" in html


# ---------------------------------------------------------------------------
# build_html
# ---------------------------------------------------------------------------


def test_build_html_contains_paper_title():
    papers = [{"id": 1, "title": "My ML Paper", "source": "arxiv", "file_name": "arxiv_2401.pdf", "authors": ["Alice"]}]
    html = generate_portfolio.build_html(papers, {}, "2026-01-01")
    assert "My ML Paper" in html
    assert "Alice" in html


def test_build_html_includes_summary_objective():
    papers = [{"id": 1, "title": "Paper", "source": "pubmed"}]
    summaries = {1: {"objective": "To study something important"}}
    html = generate_portfolio.build_html(papers, summaries, "2026-01-01")
    assert "To study something important" in html


def test_build_html_truncates_authors_at_4():
    papers = [{"id": 1, "title": "P", "source": "arxiv", "file_name": "arxiv_x.pdf",
               "authors": ["A", "B", "C", "D", "E"]}]
    html = generate_portfolio.build_html(papers, {}, "2026-01-01")
    assert "et al." in html


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------


def _fake_urlopen(pages: list[list[dict]], key: str):
    """Return a fake urlopen that yields pages sequentially."""
    call_count = [0]

    def _urlopen(url, timeout=15):
        idx = call_count[0]
        call_count[0] += 1
        batch = pages[idx] if idx < len(pages) else []
        data = json.dumps({key: batch}).encode()

        class _Resp:
            def read(self):
                return data

        return _Resp()

    return _urlopen


def test_fetch_all_paginates():
    page1 = [{"id": i} for i in range(100)]
    page2 = [{"id": i} for i in range(100, 150)]
    urlopen = _fake_urlopen([page1, page2], "papers")
    with patch("generate_portfolio.urlopen", urlopen):
        items = generate_portfolio.fetch_all("http://api/papers?")
    assert len(items) == 150


def test_fetch_all_stops_on_partial_page():
    page = [{"id": i} for i in range(50)]
    urlopen = _fake_urlopen([page], "papers")
    with patch("generate_portfolio.urlopen", urlopen):
        items = generate_portfolio.fetch_all("http://api/papers?")
    assert len(items) == 50


def test_fetch_all_handles_url_error(capsys):
    def _fail(*args, **kwargs):
        raise URLError("connection refused")

    with patch("generate_portfolio.urlopen", _fail):
        items = generate_portfolio.fetch_all("http://api/papers?")
    assert items == []
    assert "fetch failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_writes_index_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate_portfolio.py", "--api-url", "http://api", "--output-dir", str(tmp_path)])
    papers = [{"id": 1, "title": "Paper A", "source": "arxiv", "file_name": "arxiv_1.pdf", "authors": []}]
    summaries = [{"paper_id": 1, "objective": "obj"}]

    with (
        patch("generate_portfolio.fetch_all", side_effect=[papers, summaries]),
    ):
        generate_portfolio.main()

    assert (tmp_path / "index.html").exists()
    data = json.loads((tmp_path / "papers.json").read_text())
    assert data["count"] == 1
    assert data["papers"][0]["title"] == "Paper A"
