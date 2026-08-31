"""Unit tests for scripts/arxiv_collect.py.

Uses respx to intercept httpx calls; tests argument parsing, XML parsing,
date filtering, retry logic, and the ingest loop.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import arxiv_collect  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atom_feed(entries: list[dict]) -> str:
    """Build a minimal Atom feed XML for the given paper dicts."""
    ns = "http://www.w3.org/2005/Atom"
    lines = [
        f'<feed xmlns="{ns}">',
        "<title>arXiv test</title>",
    ]
    for e in entries:
        lines.append("<entry>")
        lines.append(f"<id>http://arxiv.org/abs/{e['id']}</id>")
        lines.append(f"<title>{e.get('title', 'Test Paper')}</title>")
        for a in e.get("authors", ["Author One"]):
            lines.append(f"<author><name>{a}</name></author>")
        for c in e.get("categories", ["cs.AI"]):
            lines.append(f'<category term="{c}"/>')
        pub = e.get("published", "2026-01-01T00:00:00Z")
        lines.append(f"<published>{pub}</published>")
        lines.append("</entry>")
    lines.append("</feed>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# fetch_papers
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_papers_parses_entries():
    feed = _atom_feed([
        {"id": "2401.00001", "title": "Test ML Paper", "authors": ["Alice", "Bob"], "categories": ["cs.AI", "cs.LG"]},
        {"id": "2401.00002", "title": "Test NLP Paper", "authors": ["Carol"]},
    ])
    respx.get(url__startswith="https://export.arxiv.org").mock(
        return_value=httpx.Response(200, text=feed)
    )
    papers = arxiv_collect.fetch_papers(["cs.AI"], max_results=10)
    assert len(papers) == 2
    assert papers[0]["arxiv_id"] == "2401.00001"
    assert papers[0]["authors"] == ["Alice", "Bob"]
    assert "cs.AI" in papers[0]["categories"]
    assert papers[0]["pdf_url"].endswith("2401.00001.pdf")


@respx.mock
def test_fetch_papers_respects_max_results():
    entries = [{"id": f"2401.{i:05d}"} for i in range(20)]
    respx.get(url__startswith="https://export.arxiv.org").mock(
        return_value=httpx.Response(200, text=_atom_feed(entries))
    )
    papers = arxiv_collect.fetch_papers(["cs.AI"], max_results=5)
    assert len(papers) == 5


@respx.mock
def test_fetch_papers_post_filters_by_date():
    entries = [
        {"id": "2401.00001", "published": "2026-01-15T00:00:00Z"},
        {"id": "2401.00002", "published": "2025-12-01T00:00:00Z"},  # outside range
    ]
    respx.get(url__startswith="https://export.arxiv.org").mock(
        return_value=httpx.Response(200, text=_atom_feed(entries))
    )
    papers = arxiv_collect.fetch_papers(["cs.AI"], max_results=10, from_date="2026-01-01", until_date="2026-12-31")
    assert len(papers) == 1
    assert papers[0]["arxiv_id"] == "2401.00001"


@respx.mock
def test_fetch_papers_retries_on_timeout(monkeypatch):
    """with_retry must be called; a transient timeout eventually succeeds."""
    call_count = [0]
    feed = _atom_feed([{"id": "2401.00001"}])

    def _side_effect(request):
        call_count[0] += 1
        if call_count[0] < 2:
            raise httpx.TimeoutException("timeout", request=request)
        return httpx.Response(200, text=feed)

    respx.get(url__startswith="https://export.arxiv.org").mock(side_effect=_side_effect)
    # Patch time.sleep to avoid actual delay
    monkeypatch.setattr("time.sleep", lambda _: None)
    papers = arxiv_collect.fetch_papers(["cs.AI"], max_results=10)
    assert call_count[0] == 2
    assert len(papers) == 1


# ---------------------------------------------------------------------------
# ingest_paper
# ---------------------------------------------------------------------------


def test_ingest_paper_calls_download_and_ingest():
    paper = {
        "arxiv_id": "2401.00001",
        "title": "ML Paper",
        "authors": ["Alice"],
        "categories": ["cs.AI"],
        "published_date": "2026-01-01",
        "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
        "file_name": "arxiv_2401.00001.pdf",
    }
    mock_client = MagicMock()
    fake_result = {"status": "ingested", "paper_id": 1}

    with (
        patch("arxiv_collect.download_pdf") as mock_dl,
        patch("arxiv_collect.ingest_pdf", return_value=fake_result),
    ):
        mock_dl.return_value.__enter__ = lambda s: "/tmp/fake.pdf"
        mock_dl.return_value.__exit__ = MagicMock(return_value=False)
        result = arxiv_collect.ingest_paper(mock_client, paper, "http://api/")

    mock_dl.assert_called_once_with(mock_client, paper["pdf_url"], 60)
    assert result["arxiv_id"] == "2401.00001"
    assert result["label"] == "2401.00001"
    assert result["status"] == "ingested"


# ---------------------------------------------------------------------------
# main — argument parsing + run_collect integration
# ---------------------------------------------------------------------------


def test_main_calls_run_collect(monkeypatch):
    monkeypatch.setattr("sys.argv", ["arxiv_collect.py", "--max", "2", "--api-url", "http://api/"])
    fake_papers = [
        {
            "arxiv_id": "2401.00001",
            "title": "T",
            "authors": [],
            "categories": [],
            "published_date": "2026-01-01",
            "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
            "file_name": "arxiv_2401.00001.pdf",
        }
    ]
    with (
        patch("arxiv_collect.fetch_papers", return_value=fake_papers),
        patch("arxiv_collect.run_collect") as mock_rc,
    ):
        arxiv_collect.main()

    mock_rc.assert_called_once()
    call_args = mock_rc.call_args
    assert call_args[0][0] == "arXiv"
    assert call_args[0][1] == fake_papers


def test_main_exits_on_fetch_error(monkeypatch):
    monkeypatch.setattr("sys.argv", ["arxiv_collect.py"])
    with patch("arxiv_collect.fetch_papers", side_effect=RuntimeError("API down")):
        with pytest.raises(SystemExit) as exc_info:
            arxiv_collect.main()
    assert exc_info.value.code == 1
