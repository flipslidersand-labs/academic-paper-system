"""Unit tests for semantic_scholar_collect.py, pubmed_collect.py, and openalex_collect.py."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import openalex_collect  # noqa: E402
import pubmed_collect  # noqa: E402
import semantic_scholar_collect  # noqa: E402

# ===========================================================================
# Semantic Scholar
# ===========================================================================


def _s2_paper(paper_id="abc123", pub_date="2026-01-15"):
    return {
        "paperId": paper_id,
        "title": "Test S2 Paper",
        "authors": [{"name": "Alice"}],
        "publicationDate": pub_date,
        "fieldsOfStudy": ["Computer Science"],
        "openAccessPdf": {"url": f"https://arxiv.org/pdf/{paper_id}.pdf"},
    }


@respx.mock
def test_s2_fetch_papers_returns_open_access_only():
    data = {
        "data": [
            _s2_paper("open1"),
            {"paperId": "closed", "title": "Closed", "authors": [], "openAccessPdf": None},
        ],
        "total": 2,
    }
    respx.get(url__startswith="https://api.semanticscholar.org").mock(
        return_value=httpx.Response(200, json=data)
    )
    papers, err = semantic_scholar_collect.fetch_papers("machine learning", max_results=10)
    assert err is None
    assert len(papers) == 1
    assert papers[0]["paperId"] == "open1"


@respx.mock
def test_s2_fetch_papers_handles_api_error():
    respx.get(url__startswith="https://api.semanticscholar.org").mock(
        return_value=httpx.Response(500, text="server error")
    )
    papers, err = semantic_scholar_collect.fetch_papers("test", max_results=5)
    assert len(papers) == 0
    assert err is not None


def test_s2_ingest_paper_builds_correct_metadata():
    paper = _s2_paper("abc123", "2026-01-15")
    mock_client = MagicMock()
    fake_result = {"status": "ingested", "paper_id": 1}

    with (
        patch("semantic_scholar_collect.download_pdf") as mock_dl,
        patch("semantic_scholar_collect.ingest_pdf", return_value=fake_result) as mock_ip,
    ):
        mock_dl.return_value.__enter__ = lambda s: "/tmp/fake.pdf"
        mock_dl.return_value.__exit__ = MagicMock(return_value=False)
        result = semantic_scholar_collect.ingest_paper(mock_client, paper, "http://api/")

    _, kwargs = mock_ip.call_args
    meta = mock_ip.call_args[0][4]  # metadata positional arg
    assert meta["source"] == "semantic_scholar"
    assert meta["published_date"] == "2026-01-15"
    assert json.loads(meta["authors"]) == ["Alice"]
    assert result["s2_id"] == "abc123"
    assert result["label"] == "abc123"[:8]


def test_s2_main_calls_run_collect(monkeypatch):
    monkeypatch.setattr("sys.argv", ["s2.py", "--max", "3"])
    fake_papers = [_s2_paper()]
    with (
        patch("semantic_scholar_collect.fetch_papers", return_value=(fake_papers, None)),
        patch("semantic_scholar_collect.run_collect") as mock_rc,
    ):
        semantic_scholar_collect.main()
    mock_rc.assert_called_once()
    assert mock_rc.call_args[0][0] == "Semantic Scholar"


def test_s2_main_exits_when_fetch_fails_with_no_papers(monkeypatch):
    monkeypatch.setattr("sys.argv", ["s2.py"])
    with patch("semantic_scholar_collect.fetch_papers", return_value=([], "network error")):
        with pytest.raises(SystemExit) as exc_info:
            semantic_scholar_collect.main()
    assert exc_info.value.code == 1


# ===========================================================================
# PubMed
# ===========================================================================


def _pmc_xml(pmc_id="1234567"):
    return f"""<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <front>
      <article-meta>
        <article-id pub-id-type="pmc">{pmc_id}</article-id>
        <title-group>
          <article-title>PubMed Test Paper</article-title>
        </title-group>
        <contrib-group>
          <contrib contrib-type="author">
            <name><surname>Smith</surname><given-names>John</given-names></name>
          </contrib>
        </contrib-group>
        <pub-date pub-type="epub">
          <year>2026</year><month>01</month><day>10</day>
        </pub-date>
        <article-categories>
          <subj-group><subject>Artificial Intelligence</subject></subj-group>
        </article-categories>
      </article-meta>
    </front>
  </article>
</pmc-articleset>"""


@respx.mock
def test_pubmed_fetch_pmc_ids():
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch").mock(
        return_value=httpx.Response(200, json={"esearchresult": {"idlist": ["1234567", "8901234"]}})
    )
    ids = pubmed_collect.fetch_pmc_ids(["machine learning"], max_results=10)
    assert ids == ["1234567", "8901234"]


@respx.mock
def test_pubmed_fetch_paper_metadata():
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch").mock(
        return_value=httpx.Response(200, content=_pmc_xml("1234567").encode())
    )
    papers = pubmed_collect.fetch_paper_metadata(["1234567"])
    assert len(papers) == 1
    p = papers[0]
    assert p["pmc_id"] == "1234567"
    assert p["title"] == "PubMed Test Paper"
    assert "John Smith" in p["authors"]
    assert p["pub_date"] == "2026-01-10"
    assert "Artificial Intelligence" in p["categories"]


def test_pubmed_ingest_paper_builds_metadata():
    paper = {"pmc_id": "1234567", "title": "PMC Paper", "authors": ["Alice"], "pub_date": "2026-01-10", "categories": ["AI"]}
    mock_client = MagicMock()
    fake_result = {"status": "ingested", "paper_id": 1}

    with (
        patch("pubmed_collect.download_pdf") as mock_dl,
        patch("pubmed_collect.ingest_pdf", return_value=fake_result) as mock_ip,
    ):
        mock_dl.return_value.__enter__ = lambda s: "/tmp/fake.pdf"
        mock_dl.return_value.__exit__ = MagicMock(return_value=False)
        result = pubmed_collect.ingest_paper(mock_client, paper, "http://api/")

    meta = mock_ip.call_args[0][4]
    assert meta["source"] == "pubmed"
    assert meta["published_date"] == "2026-01-10"
    assert result["pmc_id"] == "1234567"
    assert result["label"] == "PMC1234567"


def test_pubmed_main_calls_run_collect(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pubmed.py", "--max", "3"])
    with (
        patch("pubmed_collect.fetch_pmc_ids", return_value=["111"]),
        patch("pubmed_collect.fetch_paper_metadata", return_value=[
            {"pmc_id": "111", "title": "T", "authors": [], "pub_date": "2026-01-01", "categories": []}
        ]),
        patch("pubmed_collect.run_collect") as mock_rc,
    ):
        pubmed_collect.main()
    mock_rc.assert_called_once()
    assert mock_rc.call_args[0][0] == "PubMed"


def test_pubmed_main_exits_when_no_pmc_ids(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pubmed.py"])
    with patch("pubmed_collect.fetch_pmc_ids", return_value=[]):
        # No papers → exits 0 (success, not an error)
        try:
            pubmed_collect.main()
        except SystemExit as exc:
            assert exc.code == 0


# ===========================================================================
# OpenAlex
# ===========================================================================


def _oa_work(work_id="W123", pub_date="2026-01-20", arxiv_id="2401.00001"):
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": "OpenAlex Test Paper",
        "publication_date": pub_date,
        "authorships": [{"author": {"display_name": "Alice"}}],
        "topics": [{"field": {"display_name": "Machine Learning"}}],
        "abstract_inverted_index": None,
        "ids": {"arxiv": f"https://arxiv.org/abs/{arxiv_id}"},
        "open_access": {"oa_url": None},
        "primary_location": None,
        "_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    }


@respx.mock
def test_openalex_fetch_papers_returns_works_with_pdf():
    data = {
        "results": [_oa_work("W1"), _oa_work("W2", arxiv_id="")],
        "meta": {"next_cursor": None},
    }
    # W2 has no arxiv ID and no oa_url → get_pdf_url returns None → excluded
    w2 = _oa_work("W2", arxiv_id="")
    w2["ids"] = {}
    data["results"] = [_oa_work("W1"), w2]

    respx.get(url__startswith="https://api.openalex.org").mock(
        return_value=httpx.Response(200, json=data)
    )
    papers, err = openalex_collect.fetch_papers("RAG", max_results=10)
    assert err is None
    assert len(papers) == 1
    assert papers[0]["id"].endswith("W1")


def test_openalex_get_pdf_url_prefers_arxiv():
    work = _oa_work(arxiv_id="2401.12345")
    url = openalex_collect.get_pdf_url(work)
    assert url == "https://arxiv.org/pdf/2401.12345.pdf"


def test_openalex_extract_arxiv_id():
    work = {"ids": {"arxiv": "https://arxiv.org/abs/2401.12345"}}
    assert openalex_collect.extract_arxiv_id(work) == "2401.12345"


def test_openalex_ingest_paper_builds_metadata():
    work = _oa_work("W1", arxiv_id="2401.00001")
    mock_client = MagicMock()
    fake_result = {"status": "ingested", "paper_id": 1}

    with (
        patch("openalex_collect.download_pdf") as mock_dl,
        patch("openalex_collect.ingest_pdf", return_value=fake_result),
    ):
        mock_dl.return_value.__enter__ = lambda s: "/tmp/fake.pdf"
        mock_dl.return_value.__exit__ = MagicMock(return_value=False)
        result = openalex_collect.ingest_paper(mock_client, work, "http://api/")

    assert result["arxiv_id"] == "2401.00001"
    assert result["label"] == "arXiv:2401.00001"


def test_openalex_main_calls_run_collect(monkeypatch):
    monkeypatch.setattr("sys.argv", ["openalex.py", "--max", "2"])
    work = _oa_work("W1")
    with (
        patch("openalex_collect.fetch_papers", return_value=([work], None)),
        patch("openalex_collect.run_collect") as mock_rc,
    ):
        openalex_collect.main()
    mock_rc.assert_called_once()
    assert mock_rc.call_args[0][0] == "OpenAlex"


def test_openalex_reconstruct_abstract():
    inv = {"the": [0, 5], "quick": [1], "fox": [2]}
    text = openalex_collect.reconstruct_abstract(inv)
    assert text.startswith("the quick fox")
