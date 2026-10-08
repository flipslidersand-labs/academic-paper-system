"""Tests for arxiv_collect.ingest_paper() and main() (#493)."""

import contextlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import arxiv_collect  # noqa: E402


def _paper(published_date="2024-10-14"):
    return {
        "arxiv_id": "2410.10071",
        "pdf_url": "https://arxiv.org/pdf/2410.10071",
        "file_name": "2410.10071.pdf",
        "title": "A Title",
        "authors": ["Alice", "Bob"],
        "categories": ["cs.AI", "cs.LG"],
        "published_date": published_date,
    }


@pytest.fixture
def ingest_calls(monkeypatch):
    calls = {}

    @contextlib.contextmanager
    def fake_download(client, url, timeout):
        calls["download"] = (url, timeout)
        yield "/tmp/fake.pdf"

    def fake_ingest_pdf(client, api_url, file_name, tmp_path, meta, poll_timeout):
        calls["ingest"] = (api_url, file_name, tmp_path, meta, poll_timeout)
        return {"status": "indexed", "paper_id": 7}

    monkeypatch.setattr(arxiv_collect, "download_pdf", fake_download)
    monkeypatch.setattr(arxiv_collect, "ingest_pdf", fake_ingest_pdf)
    return calls


def test_ingest_paper_watermark_mismatch_warns_but_still_ingests(monkeypatch, capsys, ingest_calls):
    monkeypatch.setattr(arxiv_collect, "verify_arxiv_id", lambda path, expected: False)

    result = arxiv_collect.ingest_paper(None, _paper(), "http://api", pdf_timeout=5, poll_timeout=9)

    err = capsys.readouterr().err
    assert "WARN [2410.10071]" in err
    assert "Ingesting anyway" in err
    api_url, file_name, tmp_path, meta, poll_timeout = ingest_calls["ingest"]
    assert (api_url, file_name, tmp_path, poll_timeout) == ("http://api", "2410.10071.pdf", "/tmp/fake.pdf", 9)
    assert ingest_calls["download"] == ("https://arxiv.org/pdf/2410.10071", 5)
    assert meta == {
        "title": "A Title",
        "authors": json.dumps(["Alice", "Bob"]),
        "categories": json.dumps(["cs.AI", "cs.LG"]),
        "published_date": "2024-10-14",
        "source": "arxiv",
    }
    assert result == {"status": "indexed", "paper_id": 7, "label": "2410.10071", "arxiv_id": "2410.10071"}


@pytest.mark.parametrize("verdict", [True, None])
def test_ingest_paper_no_warning_when_match_or_unverifiable(monkeypatch, capsys, ingest_calls, verdict):
    monkeypatch.setattr(arxiv_collect, "verify_arxiv_id", lambda path, expected: verdict)

    arxiv_collect.ingest_paper(None, _paper(), "http://api")

    assert "WARN" not in capsys.readouterr().err
    assert "ingest" in ingest_calls


def test_ingest_paper_missing_published_date_sends_empty_string(monkeypatch, ingest_calls):
    monkeypatch.setattr(arxiv_collect, "verify_arxiv_id", lambda path, expected: True)

    arxiv_collect.ingest_paper(None, _paper(published_date=None), "http://api")

    assert ingest_calls["ingest"][3]["published_date"] == ""


def test_main_fetch_failure_exits_1(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("arxiv down")

    monkeypatch.setattr(sys, "argv", ["arxiv_collect.py"])
    monkeypatch.setattr(arxiv_collect, "fetch_papers", boom)
    monkeypatch.setattr(arxiv_collect, "run_collect", lambda *a, **k: pytest.fail("run_collect must not run"))

    with pytest.raises(SystemExit) as exc:
        arxiv_collect.main()

    assert exc.value.code == 1
    assert "ERROR fetching arXiv: arxiv down" in capsys.readouterr().err


def test_main_passes_arguments_to_run_collect(monkeypatch):
    papers = [_paper()]
    fetch_args = {}
    collect_args = {}

    def fake_fetch(categories, max_results, from_date="", until_date=""):
        fetch_args.update(categories=categories, max_results=max_results, from_date=from_date, until_date=until_date)
        return papers

    def fake_run_collect(label, got_papers, ingest_fn, summary_file):
        collect_args.update(label=label, papers=got_papers, ingest_fn=ingest_fn, summary_file=summary_file)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arxiv_collect.py",
            "--categories",
            "cs.CL",
            "--max",
            "3",
            "--from-date",
            "2024-10-01",
            "--until-date",
            "2024-10-31",
            "--api-url",
            "http://api:1",
            "--poll-timeout",
            "42",
            "--summary-file",
            "/tmp/s.json",
        ],
    )
    monkeypatch.setattr(arxiv_collect, "fetch_papers", fake_fetch)
    monkeypatch.setattr(arxiv_collect, "run_collect", fake_run_collect)

    arxiv_collect.main()

    assert fetch_args == {
        "categories": ["cs.CL"],
        "max_results": 3,
        "from_date": "2024-10-01",
        "until_date": "2024-10-31",
    }
    assert collect_args["label"] == "arXiv"
    assert collect_args["papers"] is papers
    assert collect_args["summary_file"] == "/tmp/s.json"

    # The ingest callback must forward api_url and poll_timeout to ingest_paper.
    seen = {}
    monkeypatch.setattr(
        arxiv_collect, "ingest_paper", lambda c, p, api, poll_timeout: seen.update(api=api, poll=poll_timeout)
    )
    collect_args["ingest_fn"]("client", papers[0])
    assert seen == {"api": "http://api:1", "poll": 42}
