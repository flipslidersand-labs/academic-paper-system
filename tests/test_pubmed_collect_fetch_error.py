"""PubMed efetch parse failures must not become a silent exit-0 run (#721)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pubmed_collect  # noqa: E402

_PAPER = {"pmc_id": "1", "title": "T", "authors": [], "pub_date": None, "categories": []}


def _run(monkeypatch, tmp_path, *, ids, meta):
    monkeypatch.setattr(pubmed_collect.time, "sleep", lambda s: None)
    monkeypatch.setattr(pubmed_collect, "fetch_pmc_ids", lambda *a, **k: ids)
    monkeypatch.setattr(pubmed_collect, "fetch_paper_metadata", meta)
    monkeypatch.setattr(pubmed_collect, "ingest_paper", lambda *a, **k: {"status": "ingested", "label": "PMC1"})
    out = tmp_path / "s.json"
    monkeypatch.setattr(sys, "argv", ["pubmed_collect.py", "--summary-file", str(out)])
    return out


def test_parse_error_writes_fetch_error_and_exits_1(monkeypatch, tmp_path):
    def _bad(*a, **k):
        raise pubmed_collect.ET.ParseError("syntax error")

    out = _run(monkeypatch, tmp_path, ids=["1", "2"], meta=_bad)
    with pytest.raises(SystemExit) as ei:
        pubmed_collect.main()
    assert ei.value.code == 1
    data = json.loads(out.read_text())
    assert data["fetched"] == 0
    assert "ParseError" in data["fetch_error"]


def test_zero_parsed_of_requested_exits_1(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, ids=["1", "2"], meta=lambda *a, **k: [])
    with pytest.raises(SystemExit) as ei:
        pubmed_collect.main()
    assert ei.value.code == 1
    assert json.loads(out.read_text())["fetch_error"] == "parsed 0 of 2 requested articles"


def test_partial_parse_records_fetch_error_but_ingests(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, ids=["1", "2", "3"], meta=lambda *a, **k: [dict(_PAPER)])
    pubmed_collect.main()
    data = json.loads(out.read_text())
    assert data["ingested"] == 1
    assert data["fetch_error"] == "parsed 1 of 3 requested articles"
