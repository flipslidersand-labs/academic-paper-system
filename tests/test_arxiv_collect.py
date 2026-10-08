"""Tests for scripts/arxiv_collect.py watermark verification (#163)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import arxiv_collect  # noqa: E402


def test_find_arxiv_id_forward():
    # Forward/mirrored scan itself is covered by tests/test_arxiv_ids.py (#424);
    # this is just a regression check on the thin wrapper's return shape.
    text = "arXiv:2410.10071v1 [cs.MA] 14 Oct 2024 Content Caching-Assisted Vehicular Edge Computing"
    assert arxiv_collect.find_arxiv_id_in_text(text) == "2410.10071"


def test_verify_arxiv_id_unreadable_pdf_returns_none():
    # Not a real PDF — extraction fails, so the check is skipped (None), not a mismatch
    assert arxiv_collect.verify_arxiv_id(b"not a pdf", "2410.10071v1") is None


def test_verify_arxiv_id_match_and_mismatch(monkeypatch):
    monkeypatch.setattr(arxiv_collect, "_extract_first_page_text", lambda content: "arXiv:2410.10071v1 [cs.MA]")
    assert arxiv_collect.verify_arxiv_id(b"%PDF-", "2410.10071v1") is True
    assert arxiv_collect.verify_arxiv_id(b"%PDF-", "2410.10071v2") is True  # version ignored
    assert arxiv_collect.verify_arxiv_id(b"%PDF-", "2005.11401v4") is False


def test_extract_first_page_text_hang_times_out(monkeypatch):
    """A pdfplumber extraction that hangs (e.g. malicious PDF) must not block
    the caller forever — the call returns None once the deadline passes (#307)."""
    import time

    class _HangingPdf:
        def __enter__(self):
            time.sleep(5)
            return self

        def __exit__(self, *exc):
            return False

    class _FakePdfplumber:
        @staticmethod
        def open(_source):
            return _HangingPdf()

    monkeypatch.setitem(sys.modules, "pdfplumber", _FakePdfplumber())

    start = time.monotonic()
    result = arxiv_collect._extract_first_page_text(b"%PDF-", timeout=0.2)
    elapsed = time.monotonic() - start

    assert result is None
    assert elapsed < 2  # bounded by the timeout, not the 5s hang


# --- #526: fetch failure summary + shared ingest path ---


def test_main_fetch_failure_writes_fetch_error_summary_and_exits_1(monkeypatch, tmp_path):
    import json

    import pytest

    def _boom(*_a, **_k):
        raise RuntimeError("arxiv down")

    monkeypatch.setattr(arxiv_collect, "fetch_papers", _boom)
    out = tmp_path / "s.json"
    monkeypatch.setattr(sys, "argv", ["arxiv_collect.py", "--summary-file", str(out)])
    with pytest.raises(SystemExit) as ei:
        arxiv_collect.main()
    assert ei.value.code == 1
    data = json.loads(out.read_text())
    assert data["fetched"] == 0
    assert data["fetch_error"] == "arxiv down"
    assert {"ingested", "duplicate", "failed", "detail"} <= data.keys()


def test_main_fetch_failure_without_summary_file_still_exits_1(monkeypatch):
    import pytest

    monkeypatch.setattr(arxiv_collect, "fetch_papers", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(sys, "argv", ["arxiv_collect.py"])
    with pytest.raises(SystemExit) as ei:
        arxiv_collect.main()
    assert ei.value.code == 1


def _paper():
    return {
        "arxiv_id": "2410.10071v1",
        "pdf_url": "https://arxiv.org/pdf/2410.10071v1",
        "file_name": "2410.10071v1.pdf",
        "title": "T",
        "authors": ["A"],
        "categories": ["cs.AI"],
        "published_date": None,
    }


def test_ingest_paper_metadata_and_warn_on_mismatch(monkeypatch, capsys):
    import json
    from contextlib import contextmanager

    captured = {}

    @contextmanager
    def _fake_download(client, url, timeout=60, max_mb=None):
        yield "/tmp/fake.pdf"

    def _fake_ingest(client, api_url, file_name, tmp_path, metadata, poll_timeout=300):
        captured["metadata"] = metadata
        captured["file_name"] = file_name
        return {"status": "ingested"}

    import _collect_common

    monkeypatch.setattr(_collect_common, "download_pdf", _fake_download)
    monkeypatch.setattr(_collect_common, "ingest_pdf", _fake_ingest)
    monkeypatch.setattr(arxiv_collect, "verify_arxiv_id", lambda *_a: False)

    result = arxiv_collect.ingest_paper(None, _paper(), "http://api")
    assert result["label"] == "2410.10071v1"
    assert result["status"] == "ingested"
    assert captured["file_name"] == "2410.10071v1.pdf"
    assert captured["metadata"] == {
        "title": "T",
        "authors": json.dumps(["A"]),
        "categories": json.dumps(["cs.AI"]),
        "published_date": "",
        "source": "arxiv",
    }
    assert "WARN" in capsys.readouterr().err
