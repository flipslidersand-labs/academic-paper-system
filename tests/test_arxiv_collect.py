"""Tests for scripts/arxiv_collect.py watermark verification (#163)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import arxiv_collect  # noqa: E402


def test_find_arxiv_id_forward():
    text = "arXiv:2410.10071v1 [cs.MA] 14 Oct 2024 Content Caching-Assisted Vehicular Edge Computing"
    assert arxiv_collect.find_arxiv_id_in_text(text) == "2410.10071"


def test_find_arxiv_id_mirrored():
    # pdfplumber extracts the sideways watermark reversed
    text = "4202 tcO 41 ]AM.sc[ 1v17001.0142:viXra Content Caching-Assisted"
    assert arxiv_collect.find_arxiv_id_in_text(text) == "2410.10071"


def test_find_arxiv_id_absent():
    assert arxiv_collect.find_arxiv_id_in_text("ACCEPTED TO IEEE TRANSACTIONS 1 Cooperative UAVs") is None


def test_verify_arxiv_id_unreadable_pdf_returns_none():
    # Not a real PDF — extraction fails, so the check is skipped (None), not a mismatch
    assert arxiv_collect.verify_arxiv_id(b"not a pdf", "2410.10071v1") is None


def test_verify_arxiv_id_match_and_mismatch(monkeypatch):
    monkeypatch.setattr(arxiv_collect, "_extract_first_page_text", lambda content: "arXiv:2410.10071v1 [cs.MA]")
    assert arxiv_collect.verify_arxiv_id(b"%PDF-", "2410.10071v1") is True
    assert arxiv_collect.verify_arxiv_id(b"%PDF-", "2410.10071v2") is True  # version ignored
    assert arxiv_collect.verify_arxiv_id(b"%PDF-", "2005.11401v4") is False
