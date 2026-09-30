"""Tests for academic_paper.arxiv_ids (shared arXiv watermark extraction, #424)."""

from academic_paper.arxiv_ids import find_arxiv_watermark


def test_find_arxiv_watermark_forward():
    text = "arXiv:2410.10071v1 [cs.MA] 14 Oct 2024 Content Caching-Assisted Vehicular Edge Computing"
    assert find_arxiv_watermark(text) == ("2410.10071", "v1")


def test_find_arxiv_watermark_mirrored():
    # pdfplumber extracts the sideways watermark reversed (#163)
    text = "4202 tcO 41 ]AM.sc[ 1v17001.0142:viXra Content Caching-Assisted"
    assert find_arxiv_watermark(text) == ("2410.10071", "v1")


def test_find_arxiv_watermark_no_version():
    text = "arXiv:2410.10071 [cs.MA]"
    assert find_arxiv_watermark(text) == ("2410.10071", None)


def test_find_arxiv_watermark_absent():
    assert find_arxiv_watermark("ACCEPTED TO IEEE TRANSACTIONS 1 Cooperative UAVs") is None
