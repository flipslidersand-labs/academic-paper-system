"""Shared arXiv ID pattern and watermark extraction (#424).

Centralizes the arXiv ID regex used independently by
scripts/arxiv_collect.py, scripts/fix_file_names.py, and academic_paper/db.py
so that a format change (e.g. supporting legacy pre-2007 IDs) only needs one
edit instead of three.
"""

import re

# Bare arXiv ID, e.g. "2410.10071" (new-style YYMM.NNNNN format).
ARXIV_ID_PATTERN = r"\d{4}\.\d{4,5}"

_ARXIV_WATERMARK_RE = re.compile(rf"arXiv:({ARXIV_ID_PATTERN})(v\d+)?")


def find_arxiv_watermark(text: str) -> tuple[str, str | None] | None:
    """Extract (arxiv_id, version) from text, scanning forward then mirrored.

    pdfplumber sometimes extracts the sideways arXiv watermark reversed
    (e.g. "1v17001.0142:viXra"), so the mirrored text is scanned too (#163).
    *version* includes the leading "v" (e.g. "v1") or is None when absent.
    Returns None when no watermark is found in either direction.
    """
    m = _ARXIV_WATERMARK_RE.search(text)
    if m:
        return m.group(1), m.group(2)
    m = _ARXIV_WATERMARK_RE.search(text[::-1])
    if m:
        return m.group(1), m.group(2)
    return None
