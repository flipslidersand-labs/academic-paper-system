"""Page-aware text chunking for academic papers.

Splits extracted pages into searchable chunks while preserving page metadata.
Adapts the paragraph-first + token-fallback algorithm from search-engine
to track page boundaries across chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SIZE = 512
DEFAULT_OVERLAP = 64


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    token_count: int


def _split_paragraphs(text: str) -> list[str]:
    """Split text by double newlines into paragraphs."""
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def chunk_pages(
    pages: list[dict],
    chunk_size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """Generate chunks from page list with page boundary tracking.

    Args:
        pages: List of dicts with "page" (int) and "text" (str) keys.
        chunk_size: Target chunk size in tokens (word count).
        overlap: Overlap between chunks in tokens.

    Returns:
        List of dicts with keys:
        - "text": chunk text
        - "page_start": starting page number
        - "page_end": ending page number
        - "chunk_index": 0-based chunk index
        - "token_count": word count in chunk
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not pages:
        return []

    # Collect all paragraphs with page tracking
    paragraphs_with_pages: list[tuple[list[str], int]] = []  # (words, page_num)

    for page_info in pages:
        page_num = page_info["page"]
        text = page_info["text"]

        for para in _split_paragraphs(text) or [text]:
            words = para.split()
            if words:
                paragraphs_with_pages.append((words, page_num))

    if not paragraphs_with_pages:
        return []

    # Apply search-engine chunking algorithm with page tracking
    chunks: list[dict] = []
    buf: list[str] = []
    buf_pages: list[int] = []

    def flush() -> None:
        nonlocal buf, buf_pages
        if buf:
            text = " ".join(buf)
            page_start = min(buf_pages) if buf_pages else 1
            page_end = max(buf_pages) if buf_pages else 1
            chunks.append(
                {
                    "text": text,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": len(chunks),
                    "token_count": len(buf),
                }
            )
            buf = []
            buf_pages = []

    for para_words, page_num in paragraphs_with_pages:
        # Try to add paragraph to current buffer
        if len(buf) + len(para_words) <= chunk_size:
            buf.extend(para_words)
            buf_pages.extend([page_num] * len(para_words))
            continue

        # Flush buffer if it has content
        flush()

        # If paragraph itself fits in chunk size
        if len(para_words) <= chunk_size:
            buf = para_words.copy()
            buf_pages = [page_num] * len(para_words)
            continue

        # Split long paragraph using sliding window
        step = chunk_size - overlap
        for start in range(0, len(para_words), step):
            window = para_words[start : start + chunk_size]
            if not window:
                break
            text = " ".join(window)
            chunks.append(
                {
                    "text": text,
                    "page_start": page_num,
                    "page_end": page_num,
                    "chunk_index": len(chunks),
                    "token_count": len(window),
                }
            )
            if start + chunk_size >= len(para_words):
                break

        buf = []
        buf_pages = []

    # Flush remaining buffer
    flush()

    # If no chunks were created but we have data, create a single chunk
    if not chunks and paragraphs_with_pages:
        all_words = []
        page_nums = []
        for words, page_num in paragraphs_with_pages:
            all_words.extend(words)
            page_nums.extend([page_num] * len(words))
        text = " ".join(all_words)
        chunks.append(
            {
                "text": text,
                "page_start": min(page_nums),
                "page_end": max(page_nums),
                "chunk_index": 0,
                "token_count": len(all_words),
            }
        )

    return chunks
