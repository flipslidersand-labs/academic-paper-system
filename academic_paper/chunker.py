"""Page-aware text chunking for academic papers.

Splits extracted pages into searchable chunks while preserving page metadata.
Adapts the paragraph-first + token-fallback algorithm from search-engine
to track page boundaries across chunk boundaries.
"""

from __future__ import annotations


def _split_paragraphs(text: str) -> list[str]:
    """Split text by double newlines into paragraphs."""
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def chunk_pages(
    pages: list[dict],
    chunk_size: int,
    overlap: int,
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
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
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
    buf_page_start = buf_page_end = 0

    def flush() -> None:
        nonlocal buf
        if buf:
            text = " ".join(buf)
            chunks.append(
                {
                    "text": text,
                    "page_start": buf_page_start,
                    "page_end": buf_page_end,
                    "chunk_index": len(chunks),
                    "token_count": len(buf),
                }
            )
            buf = []

    for para_words, page_num in paragraphs_with_pages:
        # Try to add paragraph to current buffer
        if len(buf) + len(para_words) <= chunk_size:
            if not buf:
                buf_page_start = page_num
            buf_page_end = page_num
            buf.extend(para_words)
            continue

        # Flush buffer if it has content
        flush()

        # If paragraph itself fits in chunk size
        if len(para_words) <= chunk_size:
            buf = para_words.copy()
            buf_page_start = buf_page_end = page_num
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

    # Flush remaining buffer
    flush()

    # Unreachable: with chunk_size > 0 validated above, every paragraph's words
    # are guaranteed to land in `chunks` either via the sliding window branch or
    # via the final flush(), so paragraphs_with_pages non-empty implies chunks
    # non-empty. Kept as an explicit safety net (see #344) instead of silently
    # returning an incomplete result if that invariant is ever broken.
    assert chunks, "unreachable: paragraphs present but no chunks created"

    return chunks
