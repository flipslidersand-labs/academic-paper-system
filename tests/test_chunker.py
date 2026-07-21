"""Tests for page-aware text chunking."""

import pytest

from academic_paper.chunker import chunk_pages, DEFAULT_SIZE, DEFAULT_OVERLAP


def test_chunk_pages_basic() -> None:
    """Test that short text fits in a single chunk."""
    pages = [
        {"page": 1, "text": "This is a short text."}
    ]

    chunks = chunk_pages(pages, chunk_size=512, overlap=64)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["page_start"] == 1
    assert chunk["page_end"] == 1
    assert chunk["chunk_index"] == 0
    assert chunk["token_count"] == 5  # "This", "is", "a", "short", "text."
    assert "short text" in chunk["text"]


def test_chunk_pages_splits_long_text() -> None:
    """Test that long text is split into multiple chunks."""
    # Create a text with many words (more than chunk_size)
    words = ["word"] * 600  # 600 words, default chunk_size is 512
    long_text = " ".join(words)

    pages = [{"page": 1, "text": long_text}]

    chunks = chunk_pages(pages, chunk_size=512, overlap=64)

    # Should create multiple chunks
    assert len(chunks) > 1

    # Each chunk should have tokens <= chunk_size
    for chunk in chunks:
        assert chunk["token_count"] <= 512

    # First chunk should start at page 1
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 1

    # All chunks should be consecutive
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_chunk_pages_preserves_page_info() -> None:
    """Test that page_start and page_end are correctly set across pages."""
    # Create pages with specific page numbers
    pages = [
        {"page": 1, "text": "First page content with some text."},
        {"page": 2, "text": "Second page content with some text."},
        {"page": 3, "text": "Third page content with some text."},
    ]

    chunks = chunk_pages(pages, chunk_size=512, overlap=64)

    # Each page has 6 words, so should result in multiple chunks
    # but likely still within one per page since 6 * 3 = 18 words total
    assert len(chunks) >= 1

    # First chunk should start at page 1
    assert chunks[0]["page_start"] == 1

    # All chunks should have valid page ranges
    for chunk in chunks:
        assert chunk["page_start"] <= chunk["page_end"]
        assert chunk["page_start"] >= 1
        assert chunk["page_end"] <= 3


def test_chunk_pages_empty_input() -> None:
    """Test that empty input returns empty list."""
    chunks = chunk_pages([])
    assert chunks == []


def test_chunk_pages_overlap_validation() -> None:
    """Test that overlap >= chunk_size raises ValueError."""
    pages = [{"page": 1, "text": "Test text"}]

    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_pages(pages, chunk_size=100, overlap=100)

    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_pages(pages, chunk_size=100, overlap=150)


def test_chunk_pages_multipage_boundary() -> None:
    """Test chunking across page boundaries with small chunk size."""
    # Create pages with many small paragraphs to force splitting
    page1_text = "\n\n".join([
        "First paragraph with several words in it.",
        "Second paragraph also has multiple words.",
        "Third paragraph continues the content here.",
    ])
    page2_text = "\n\n".join([
        "Page two starts with new content here.",
        "Another paragraph on page two now.",
        "Final paragraph with more information.",
    ])

    pages = [
        {"page": 1, "text": page1_text},
        {"page": 2, "text": page2_text},
    ]

    chunks = chunk_pages(pages, chunk_size=20, overlap=5)

    # Should have multiple chunks
    assert len(chunks) > 1

    # All chunks should have valid structure
    for chunk in chunks:
        assert "text" in chunk
        assert "page_start" in chunk
        assert "page_end" in chunk
        assert "chunk_index" in chunk
        assert "token_count" in chunk
        assert chunk["page_start"] <= chunk["page_end"]
        assert chunk["token_count"] > 0


def test_chunk_pages_token_count_matches() -> None:
    """Test that token_count matches actual word count in text."""
    pages = [
        {"page": 1, "text": "The quick brown fox jumps over the lazy dog"}
    ]

    chunks = chunk_pages(pages, chunk_size=512, overlap=64)

    for chunk in chunks:
        expected_token_count = len(chunk["text"].split())
        assert chunk["token_count"] == expected_token_count


def test_chunk_pages_whitespace_handling() -> None:
    """Test that whitespace is properly handled in chunking."""
    pages = [
        {"page": 1, "text": "Line one.\n\nParagraph two with\nmultiple lines.\n\nFinal paragraph."}
    ]

    chunks = chunk_pages(pages, chunk_size=512, overlap=64)

    assert len(chunks) >= 1
    # All chunks should have non-empty text
    for chunk in chunks:
        assert chunk["text"].strip()
        assert len(chunk["text"]) > 0


def test_chunk_pages_long_paragraph_across_pages() -> None:
    """Test that a very long single paragraph is split across pages."""
    # Create one very long paragraph spanning conceptually across pages
    long_para = " ".join(["word"] * 100)

    pages = [
        {"page": 1, "text": long_para},
        {"page": 2, "text": long_para},
    ]

    # With small chunk size, should force splitting
    chunks = chunk_pages(pages, chunk_size=30, overlap=5)

    assert len(chunks) > 1
    # Each chunk should be under chunk_size
    for chunk in chunks:
        assert chunk["token_count"] <= 30
