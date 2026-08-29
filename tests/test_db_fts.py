"""FTS5 keyword search tests for academic_paper.db module."""

import tempfile
from pathlib import Path

import pytest

from academic_paper.db import (
    escape_fts5_query,
    get_connection,
    init_db,
    save_chunks,
    save_paper,
    search_fts,
)


@pytest.fixture
def temp_db():
    """Create a temporary in-memory database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


def test_fts_search_returns_results(temp_db):
    """Test that FTS5 search returns results when query matches."""
    init_db(temp_db)
    conn = get_connection(temp_db)

    # Create a paper
    paper_id = save_paper(
        conn,
        file_name="test.pdf",
        file_hash="abc123",
        title="Machine Learning Paper",
    )

    # Create chunks with searchable content
    chunks = [
        {
            "text": "Machine learning is a field of artificial intelligence",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "qdrant_id": "q-001",
            "token_count": 10,
        },
        {
            "text": "Deep learning involves neural networks",
            "page_start": 2,
            "page_end": 2,
            "chunk_index": 1,
            "qdrant_id": "q-002",
            "token_count": 6,
        },
    ]

    # Save chunks
    save_chunks(conn, paper_id, chunks)

    # Search for "machine"
    results = search_fts(conn, "machine", limit=10)

    assert len(results) > 0
    assert "chunk_id" in results[0]
    assert "paper_id" in results[0]
    assert "text" in results[0]
    assert "rank" in results[0]
    assert any("machine" in r["text"].lower() for r in results)

    conn.close()


def test_fts_search_with_paper_id_filter(temp_db):
    """Test that FTS5 search correctly filters by paper_id."""
    init_db(temp_db)
    conn = get_connection(temp_db)

    # Create two papers
    paper_id_1 = save_paper(
        conn,
        file_name="paper1.pdf",
        file_hash="hash1",
        title="Machine Learning Paper",
    )

    paper_id_2 = save_paper(
        conn,
        file_name="paper2.pdf",
        file_hash="hash2",
        title="Deep Learning Paper",
    )

    # Create chunks for paper 1
    chunks_1 = [
        {
            "text": "Machine learning is about training algorithms",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "qdrant_id": "q-001",
            "token_count": 7,
        },
    ]

    # Create chunks for paper 2
    chunks_2 = [
        {
            "text": "Deep learning uses multiple layers",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "qdrant_id": "q-002",
            "token_count": 6,
        },
        {
            "text": "Machine learning and deep learning are related",
            "page_start": 2,
            "page_end": 2,
            "chunk_index": 1,
            "qdrant_id": "q-003",
            "token_count": 8,
        },
    ]

    # Save chunks
    save_chunks(conn, paper_id_1, chunks_1)
    save_chunks(conn, paper_id_2, chunks_2)

    # Search for "machine" in both papers
    results_all = search_fts(conn, "machine", limit=10)
    assert len(results_all) >= 2

    # Search for "machine" in paper 1 only
    results_filtered = search_fts(conn, "machine", limit=10, paper_id=paper_id_1)
    assert len(results_filtered) > 0
    assert all(r["paper_id"] == paper_id_1 for r in results_filtered)

    # Search for "machine" in paper 2 only
    results_filtered_2 = search_fts(conn, "machine", limit=10, paper_id=paper_id_2)
    assert len(results_filtered_2) > 0
    assert all(r["paper_id"] == paper_id_2 for r in results_filtered_2)

    conn.close()


def test_fts_search_no_match(temp_db):
    """Test that FTS5 search returns empty list when query doesn't match."""
    init_db(temp_db)
    conn = get_connection(temp_db)

    # Create a paper
    paper_id = save_paper(
        conn,
        file_name="test.pdf",
        file_hash="abc123",
        title="Machine Learning Paper",
    )

    # Create chunks with specific content
    chunks = [
        {
            "text": "This chunk talks about computer science",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "qdrant_id": "q-001",
            "token_count": 7,
        },
    ]

    # Save chunks
    save_chunks(conn, paper_id, chunks)

    # Search for something that doesn't exist
    results = search_fts(conn, "nonexistentword12345", limit=10)

    assert len(results) == 0
    assert isinstance(results, list)

    conn.close()


def test_escape_fts5_query_quotes_tokens():
    """Each token is wrapped as an FTS5 string literal; internal quotes doubled."""
    assert escape_fts5_query("machine learning") == '"machine" "learning"'
    assert escape_fts5_query('say "hi"') == '"say" """hi"""'
    assert escape_fts5_query("   ") == '""'


@pytest.mark.parametrize(
    "bad_query",
    [
        'unbalanced "quote',
        "NEAR(term",
        "text:column",
        "*leadingstar",
        "a AND b OR c",
        "foo^bar (baz",
    ],
)
def test_fts_search_handles_special_chars_without_error(temp_db, bad_query):
    """FTS5-special input must not raise a syntax error; it is treated literally."""
    init_db(temp_db)
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, file_name="t.pdf", file_hash="h-special", title="T")
    save_chunks(
        conn,
        paper_id,
        [
            {
                "text": "ordinary machine learning content",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
                "qdrant_id": "q-sp-0",
                "token_count": 4,
            }
        ],
    )

    # Must not raise sqlite3.OperationalError (fts5 syntax error)
    results = search_fts(conn, bad_query, limit=10)
    assert isinstance(results, list)
    conn.close()


def test_fts_operator_injection_is_neutralized(temp_db):
    """'a OR b' is matched as literal tokens, not an FTS OR, so a doc with only 'b' misses."""
    init_db(temp_db)
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, file_name="t.pdf", file_hash="h-or", title="T")
    save_chunks(
        conn,
        paper_id,
        [
            {
                "text": "alpha content only",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
                "qdrant_id": "q-or-0",
                "token_count": 3,
            }
        ],
    )

    # As a real FTS OR this would match (doc has 'alpha'); escaped it needs BOTH literals.
    results = search_fts(conn, "alpha OR zzzmissing", limit=10)
    assert results == []
    conn.close()
