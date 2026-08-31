"""Tests for academic_paper.db module."""

import json
import tempfile
from pathlib import Path

import pytest

from academic_paper.db import (
    arxiv_id_from_file_name,
    get_chunks,
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


def test_init_db_creates_tables(temp_db):
    """Test that init_db creates all required tables."""
    init_db(temp_db)

    conn = get_connection(temp_db)
    cursor = conn.cursor()

    # Check papers table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
    assert cursor.fetchone() is not None

    # Check chunks table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
    assert cursor.fetchone() is not None

    # Check summaries table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='summaries'")
    assert cursor.fetchone() is not None

    # Check chunks_fts virtual table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
    assert cursor.fetchone() is not None

    conn.close()


def test_save_paper_returns_id(temp_db):
    """Test that save_paper returns a valid paper_id."""
    init_db(temp_db)
    conn = get_connection(temp_db)

    paper_id = save_paper(
        conn,
        file_name="test.pdf",
        file_hash="abc123",
        title="Test Paper",
        authors=["Author 1", "Author 2"],
        year=2023,
        pages=10,
    )

    assert isinstance(paper_id, int)
    assert paper_id > 0

    # Verify paper was saved
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()
    assert row is not None
    assert row["file_name"] == "test.pdf"
    assert row["title"] == "Test Paper"
    assert row["status"] == "pending"

    # Verify authors were serialized correctly
    authors = json.loads(row["authors"])
    assert authors == ["Author 1", "Author 2"]

    conn.close()


def test_save_chunks_and_get_chunks(temp_db):
    """Test saving and retrieving chunks."""
    init_db(temp_db)
    conn = get_connection(temp_db)

    # Create a paper
    paper_id = save_paper(
        conn,
        file_name="test.pdf",
        file_hash="abc123",
        title="Test Paper",
    )

    # Create chunks
    chunks = [
        {
            "text": "This is the first chunk",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "qdrant_id": "q-001",
            "token_count": 5,
        },
        {
            "text": "This is the second chunk",
            "page_start": 2,
            "page_end": 2,
            "chunk_index": 1,
            "qdrant_id": "q-002",
            "token_count": 5,
        },
    ]

    # Save chunks
    save_chunks(conn, paper_id, chunks)

    # Retrieve chunks
    retrieved_chunks = get_chunks(conn, paper_id)

    assert len(retrieved_chunks) == 2
    assert retrieved_chunks[0]["text"] == "This is the first chunk"
    assert retrieved_chunks[0]["qdrant_id"] == "q-001"
    assert retrieved_chunks[1]["text"] == "This is the second chunk"
    assert retrieved_chunks[1]["chunk_index"] == 1

    conn.close()


def test_search_fts(temp_db):
    """Test FTS5 search functionality."""
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
            "text": "Machine learning is a subset of artificial intelligence",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "qdrant_id": "q-001",
            "token_count": 10,
        },
        {
            "text": "Deep learning uses neural networks",
            "page_start": 2,
            "page_end": 2,
            "chunk_index": 1,
            "qdrant_id": "q-002",
            "token_count": 6,
        },
        {
            "text": "Supervised learning requires labeled data",
            "page_start": 3,
            "page_end": 3,
            "chunk_index": 2,
            "qdrant_id": "q-003",
            "token_count": 6,
        },
    ]

    # Save chunks
    save_chunks(conn, paper_id, chunks)

    # Search for "machine learning"
    results = search_fts(conn, "machine learning", limit=10)

    assert len(results) > 0
    # The first result should contain the search terms
    assert any("machine" in r["text"].lower() for r in results)

    # Search for "neural"
    results_neural = search_fts(conn, "neural", limit=10)
    assert len(results_neural) > 0
    assert any("neural" in r["text"].lower() for r in results_neural)

    # Search with paper_id filter
    results_filtered = search_fts(conn, "learning", limit=10, paper_id=paper_id)
    assert len(results_filtered) > 0
    assert all(r["paper_id"] == paper_id for r in results_filtered)

    conn.close()


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("arxiv_2410.10071v1.pdf", "2410.10071"),
        ("arxiv_2005.11401v4.pdf", "2005.11401"),
        ("arxiv_2608.27417.pdf", "2608.27417"),
        ("2410.10071.pdf", None),
        ("regular-paper.pdf", None),
        ("arxiv_notanid.pdf", None),
    ],
)
def test_arxiv_id_from_file_name(file_name, expected):
    assert arxiv_id_from_file_name(file_name) == expected


def test_save_paper_derives_arxiv_id(temp_db):
    """save_paper fills arxiv_id from an arxiv_*.pdf file name (#163)."""
    init_db(temp_db)
    conn = get_connection(temp_db)

    arxiv_paper = save_paper(conn, file_name="arxiv_2005.11401v4.pdf", file_hash="h-arxiv")
    other_paper = save_paper(conn, file_name="uploaded.pdf", file_hash="h-other")

    rows = dict(conn.execute("SELECT id, arxiv_id FROM papers").fetchall())
    assert rows[arxiv_paper] == "2005.11401"
    assert rows[other_paper] is None
    conn.close()


def test_init_db_backfills_arxiv_id(temp_db):
    """Re-running init_db backfills arxiv_id for legacy rows (#163)."""
    init_db(temp_db)
    conn = get_connection(temp_db)
    conn.execute(
        "INSERT INTO papers (file_name, file_hash, ingested_at, status, source) "
        "VALUES ('arxiv_2410.10071v1.pdf', 'h-legacy', '2026-01-01T00:00:00', 'indexed', 'arxiv')"
    )
    conn.execute("UPDATE papers SET arxiv_id = NULL")
    conn.commit()
    conn.close()

    init_db(temp_db)

    conn = get_connection(temp_db)
    row = conn.execute("SELECT arxiv_id FROM papers WHERE file_hash = 'h-legacy'").fetchone()
    assert row[0] == "2410.10071"
    conn.close()
