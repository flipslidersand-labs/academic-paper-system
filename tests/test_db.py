"""Tests for academic_paper.db module."""

import json
import sqlite3

import pytest

from academic_paper.db import (
    _migrate_add_columns,
    arxiv_id_from_file_name,
    get_chunks,
    get_connection,
    init_db,
    list_papers_filtered,
    save_chunks,
    save_paper,
    save_summary,
    search_fts,
    upsert_job,
)


def test_migrate_add_columns_ignores_duplicate_column(temp_db):
    """Re-adding an existing column should be silently ignored (idempotent migration)."""
    conn = get_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")

    # Should not raise even though "name" already exists.
    _migrate_add_columns(cursor, "t", [("name", "TEXT")])
    conn.close()


def test_migrate_add_columns_reraises_real_errors(temp_db):
    """A genuine OperationalError (e.g. missing table) must not be swallowed."""
    conn = get_connection(temp_db)
    cursor = conn.cursor()

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        _migrate_add_columns(cursor, "nonexistent_table", [("col", "TEXT")])
    conn.close()


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


def _jobs_columns(db_path):
    conn = get_connection(db_path)
    try:
        return {r[1]: r for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    finally:
        conn.close()


def test_init_db_migrates_legacy_jobs_table_without_kind(tmp_path):
    """A pre-kind jobs table gains kind (NOT NULL DEFAULT '') and result (#476)."""
    temp_db = str(tmp_path / "legacy.db")
    conn = get_connection(temp_db)
    conn.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending', "
        "total INTEGER NOT NULL DEFAULT 0, processed INTEGER NOT NULL DEFAULT 0, "
        "failed INTEGER NOT NULL DEFAULT 0, errors TEXT NOT NULL DEFAULT '[]', "
        "started_at REAL NOT NULL, finished_at REAL)"
    )
    conn.execute("INSERT INTO jobs (id, started_at) VALUES ('old', 1.0)")
    conn.commit()
    conn.close()

    init_db(temp_db)

    cols = _jobs_columns(temp_db)
    assert "kind" in cols and "result" in cols
    conn = get_connection(temp_db)
    assert conn.execute("SELECT kind FROM jobs WHERE id='old'").fetchone()[0] == ""
    conn.close()


def test_init_db_keeps_existing_jobs_kind_column_idempotent(temp_db):
    """init_db twice on a DB that already has kind must not fail or duplicate (#476)."""
    init_db(temp_db)
    init_db(temp_db)
    assert {"kind", "result"} <= set(_jobs_columns(temp_db))


# --- #309: failed writes must not leave an open transaction -----------------


def _chunk(idx: int, qdrant_id: str) -> dict:
    return {
        "chunk_index": idx,
        "page_start": 1,
        "page_end": 1,
        "text": f"text {idx}",
        "token_count": 2,
        "qdrant_id": qdrant_id,
    }


def _fail_save_chunks(conn):
    pid = save_paper(conn, "a.pdf", "hash-a")
    with pytest.raises(sqlite3.IntegrityError):
        # Second chunk violates UNIQUE(qdrant_id) after the first was inserted.
        save_chunks(conn, pid, [_chunk(0, "same"), _chunk(1, "same")])
    return pid


def _fail_save_paper(conn):
    save_paper(conn, "a.pdf", "dup-hash")
    with pytest.raises(sqlite3.IntegrityError):
        save_paper(conn, "b.pdf", "dup-hash")


def _fail_save_summary(conn):
    with pytest.raises(sqlite3.IntegrityError):
        save_summary(conn, 9999, "m", {"objective": "x"})  # FK violation


def _fail_upsert_job(conn):
    with pytest.raises(sqlite3.IntegrityError):
        upsert_job(conn, "j1", None, 0, 0, 0, [], 0.0, None)  # status NOT NULL


@pytest.mark.parametrize("fail", [_fail_save_chunks, _fail_save_paper, _fail_save_summary, _fail_upsert_job])
def test_failed_write_rolls_back_open_transaction(temp_db, fail):
    conn = get_connection(temp_db)
    try:
        fail(conn)
        assert not conn.in_transaction
    finally:
        conn.close()


def test_failed_save_chunks_leaves_no_partial_rows(temp_db):
    conn = get_connection(temp_db)
    try:
        pid = _fail_save_chunks(conn)
        assert get_chunks(conn, pid) == []
    finally:
        conn.close()


def test_failed_write_does_not_hold_write_lock_for_other_connections(temp_db):
    """A failed save_* followed by list_papers_filtered must not pin the write lock."""
    conn = get_connection(temp_db)
    other = get_connection(temp_db)
    other.execute("PRAGMA busy_timeout = 100")
    try:
        _fail_save_chunks(conn)
        list_papers_filtered(conn)
        # Would raise "database is locked" if conn kept its failed transaction.
        save_paper(other, "c.pdf", "hash-c")
    finally:
        conn.close()
        other.close()


# --- summaries columns derived from PaperSummary (#530) ---


def test_summaries_ddl_matches_paper_summary_fields(temp_db):
    """DDL is hand-written; fail loudly if it drifts from PaperSummary.model_fields."""
    from academic_paper.db import SUMMARY_COLUMNS

    init_db(temp_db)
    conn = get_connection(temp_db)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(summaries)")}
    conn.close()
    assert set(SUMMARY_COLUMNS) <= cols
    assert cols == set(SUMMARY_COLUMNS) | {"id", "paper_id", "model"}


def test_save_summary_roundtrip_get_and_list(temp_db):
    from academic_paper.db import get_summary, list_summaries, save_summary

    init_db(temp_db)
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "a.pdf", "hash-a", title="T")
    s1 = {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k1", "k2"]}
    save_summary(conn, paper_id, "mdl", s1)

    got = get_summary(conn, paper_id)
    assert got["model"] == "mdl"
    assert (got["objective"], got["method"], got["results"], got["limitations"]) == ("o", "m", "r", "l")
    assert got["keywords"] == ["k1", "k2"]
    assert got["raw_json"] == s1

    total, items = list_summaries(conn)
    assert total == 1
    assert items[0]["objective"] == "o" and items[0]["keywords"] == ["k1", "k2"]
    assert items[0]["title"] == "T"

    # upsert overwrites; missing fields default to ""
    save_summary(conn, paper_id, "mdl2", {"objective": "new"})
    got = get_summary(conn, paper_id)
    assert got["model"] == "mdl2" and got["objective"] == "new" and got["method"] == ""
    assert got["keywords"] is not None
    conn.close()
