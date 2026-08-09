"""SQLite database module for academic paper system."""

import json
import sqlite3
from datetime import datetime


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    """Initialize database schema (idempotent).

    Creates tables: papers, chunks, summaries, and FTS5 virtual table.
    Runs ALTER TABLE migrations for columns added after v1.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            title TEXT,
            authors TEXT,
            year INTEGER,
            pages INTEGER,
            ingested_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
        )
    """)

    # Migration: add columns introduced after v1
    _migrate_add_columns(cursor, "papers", [
        ("categories", "TEXT"),
        ("published_date", "TEXT"),
        ("source", "TEXT"),
    ])

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL REFERENCES papers(id),
            chunk_index INTEGER NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            text TEXT NOT NULL,
            token_count INTEGER,
            qdrant_id TEXT NOT NULL UNIQUE,
            UNIQUE (paper_id, chunk_index)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL UNIQUE REFERENCES papers(id),
            model TEXT NOT NULL,
            objective TEXT,
            method TEXT,
            results TEXT,
            limitations TEXT,
            keywords TEXT,
            raw_json TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61'
        )
    """)

    conn.commit()
    conn.close()


def _migrate_add_columns(cursor: sqlite3.Cursor, table: str, columns: list[tuple[str, str]]) -> None:
    """Add columns to a table if they don't already exist."""
    for col_name, col_def in columns:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError:
            pass  # column already exists


def save_paper(
    conn: sqlite3.Connection,
    file_name: str,
    file_hash: str,
    **kwargs,
) -> int:
    """Save paper to database and return paper_id.

    Args:
        conn: Database connection.
        file_name: Name of the PDF file.
        file_hash: Hash of the file content.
        **kwargs: Optional metadata: title, authors (list), year, pages,
                  categories (list), published_date (str), source (str).

    Returns:
        The paper_id of the saved paper.
    """
    cursor = conn.cursor()
    ingested_at = datetime.utcnow().isoformat()

    authors = kwargs.get("authors")
    authors_json = json.dumps(authors) if authors else None

    categories = kwargs.get("categories")
    categories_json = json.dumps(categories) if categories else None

    cursor.execute("""
        INSERT INTO papers
            (file_name, file_hash, title, authors, year, pages, ingested_at, status,
             categories, published_date, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        file_name,
        file_hash,
        kwargs.get("title"),
        authors_json,
        kwargs.get("year"),
        kwargs.get("pages"),
        ingested_at,
        "pending",
        categories_json,
        kwargs.get("published_date"),
        kwargs.get("source"),
    ))
    conn.commit()
    return cursor.lastrowid


def update_paper_status(conn: sqlite3.Connection, paper_id: int, status: str) -> None:
    """Update paper status."""
    cursor = conn.cursor()
    cursor.execute("UPDATE papers SET status = ? WHERE id = ?", (status, paper_id))
    conn.commit()


def save_chunks(conn: sqlite3.Connection, paper_id: int, chunks: list[dict]) -> None:
    """Save chunks to database and FTS5 index."""
    cursor = conn.cursor()
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO chunks (paper_id, chunk_index, page_start, page_end, text, token_count, qdrant_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_id,
            chunk["chunk_index"],
            chunk["page_start"],
            chunk["page_end"],
            chunk["text"],
            chunk["token_count"],
            chunk["qdrant_id"],
        ))
        chunk_id = cursor.lastrowid
        cursor.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (chunk_id, chunk["text"]))
    conn.commit()


def _deserialize_paper(row) -> dict:
    """Convert a papers row to a dict, deserializing JSON list fields."""
    paper = dict(row)
    if paper.get("authors"):
        paper["authors"] = json.loads(paper["authors"])
    if paper.get("categories"):
        paper["categories"] = json.loads(paper["categories"])
    return paper


def list_papers(conn: sqlite3.Connection) -> list[dict]:
    """Get list of all papers."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers ORDER BY ingested_at DESC")
    return [_deserialize_paper(row) for row in cursor.fetchall()]


def list_papers_filtered(
    conn: sqlite3.Connection,
    limit: int = 20,
    offset: int = 0,
    author: str | None = None,
    category: str | None = None,
) -> tuple[int, list[dict]]:
    """List papers with optional author/category filters and pagination.

    Filters use LIKE substring match against the JSON-serialised list columns.

    Args:
        conn: Database connection.
        limit: Maximum rows to return.
        offset: Rows to skip.
        author: Substring to match against the authors JSON column.
        category: Substring to match against the categories JSON column.

    Returns:
        (total, papers) where total is the unfiltered count.
    """
    cursor = conn.cursor()
    conditions: list[str] = []
    params: list = []

    if author:
        conditions.append("authors LIKE ?")
        params.append(f"%{author}%")
    if category:
        conditions.append("categories LIKE ?")
        params.append(f"%{category}%")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    cursor.execute(f"SELECT COUNT(*) FROM papers {where}", params)
    total = cursor.fetchone()[0]

    cursor.execute(
        f"SELECT * FROM papers {where} ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    )
    papers = [_deserialize_paper(row) for row in cursor.fetchall()]
    return total, papers


def get_paper(conn: sqlite3.Connection, paper_id: int) -> dict | None:
    """Get a specific paper by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()
    return _deserialize_paper(row) if row is not None else None


def get_chunks(conn: sqlite3.Connection, paper_id: int) -> list[dict]:
    """Get all chunks for a paper."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_index", (paper_id,))
    return [dict(row) for row in cursor.fetchall()]


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    paper_id: int | None = None,
) -> list[dict]:
    """Search chunks using FTS5 with BM25 ranking."""
    cursor = conn.cursor()
    if paper_id is None:
        cursor.execute("""
            SELECT c.id as chunk_id, c.paper_id, c.text, bm25(chunks_fts) as rank
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
    else:
        cursor.execute("""
            SELECT c.id as chunk_id, c.paper_id, c.text, bm25(chunks_fts) as rank
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.id
            WHERE chunks_fts MATCH ? AND c.paper_id = ?
            ORDER BY rank
            LIMIT ?
        """, (query, paper_id, limit))
    return [dict(row) for row in cursor.fetchall()]


def get_summary(conn: sqlite3.Connection, paper_id: int) -> dict | None:
    """Get cached summary for a paper."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model, objective, method, results, limitations, keywords, raw_json, created_at
        FROM summaries WHERE paper_id = ?
    """, (paper_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    summary = dict(row)
    if summary["keywords"]:
        summary["keywords"] = json.loads(summary["keywords"])
    if summary["raw_json"]:
        summary["raw_json"] = json.loads(summary["raw_json"])
    return summary


def list_summaries(
    conn: sqlite3.Connection,
    limit: int = 20,
    offset: int = 0,
) -> tuple[int, list[dict]]:
    """List all summaries joined with paper info.

    Args:
        conn: Database connection.
        limit: Maximum rows to return.
        offset: Rows to skip.

    Returns:
        (total, summaries) where total is the count of all summaries.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM summaries")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT
            s.paper_id,
            s.model,
            s.objective,
            s.method,
            s.results,
            s.limitations,
            s.keywords,
            s.created_at,
            p.file_name,
            p.title,
            p.authors,
            p.categories,
            p.published_date,
            p.source
        FROM summaries s
        JOIN papers p ON s.paper_id = p.id
        ORDER BY s.created_at DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    items = []
    for row in cursor.fetchall():
        item = dict(row)
        if item.get("keywords"):
            item["keywords"] = json.loads(item["keywords"])
        if item.get("authors"):
            item["authors"] = json.loads(item["authors"])
        if item.get("categories"):
            item["categories"] = json.loads(item["categories"])
        items.append(item)
    return total, items


def save_summary(conn: sqlite3.Connection, paper_id: int, model: str, summary: dict) -> None:
    """Save or update summary for a paper (upsert)."""
    cursor = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    keywords_json = json.dumps(summary.get("keywords", []))
    raw_json = json.dumps(summary)
    cursor.execute("""
        INSERT INTO summaries (paper_id, model, objective, method, results, limitations, keywords, raw_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            model = excluded.model,
            objective = excluded.objective,
            method = excluded.method,
            results = excluded.results,
            limitations = excluded.limitations,
            keywords = excluded.keywords,
            raw_json = excluded.raw_json,
            created_at = excluded.created_at
    """, (
        paper_id, model,
        summary.get("objective", ""),
        summary.get("method", ""),
        summary.get("results", ""),
        summary.get("limitations", ""),
        keywords_json, raw_json, created_at,
    ))
    conn.commit()
