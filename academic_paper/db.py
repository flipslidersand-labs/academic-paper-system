"""SQLite database module for academic paper system."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get SQLite connection with foreign keys enabled.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection object.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
    return conn


def init_db(db_path: str) -> None:
    """Initialize database schema (idempotent).

    Creates tables: papers, chunks, summaries, and FTS5 virtual table.

    Args:
        db_path: Path to the SQLite database file.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Create papers table
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

    # Create chunks table
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

    # Create summaries table
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

    # Create FTS5 virtual table
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


def save_paper(
    conn: sqlite3.Connection,
    file_name: str,
    file_hash: str,
    **kwargs
) -> int:
    """Save paper to database and return paper_id.

    Args:
        conn: Database connection.
        file_name: Name of the PDF file.
        file_hash: Hash of the file content.
        **kwargs: Optional fields (title, authors, year, pages).
                 authors should be a list, will be serialized to JSON.

    Returns:
        The paper_id of the saved paper.
    """
    cursor = conn.cursor()

    # Prepare fields
    title = kwargs.get("title")
    authors = kwargs.get("authors")
    year = kwargs.get("year")
    pages = kwargs.get("pages")
    ingested_at = datetime.utcnow().isoformat()
    status = "pending"

    # Serialize authors if provided
    authors_json = None
    if authors:
        authors_json = json.dumps(authors)

    cursor.execute("""
        INSERT INTO papers (file_name, file_hash, title, authors, year, pages, ingested_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (file_name, file_hash, title, authors_json, year, pages, ingested_at, status))

    conn.commit()
    return cursor.lastrowid


def update_paper_status(conn: sqlite3.Connection, paper_id: int, status: str) -> None:
    """Update paper status.

    Args:
        conn: Database connection.
        paper_id: ID of the paper to update.
        status: New status ('pending', 'indexed', 'failed', etc.).
    """
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE papers SET status = ? WHERE id = ?
    """, (status, paper_id))
    conn.commit()


def save_chunks(
    conn: sqlite3.Connection,
    paper_id: int,
    chunks: list[dict]
) -> None:
    """Save chunks to database and FTS5 index.

    Args:
        conn: Database connection.
        paper_id: ID of the paper.
        chunks: List of chunk dictionaries with keys:
               - text: Chunk text
               - page_start: Starting page number
               - page_end: Ending page number
               - chunk_index: Index of chunk in paper
               - qdrant_id: ID in Qdrant vector store
               - token_count: Number of tokens in chunk
    """
    cursor = conn.cursor()

    for chunk in chunks:
        # Insert into chunks table
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
            chunk["qdrant_id"]
        ))

        chunk_id = cursor.lastrowid

        # Insert into FTS5
        cursor.execute("""
            INSERT INTO chunks_fts(rowid, text)
            VALUES (?, ?)
        """, (chunk_id, chunk["text"]))

    conn.commit()


def list_papers(conn: sqlite3.Connection) -> list[dict]:
    """Get list of all papers.

    Args:
        conn: Database connection.

    Returns:
        List of paper dictionaries.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers ORDER BY ingested_at DESC")
    rows = cursor.fetchall()

    papers = []
    for row in rows:
        paper = dict(row)
        # Deserialize authors JSON if present
        if paper["authors"]:
            paper["authors"] = json.loads(paper["authors"])
        papers.append(paper)

    return papers


def get_paper(conn: sqlite3.Connection, paper_id: int) -> dict | None:
    """Get a specific paper by ID.

    Args:
        conn: Database connection.
        paper_id: ID of the paper to retrieve.

    Returns:
        Paper dictionary or None if not found.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()

    if row is None:
        return None

    paper = dict(row)
    # Deserialize authors JSON if present
    if paper["authors"]:
        paper["authors"] = json.loads(paper["authors"])

    return paper


def get_chunks(conn: sqlite3.Connection, paper_id: int) -> list[dict]:
    """Get all chunks for a paper.

    Args:
        conn: Database connection.
        paper_id: ID of the paper.

    Returns:
        List of chunk dictionaries.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_index
    """, (paper_id,))
    rows = cursor.fetchall()

    return [dict(row) for row in rows]


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    paper_id: int | None = None
) -> list[dict]:
    """Search chunks using FTS5 with BM25 ranking.

    Args:
        conn: Database connection.
        query: Search query string.
        limit: Maximum number of results to return.
        paper_id: Optional paper ID to limit search to a single paper.

    Returns:
        List of result dictionaries with keys:
        - chunk_id: ID in chunks table
        - paper_id: ID of paper
        - text: Chunk text
        - rank: BM25 ranking score
    """
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

    rows = cursor.fetchall()
    return [dict(row) for row in rows]


def get_summary(conn: sqlite3.Connection, paper_id: int) -> dict | None:
    """Get cached summary for a paper.

    Args:
        conn: Database connection.
        paper_id: ID of the paper.

    Returns:
        Summary dictionary or None if not cached.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model, objective, method, results, limitations, keywords, raw_json, created_at
        FROM summaries
        WHERE paper_id = ?
    """, (paper_id,))
    row = cursor.fetchone()

    if row is None:
        return None

    summary = dict(row)
    # Deserialize keywords JSON if present
    if summary["keywords"]:
        summary["keywords"] = json.loads(summary["keywords"])
    # Deserialize raw_json if present for debugging
    if summary["raw_json"]:
        summary["raw_json"] = json.loads(summary["raw_json"])

    return summary


def save_summary(conn: sqlite3.Connection, paper_id: int, model: str, summary: dict) -> None:
    """Save or update summary for a paper (upsert).

    Args:
        conn: Database connection.
        paper_id: ID of the paper.
        model: Name of the LLM model used.
        summary: Summary dictionary with keys:
                - objective: str
                - method: str
                - results: str
                - limitations: str
                - keywords: list[str]
    """
    cursor = conn.cursor()
    created_at = datetime.utcnow().isoformat()

    # Serialize keywords to JSON
    keywords_json = json.dumps(summary.get("keywords", []))

    # Store raw summary as JSON for debugging
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
        paper_id,
        model,
        summary.get("objective", ""),
        summary.get("method", ""),
        summary.get("results", ""),
        summary.get("limitations", ""),
        keywords_json,
        raw_json,
        created_at
    ))

    conn.commit()
