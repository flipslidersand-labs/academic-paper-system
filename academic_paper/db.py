"""SQLite database module for academic paper system."""

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get SQLite connection with foreign keys, WAL, and busy timeout enabled.

    WAL lets readers proceed while a writer holds the lock, and busy_timeout
    makes concurrent writers wait instead of failing immediately with
    "database is locked" — important now that background ingest jobs hold
    connections across awaits.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def db_connection(db_path: str):
    """Context manager that opens a connection and guarantees close on exit."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Initialize database schema (idempotent).

    Creates tables: papers, chunks, summaries, jobs, and FTS5 virtual table.
    Runs ALTER TABLE migrations for columns added after v1.
    """
    with db_connection(db_path) as conn:
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
        _migrate_add_columns(
            cursor,
            "papers",
            [
                ("categories", "TEXT"),
                ("published_date", "TEXT"),
                ("source", "TEXT"),
                ("score", "REAL"),
                ("arxiv_id", "TEXT"),
            ],
        )

        # Backfill arxiv_id for rows ingested before the column existed (#163).
        for row_id, file_name in cursor.execute(
            "SELECT id, file_name FROM papers WHERE arxiv_id IS NULL AND file_name LIKE 'arxiv\\_%' ESCAPE '\\'"
        ).fetchall():
            arxiv_id = arxiv_id_from_file_name(file_name)
            if arxiv_id:
                cursor.execute("UPDATE papers SET arxiv_id = ? WHERE id = ?", (arxiv_id, row_id))

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
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                kind TEXT NOT NULL DEFAULT '',
                total INTEGER NOT NULL DEFAULT 0,
                processed INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                errors TEXT NOT NULL DEFAULT '[]',
                started_at REAL NOT NULL,
                finished_at REAL
            )
        """)

        # Migration for databases created before the kind column existed.
        cursor.execute("PRAGMA table_info(jobs)")
        if "kind" not in [row[1] for row in cursor.fetchall()]:
            cursor.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT ''")

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61'
            )
        """)

        conn.commit()


_ARXIV_FILE_NAME_RE = re.compile(r"^arxiv_(\d{4}\.\d{4,5})(?:v\d+)?\.pdf$")


def arxiv_id_from_file_name(file_name: str) -> str | None:
    """Extract the bare arXiv ID (no version) from an `arxiv_<id>vX.pdf` file name."""
    m = _ARXIV_FILE_NAME_RE.match(file_name)
    return m.group(1) if m else None


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
                  categories (list), published_date (str), source (str),
                  arxiv_id (str).

    Returns:
        The paper_id of the saved paper.
    """
    cursor = conn.cursor()
    ingested_at = datetime.now(UTC).isoformat()

    authors = kwargs.get("authors")
    authors_json = json.dumps(authors) if authors else None

    categories = kwargs.get("categories")
    categories_json = json.dumps(categories) if categories else None

    cursor.execute(
        """
        INSERT INTO papers
            (file_name, file_hash, title, authors, year, pages, ingested_at, status,
             categories, published_date, source, arxiv_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
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
            kwargs.get("arxiv_id") or arxiv_id_from_file_name(file_name),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def update_paper_status(conn: sqlite3.Connection, paper_id: int, status: str) -> None:
    """Update paper status."""
    cursor = conn.cursor()
    cursor.execute("UPDATE papers SET status = ? WHERE id = ?", (status, paper_id))
    conn.commit()


def delete_paper(conn: sqlite3.Connection, paper_id: int) -> None:
    """Delete a paper and its chunks/summary by id (#145).

    chunks_fts is an FTS5 external-content table (content='chunks').  SQLite
    does NOT auto-update FTS indexes when the content table changes, so we
    must delete the FTS rows explicitly before removing the chunks rows (#188).
    """
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE paper_id = ?)",
        (paper_id,),
    )
    cursor.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
    cursor.execute("DELETE FROM summaries WHERE paper_id = ?", (paper_id,))
    cursor.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    conn.commit()


def update_paper_score(conn: sqlite3.Connection, paper_id: int, score: float) -> None:
    """Update computed relevance score for a paper."""
    cursor = conn.cursor()
    cursor.execute("UPDATE papers SET score = ? WHERE id = ?", (score, paper_id))
    conn.commit()


def save_chunks(conn: sqlite3.Connection, paper_id: int, chunks: list[dict]) -> None:
    """Save chunks to database and FTS5 index."""
    cursor = conn.cursor()
    for chunk in chunks:
        cursor.execute(
            """
            INSERT INTO chunks (paper_id, chunk_index, page_start, page_end, text, token_count, qdrant_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                paper_id,
                chunk["chunk_index"],
                chunk["page_start"],
                chunk["page_end"],
                chunk["text"],
                chunk["token_count"],
                chunk["qdrant_id"],
            ),
        )
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
    sort: str = "ingested_at",
) -> tuple[int, list[dict]]:
    """List papers with optional author/category filters, pagination, and sort.

    Args:
        conn: Database connection.
        limit: Maximum rows to return.
        offset: Rows to skip.
        author: Substring to match against the authors JSON column.
        category: Substring to match against the categories JSON column.
        sort: Sort field — 'ingested_at' (default) or 'score' (highest first,
              unscored papers last).

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

    order_by = "COALESCE(score, -1) DESC" if sort == "score" else "ingested_at DESC"

    cursor.execute(f"SELECT COUNT(*) FROM papers {where}", params)
    total = cursor.fetchone()[0]

    cursor.execute(
        f"SELECT * FROM papers {where} ORDER BY {order_by} LIMIT ? OFFSET ?",
        params + [limit, offset],
    )
    papers = [_deserialize_paper(row) for row in cursor.fetchall()]
    return total, papers


def get_all_papers_for_scoring(conn: sqlite3.Connection) -> list[dict]:
    """Fetch minimal paper data needed for score computation."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, categories, published_date FROM papers")
    rows = []
    for row in cursor.fetchall():
        d = dict(row)
        if d.get("categories"):
            d["categories"] = json.loads(d["categories"])
        rows.append(d)
    return rows


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


def escape_fts5_query(query: str) -> str:
    """Quote each whitespace-separated token as an FTS5 string literal.

    Wrapping every token in double quotes (with any internal ``"`` doubled)
    makes FTS5 treat it as a literal term, so user input can't inject
    operators (AND/OR/NEAR/``*``, ``column:`` filters) or trigger fts5 syntax
    errors via unbalanced quotes. Multi-word input keeps its implicit-AND
    semantics. Returns ``'""'`` (matches nothing) for blank input.
    """
    tokens = query.split()
    if not tokens:
        return '""'
    return " ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    paper_id: int | None = None,
) -> list[dict]:
    """Search chunks using FTS5 with BM25 ranking.

    The query is escaped via :func:`escape_fts5_query` before being passed to
    ``MATCH`` so arbitrary user input is treated as literal terms.
    """
    match_query = escape_fts5_query(query)
    cursor = conn.cursor()
    if paper_id is None:
        cursor.execute(
            """
            SELECT c.id as chunk_id, c.paper_id, c.text, bm25(chunks_fts) as rank
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """,
            (match_query, limit),
        )
    else:
        cursor.execute(
            """
            SELECT c.id as chunk_id, c.paper_id, c.text, bm25(chunks_fts) as rank
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.id
            WHERE chunks_fts MATCH ? AND c.paper_id = ?
            ORDER BY rank
            LIMIT ?
        """,
            (match_query, paper_id, limit),
        )
    return [dict(row) for row in cursor.fetchall()]


def get_summary(conn: sqlite3.Connection, paper_id: int) -> dict | None:
    """Get cached summary for a paper."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT model, objective, method, results, limitations, keywords, raw_json, created_at
        FROM summaries WHERE paper_id = ?
    """,
        (paper_id,),
    )
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

    cursor.execute(
        """
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
    """,
        (limit, offset),
    )

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
    created_at = datetime.now(UTC).isoformat()
    keywords_json = json.dumps(summary.get("keywords", []))
    raw_json = json.dumps(summary)
    cursor.execute(
        """
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
    """,
        (
            paper_id,
            model,
            summary.get("objective", ""),
            summary.get("method", ""),
            summary.get("results", ""),
            summary.get("limitations", ""),
            keywords_json,
            raw_json,
            created_at,
        ),
    )
    conn.commit()


def upsert_job(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    total: int,
    processed: int,
    failed: int,
    errors: list[str],
    started_at: float,
    finished_at: float | None,
    kind: str = "",
) -> None:
    """Insert or update a job row."""
    conn.execute(
        """
        INSERT INTO jobs (id, status, kind, total, processed, failed, errors, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status = excluded.status,
            kind = excluded.kind,
            total = excluded.total,
            processed = excluded.processed,
            failed = excluded.failed,
            errors = excluded.errors,
            finished_at = excluded.finished_at
    """,
        (job_id, status, kind, total, processed, failed, json.dumps(errors), started_at, finished_at),
    )
    conn.commit()


def load_all_jobs(conn: sqlite3.Connection) -> list[dict]:
    """Load all job rows ordered by start time (oldest first)."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs ORDER BY started_at ASC")
    rows = []
    for row in cursor.fetchall():
        d = dict(row)
        d["errors"] = json.loads(d["errors"])
        rows.append(d)
    return rows
