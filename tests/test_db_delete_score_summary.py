"""Direct tests for db.delete_paper (#661) and score/summary helpers (#665)."""

import pytest

from academic_paper.db import (
    delete_paper,
    get_all_papers_for_scoring,
    get_connection,
    get_summary,
    list_papers_filtered,
    save_chunks,
    save_paper,
    save_summary,
    search_fts,
    update_paper_score,
)


def _chunk(i, text):
    return {
        "chunk_index": i,
        "page_start": 1,
        "page_end": 1,
        "text": text,
        "token_count": 3,
        "qdrant_id": f"q{i}-{text}",
    }


@pytest.fixture
def conn(temp_db):
    c = get_connection(temp_db)
    yield c
    c.close()


def _count(conn, table, col, pid):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (pid,)).fetchone()[0]


def test_delete_paper_removes_fts_rows_and_keeps_other_paper(conn):
    p1 = save_paper(conn, "a.pdf", "h1", title="A")
    p2 = save_paper(conn, "b.pdf", "h2", title="B")
    save_chunks(conn, p1, [_chunk(0, "zebra quantum alpha")])
    save_chunks(conn, p2, [_chunk(0, "zebra quantum beta")])
    save_summary(conn, p1, "m", {"objective": "o"})
    save_summary(conn, p2, "m", {"objective": "o2"})
    assert {r["paper_id"] for r in search_fts(conn, "zebra")} == {p1, p2}

    delete_paper(conn, p1)

    # FTS no longer returns the deleted paper's chunk; the other is intact.
    hits = search_fts(conn, "zebra")
    assert [h["paper_id"] for h in hits] == [p2]
    assert search_fts(conn, "zebra", paper_id=p1) == []
    assert _count(conn, "chunks", "paper_id", p1) == 0
    assert _count(conn, "summaries", "paper_id", p1) == 0
    assert _count(conn, "papers", "id", p1) == 0
    assert _count(conn, "chunks", "paper_id", p2) == 1
    assert _count(conn, "summaries", "paper_id", p2) == 1
    assert _count(conn, "papers", "id", p2) == 1
    # FTS index itself holds only paper2's row (no orphaned rowid).
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts_docsize").fetchone()[0] == 1


def test_delete_paper_nonexistent_id_is_noop(conn):
    p = save_paper(conn, "a.pdf", "h1")
    save_chunks(conn, p, [_chunk(0, "keepme token")])
    delete_paper(conn, 99999)  # must not raise
    assert len(search_fts(conn, "keepme")) == 1
    assert _count(conn, "papers", "id", p) == 1


def test_delete_paper_without_chunks(conn):
    p = save_paper(conn, "a.pdf", "h1")
    delete_paper(conn, p)
    assert _count(conn, "papers", "id", p) == 0


def test_list_papers_sort_score_orders_desc_with_zero_before_null(conn):
    hi = save_paper(conn, "hi.pdf", "h1")
    zero = save_paper(conn, "zero.pdf", "h2")
    null = save_paper(conn, "null.pdf", "h3")
    mid = save_paper(conn, "mid.pdf", "h4")
    update_paper_score(conn, hi, 0.9)
    update_paper_score(conn, mid, 0.5)
    update_paper_score(conn, zero, 0.0)

    total, papers = list_papers_filtered(conn, sort="score")

    assert total == 4
    assert [p["id"] for p in papers] == [hi, mid, zero, null]
    assert papers[-1]["score"] is None
    assert papers[2]["score"] == 0.0


def test_save_summary_upsert_keeps_single_row_with_new_values(conn):
    p = save_paper(conn, "a.pdf", "h1")
    save_summary(conn, p, "model-1", {"objective": "old", "keywords": ["a"]})
    save_summary(conn, p, "model-2", {"objective": "new", "method": "m", "keywords": ["x", "y"]})

    assert _count(conn, "summaries", "paper_id", p) == 1
    s = get_summary(conn, p)
    assert s["model"] == "model-2"
    assert s["objective"] == "new"
    assert s["method"] == "m"
    assert s["keywords"] == ["x", "y"]
    assert isinstance(s["keywords"], list)


def test_get_summary_missing_returns_none(conn):
    p = save_paper(conn, "a.pdf", "h1")
    assert get_summary(conn, p) is None


def test_get_all_papers_for_scoring_decodes_categories(conn):
    p1 = save_paper(conn, "a.pdf", "h1", categories=["cs.AI", "cs.LG"], published_date="2025-01-01")
    p2 = save_paper(conn, "b.pdf", "h2")
    rows = {r["id"]: r for r in get_all_papers_for_scoring(conn)}
    assert rows[p1]["categories"] == ["cs.AI", "cs.LG"]
    assert rows[p1]["published_date"] == "2025-01-01"
    assert not rows[p2]["categories"]  # NULL stays falsy, not decoded
    assert set(rows[p1]) == {"id", "categories", "published_date"}


def test_update_paper_score_persists(conn):
    p = save_paper(conn, "a.pdf", "h1")
    update_paper_score(conn, p, 0.25)
    assert conn.execute("SELECT score FROM papers WHERE id = ?", (p,)).fetchone()[0] == 0.25
