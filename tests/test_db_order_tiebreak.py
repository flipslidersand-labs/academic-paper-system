"""Pagination order must be total: ties on the sort key break by primary key (#826)."""

from academic_paper.db import get_connection, list_papers_filtered, list_summaries

TS = "2026-01-01T00:00:00+00:00"


def _seed(conn, n=25, scored_every=0):
    """n papers with identical ingested_at; scores all NULL (or ties every k-th)."""
    for i in range(1, n + 1):
        score = 5.0 if scored_every and i % scored_every == 0 else None
        conn.execute(
            "INSERT INTO papers (file_name, file_hash, ingested_at, status, score) VALUES (?, ?, ?, 'indexed', ?)",
            (f"p{i}.pdf", f"h{i}", TS, score),
        )
    conn.commit()


def _walk(conn, page, **kw):
    ids, offset = [], 0
    while True:
        total, rows = list_papers_filtered(conn, limit=page, offset=offset, **kw)
        if not rows:
            return total, ids
        ids += [r["id"] for r in rows]
        offset += page


def test_papers_sort_ingested_at_ties_break_by_id_desc(temp_db):
    conn = get_connection(temp_db)
    _seed(conn)
    total, ids = _walk(conn, 7)
    assert len(ids) == len(set(ids)) == total == 25
    assert ids == sorted(ids, reverse=True)


def test_papers_sort_score_unscored_ties_break_by_id_desc(temp_db):
    conn = get_connection(temp_db)
    _seed(conn, scored_every=5)
    total, ids = _walk(conn, 4, sort="score")
    assert len(ids) == len(set(ids)) == total == 25
    scored = [i for i in ids if i % 5 == 0]
    unscored = [i for i in ids if i % 5 != 0]
    assert ids == scored + unscored
    assert scored == sorted(scored, reverse=True)
    assert unscored == sorted(unscored, reverse=True)


def test_summaries_same_created_at_ties_break_by_paper_id_desc(temp_db):
    conn = get_connection(temp_db)
    _seed(conn, n=12)
    for pid in range(1, 13):
        conn.execute("INSERT INTO summaries (paper_id, model, created_at) VALUES (?, 'm', ?)", (pid, TS))
    conn.commit()
    ids, offset = [], 0
    while True:
        _, rows = list_summaries(conn, limit=5, offset=offset)
        if not rows:
            break
        ids += [r["paper_id"] for r in rows]
        offset += 5
    assert ids == list(range(12, 0, -1))
