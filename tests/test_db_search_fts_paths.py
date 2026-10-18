"""search_fts: paper_id-filtered and unfiltered paths return the same shape (#745)."""

from academic_paper.db import get_connection, init_db, save_chunks, save_paper, search_fts


def test_search_fts_both_paths_return_same_columns_and_filter(temp_db):
    init_db(temp_db)
    conn = get_connection(temp_db)
    paper_a = save_paper(conn, file_name="a.pdf", file_hash="hash-a", title="A")
    paper_b = save_paper(conn, file_name="b.pdf", file_hash="hash-b", title="B")
    base = {"page_start": 1, "page_end": 1, "token_count": 5}
    save_chunks(conn, paper_a, [{**base, "text": "machine learning alpha", "chunk_index": 0, "qdrant_id": "qa-0"}])
    save_chunks(conn, paper_b, [{**base, "text": "machine learning beta", "chunk_index": 0, "qdrant_id": "qb-0"}])

    unfiltered = search_fts(conn, "machine", limit=10)
    filtered = search_fts(conn, "machine", limit=10, paper_id=paper_b)

    expected_keys = {"chunk_id", "paper_id", "chunk_index", "page_start", "text", "rank"}
    assert len(unfiltered) == 2
    assert {tuple(sorted(r.keys())) for r in unfiltered} == {tuple(sorted(expected_keys))}
    assert len(filtered) == 1
    assert set(filtered[0].keys()) == expected_keys
    assert filtered[0]["paper_id"] == paper_b
    assert filtered[0]["text"] == "machine learning beta"
