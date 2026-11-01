"""Unit tests for academic_paper.services.search_service, run without booting the FastAPI app (#357)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from academic_paper.db import get_connection, save_chunks, save_paper
from academic_paper.services.search_service import _fetch_chunk_meta, run_search


def test_fetch_chunk_meta_empty_ids_skips_query_without_erroring(temp_db):
    """SQLite rejects `IN ()`; the helper must short-circuit before building that SQL."""
    conn = get_connection(temp_db)
    try:
        assert _fetch_chunk_meta(conn.cursor(), [], "id", "id, page_start") == {}
    finally:
        conn.close()


def test_fetch_chunk_meta_keys_rows_by_id_column(temp_db):
    conn = get_connection(temp_db)
    try:
        paper_id = save_paper(conn, "p.pdf", "h1")
        save_chunks(
            conn,
            paper_id,
            [{"text": "a", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": "q-0", "token_count": 1}],
        )
        row = conn.execute("SELECT id FROM chunks").fetchone()
        chunk_id = row["id"]

        meta = _fetch_chunk_meta(conn.cursor(), [chunk_id], "id", "id, page_start")

        assert set(meta) == {chunk_id}
        assert meta[chunk_id]["page_start"] == 1
    finally:
        conn.close()


def _make_embedder(vector=None):
    embedder = MagicMock()
    embedder.embed_single = AsyncMock(return_value=vector or [0.1] * 768)
    embedder.embed = AsyncMock(return_value=[])
    return embedder


def _make_vector_store(search_results=None):
    store = MagicMock()
    store.asearch = AsyncMock(return_value=search_results or [])
    return store


@pytest.mark.anyio
async def test_run_search_keyword_mode_returns_fts_results_without_touching_embedder(temp_db):
    conn = get_connection(temp_db)
    try:
        paper_id = save_paper(conn, "p.pdf", "h1")
        save_chunks(
            conn,
            paper_id,
            [
                {
                    "text": "deep learning survey",
                    "page_start": 3,
                    "page_end": 3,
                    "chunk_index": 0,
                    "qdrant_id": "q-0",
                    "token_count": 3,
                }
            ],
        )
        embedder = _make_embedder()
        vector_store = _make_vector_store()

        result = await run_search(
            temp_db,
            embedder,
            vector_store,
            q="deep learning",
            mode="keyword",
            limit=10,
            paper_id=None,
            snippet_length=200,
            nuggets_per_chunk=3,
            nugget_embed_weight=0.7,
        )

        assert result["mode"] == "keyword"
        assert len(result["results"]) == 1
        assert result["results"][0]["page_start"] == 3
        embedder.embed_single.assert_not_called()
    finally:
        conn.close()


@pytest.mark.anyio
async def test_run_search_vector_mode_skips_orphan_payload(temp_db):
    conn = get_connection(temp_db)
    try:
        embedder = _make_embedder()
        vector_store = _make_vector_store(
            search_results=[
                {"id": "orphan", "score": 0.9, "payload": {"text": "no paper id"}},
                {"id": "ok", "score": 0.8, "payload": {"paper_id": 1, "chunk_index": 2, "text": "complete"}},
            ]
        )

        result = await run_search(
            temp_db,
            embedder,
            vector_store,
            q="q",
            mode="vector",
            limit=10,
            paper_id=None,
            snippet_length=200,
            nuggets_per_chunk=3,
            nugget_embed_weight=0.7,
        )

        assert [r["paper_id"] for r in result["results"]] == [1]
        embedder.embed_single.assert_awaited_once_with("q", mode="search")
    finally:
        conn.close()


@pytest.mark.anyio
async def test_run_search_hybrid_mode_merges_fts_and_vector(temp_db):
    conn = get_connection(temp_db)
    try:
        paper_id = save_paper(conn, "p.pdf", "h1")
        save_chunks(
            conn,
            paper_id,
            [
                {
                    "text": "hybrid retrieval paper",
                    "page_start": 1,
                    "page_end": 1,
                    "chunk_index": 0,
                    "qdrant_id": "q-0",
                    "token_count": 3,
                }
            ],
        )
        embedder = _make_embedder()
        vector_store = _make_vector_store(search_results=[])

        result = await run_search(
            temp_db,
            embedder,
            vector_store,
            q="hybrid retrieval",
            mode="hybrid",
            limit=10,
            paper_id=None,
            snippet_length=200,
            nuggets_per_chunk=3,
            nugget_embed_weight=0.7,
        )

        assert result["mode"] == "hybrid"
        assert len(result["results"]) == 1
        assert result["results"][0]["paper_id"] == paper_id
    finally:
        conn.close()


# --- mode dispatch and shared row shape (#677) -----------------------------------

_ROW_KEYS = ["rank", "score", "paper_id", "chunk_index", "page_start", "snippet"]


@pytest.mark.anyio
async def test_run_search_unknown_mode_raises_instead_of_falling_through_to_hybrid(temp_db):
    conn = get_connection(temp_db)
    try:
        with pytest.raises(ValueError, match="unknown search mode"):
            await run_search(
                temp_db,
                _make_embedder(),
                _make_vector_store(),
                q="x",
                mode="bogus",
                limit=5,
                paper_id=None,
                snippet_length=10,
                nuggets_per_chunk=3,
                nugget_embed_weight=0.7,
            )
    finally:
        conn.close()


def test_search_modes_are_the_single_source_for_the_endpoint_pattern():
    import re

    from academic_paper.server import app
    from academic_paper.services.search_service import SEARCH_MODES

    assert sorted(SEARCH_MODES) == ["hybrid", "keyword", "nugget", "vector"]
    search_route = next(r for r in app.routes if getattr(r, "path", "") == "/search")
    pattern = next(p.field_info.metadata[0].pattern for p in search_route.dependant.query_params if p.name == "mode")
    assert all(re.match(pattern, m) for m in SEARCH_MODES)
    assert not re.match(pattern, "bogus")


def test_result_row_has_one_shape_and_key_order():
    from academic_paper.services.search_service import _make_snippet, _result_row

    row = _result_row(rank=1, score=0.5, paper_id=2, chunk_index=3, page_start=None, snippet="s")
    assert list(row) == _ROW_KEYS
    assert _make_snippet("abcdef", 3) == "abc"
    assert _make_snippet("abcdef", 0) == "abcdef"


# --- DB reads run off the event loop (#678) --------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid"])
async def test_run_search_db_reads_run_off_the_event_loop_thread(temp_db, monkeypatch, mode):
    import threading

    from academic_paper.services import search_service

    conn = get_connection(temp_db)
    try:
        paper_id = save_paper(conn, "p.pdf", "h1")
        save_chunks(
            conn,
            paper_id,
            [
                {
                    "text": "deep learning",
                    "page_start": 1,
                    "page_end": 1,
                    "chunk_index": 0,
                    "qdrant_id": "q-0",
                    "token_count": 2,
                }
            ],
        )
    finally:
        conn.close()

    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def _wrap(name):
        orig = getattr(search_service, name)

        def wrapper(*a, **kw):
            seen[name] = threading.get_ident()
            return orig(*a, **kw)

        monkeypatch.setattr(search_service, name, wrapper)

    _wrap("search_fts")
    _wrap("_fetch_chunk_meta")
    hits = [{"id": "q-0", "score": 0.9, "payload": {"paper_id": paper_id, "text": "deep learning", "chunk_index": 0}}]
    await run_search(
        temp_db,
        _make_embedder(),
        _make_vector_store(hits),
        q="deep learning",
        mode=mode,
        limit=5,
        paper_id=None,
        snippet_length=50,
        nuggets_per_chunk=3,
        nugget_embed_weight=0.7,
    )

    assert seen, "the DB read helpers were never called"
    assert all(tid != loop_thread for tid in seen.values()), seen
