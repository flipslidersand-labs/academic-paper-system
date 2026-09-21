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
            conn,
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
            conn,
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
            conn,
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
