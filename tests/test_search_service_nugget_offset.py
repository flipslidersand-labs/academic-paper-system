"""nugget-mode batch embedding and per-chunk vector re-slicing in run_search (#492)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from academic_paper.db import get_connection, save_chunks, save_paper
from academic_paper.services.search_service import run_search


def _hit(qid, chunk_id, text):
    return {
        "id": qid,
        "score": 0.9,
        # Real ingest payload shape (4 keys, no chunk_id); chunk_id is resolved via chunks.qdrant_id.
        "payload": {"paper_id": 1, "chunk_index": chunk_id, "text": text, "file_name": "t.pdf"},
    }


async def _search(temp_db, hits, embed_return):
    conn = get_connection(temp_db)
    try:
        # Production payloads carry no chunk_id; run_search resolves chunks via chunks.qdrant_id
        # and drops points with no chunks row as orphans (#497), so seed a row per hit.
        paper_id = save_paper(conn, "t.pdf", "h")
        save_chunks(
            conn,
            paper_id,
            [
                {
                    "chunk_index": i,
                    "page_start": 1,
                    "page_end": 1,
                    "text": h["payload"]["text"],
                    "token_count": 1,
                    "qdrant_id": h["id"],
                }
                for i, h in enumerate(hits)
            ],
        )
        embedder = MagicMock()
        embedder.embed_single = AsyncMock(return_value=[9.0])
        embedder.embed = AsyncMock(return_value=embed_return)
        store = MagicMock()
        store.asearch = AsyncMock(return_value=hits)
        captured = []

        def fake_extract(q, text, **kwargs):
            captured.append((text, kwargs["sentence_vecs"]))
            return "snippet"

        with patch("academic_paper.services.search_service.extract_nuggets", side_effect=fake_extract):
            result = await run_search(
                conn,
                embedder,
                store,
                q="alpha",
                mode="nugget",
                limit=10,
                paper_id=None,
                snippet_length=200,
                nuggets_per_chunk=3,
                nugget_embed_weight=0.7,
            )
        return embedder, captured, result
    finally:
        conn.close()


@pytest.mark.anyio
async def test_nugget_empty_chunk_gets_none_and_offsets_stay_aligned(temp_db):
    hits = [
        _hit("a", 1, "One. Two."),  # 2 sentences
        _hit("b", 2, ""),  # 0 sentences
        _hit("c", 3, "Three. Four. Five."),  # 3 sentences
    ]
    vecs = [[1.0], [2.0], [3.0], [4.0], [5.0]]

    embedder, captured, result = await _search(temp_db, hits, vecs)

    embedder.embed.assert_awaited_once()
    assert embedder.embed.await_args.args[0] == ["One.", "Two.", "Three.", "Four.", "Five."]
    by_text = dict(captured)
    assert by_text["One. Two."] == [[1.0], [2.0]]
    assert by_text[""] is None
    assert by_text["Three. Four. Five."] == [[3.0], [4.0], [5.0]]
    assert len(result["results"]) == 3


@pytest.mark.anyio
async def test_nugget_all_chunks_without_sentences_skips_embed(temp_db):
    hits = [_hit("a", 1, ""), _hit("b", 2, "   ")]

    embedder, captured, result = await _search(temp_db, hits, [])

    embedder.embed.assert_not_called()
    assert [vecs for _, vecs in captured] == [None, None]
    assert len(result["results"]) == 2
