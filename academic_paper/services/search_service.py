"""Search logic for the /search endpoint's 4 modes, extracted from server.py (#357).

server.py's search() handler stays a thin FastAPI shim: validate query params,
call run_search(), let its own try/except turn exceptions into HTTP responses.
This module holds the actual retrieval logic so it's testable without booting
the FastAPI app.
"""

import asyncio
import logging
import sqlite3
from typing import Literal, get_args

from academic_paper.db import db_connection, search_fts
from academic_paper.embedder import EmbedderClient
from academic_paper.hybrid import rrf_merge
from academic_paper.nugget import extract_nuggets, split_sentences
from academic_paper.telemetry import get_tracer
from academic_paper.vector_store import QdrantStore

logger = logging.getLogger(__name__)
tracer = get_tracer()

SearchMode = Literal["vector", "keyword", "hybrid", "nugget"]
# Single source of truth for the accepted modes; server.py builds its Query pattern from this (#677).
SEARCH_MODES: tuple[str, ...] = get_args(SearchMode)


async def _read_db(db_path: str, fn, *args):
    """Run ``fn(conn, *args)`` on a worker thread with its own short-lived connection (#678).

    sqlite3 calls are blocking (up to busy_timeout on lock contention); running them on the
    event loop would stall every request. A connection must not cross threads, so each read
    opens and closes its own inside the worker.
    """

    def _run():
        with db_connection(db_path) as conn:
            return fn(conn, *args)

    return await asyncio.to_thread(_run)


def _page_map(conn: sqlite3.Connection, ids: list, id_column: str) -> dict:
    meta = _fetch_chunk_meta(conn.cursor(), ids, id_column, f"{id_column}, page_start")
    return {k: v["page_start"] for k, v in meta.items()}


def _resolve_vector_chunk_ids(conn: sqlite3.Connection, vector_results: list, mode: str) -> list:
    """Attach each Qdrant point's chunk_id (looked up in `chunks`); orphan points are dropped (#145).

    Returns new dicts; the input results are not modified.
    """
    vec_qids = [v["id"] for v in vector_results]
    qid_to_cid = {
        k: v["id"] for k, v in _fetch_chunk_meta(conn.cursor(), vec_qids, "qdrant_id", "id, qdrant_id").items()
    }
    resolved = []
    for vec_result in vector_results:
        chunk_id = qid_to_cid.get(vec_result["id"])
        if chunk_id is None:
            logger.warning("search mode=%s: skipping orphan point %s (no chunks row)", mode, vec_result["id"])
            continue
        resolved.append({**vec_result, "payload": {**(vec_result.get("payload") or {}), "chunk_id": chunk_id}})
    return resolved


def _make_snippet(text: str, snippet_length: int) -> str:
    """Leading `snippet_length` chars of `text`; 0 (or less) means the full text."""
    return text[:snippet_length] if snippet_length > 0 else text


def _result_row(
    *, rank: int, score: float, paper_id: int, chunk_index: int, page_start: int | None, snippet: str
) -> dict:
    """One /search result entry; the only place the response row shape is defined."""
    return {
        "rank": rank,
        "score": score,
        "paper_id": paper_id,
        "chunk_index": chunk_index,
        "page_start": page_start,
        "snippet": snippet,
    }


def _fetch_chunk_meta(
    cursor: sqlite3.Cursor, ids: list, id_column: str, select_columns: str
) -> dict[object, sqlite3.Row]:
    """Fetch `chunks` rows for the given ids, keyed by `id_column`'s value.

    Builds the `IN (?,?,...)` placeholder SQL shared by every /search mode
    (#346) and guards the empty-ids case, where SQLite would reject
    `IN ()` — returns `{}` without querying.
    """
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = cursor.execute(
        f"SELECT {select_columns} FROM chunks WHERE {id_column} IN ({placeholders})",  # noqa: S608
        ids,
    ).fetchall()
    return {row[id_column]: row for row in rows}


async def _search_keyword(db_path: str, *, q: str, limit: int, paper_id: int | None, snippet_length: int) -> list:
    fts_results = await _read_db(db_path, lambda conn: search_fts(conn, query=q, limit=limit, paper_id=paper_id))
    return [
        _result_row(
            rank=rank,
            score=r["rank"],
            paper_id=r["paper_id"],
            chunk_index=r["chunk_index"],
            page_start=r["page_start"],
            snippet=_make_snippet(r["text"], snippet_length),
        )
        for rank, r in enumerate(fts_results, start=1)
    ]


async def _search_vector(
    db_path: str,
    embedder: EmbedderClient,
    vector_store: QdrantStore,
    *,
    q: str,
    limit: int,
    paper_id: int | None,
    snippet_length: int,
) -> list:
    with tracer.start_as_current_span("embed.query"):
        query_vector = await embedder.embed_single(q, mode="search")
    search_results = await vector_store.asearch(query_vector=query_vector, limit=limit, paper_id_filter=paper_id)
    vec_qids = [r["id"] for r in search_results]
    vec_page_map = await _read_db(db_path, _page_map, vec_qids, "qdrant_id")
    results = []
    for result in search_results:
        qdrant_id = result["id"]
        payload = result.get("payload") or {}
        # Orphan points (partial ingest) lack paper_id; skip them like rrf_merge does
        # instead of surfacing a KeyError as a 500.
        if payload.get("paper_id") is None:
            logger.warning("search mode=vector: skipping orphan point %s (payload missing paper_id)", qdrant_id)
            continue
        results.append(
            _result_row(
                rank=len(results) + 1,
                score=result["score"],
                paper_id=payload["paper_id"],
                chunk_index=payload.get("chunk_index", 0),
                page_start=vec_page_map.get(qdrant_id),
                snippet=_make_snippet(payload.get("text", ""), snippet_length),
            )
        )
    return results


async def _embed_nugget_sentences(embedder: EmbedderClient, merged_slice: list) -> list:
    """Batch-embed all sentences across all chunks in one HTTP call (#143), re-sliced per chunk."""
    chunk_sentences = [split_sentences(r["text"]) for r in merged_slice]
    flat_sentences = [s for sents in chunk_sentences for s in sents]
    if flat_sentences:
        with tracer.start_as_current_span("embed.nuggets"):
            flat_vecs = await embedder.embed(flat_sentences, mode="search")
    else:
        flat_vecs = []
    nugget_vecs: list[list[list[float]] | None] = []
    offset = 0
    for sents in chunk_sentences:
        if sents:
            nugget_vecs.append(flat_vecs[offset : offset + len(sents)])
            offset += len(sents)
        else:
            nugget_vecs.append(None)
    return nugget_vecs


async def _search_hybrid(
    db_path: str,
    embedder: EmbedderClient,
    vector_store: QdrantStore,
    *,
    q: str,
    mode: str,
    limit: int,
    paper_id: int | None,
    snippet_length: int,
    nuggets_per_chunk: int,
    nugget_embed_weight: float,
) -> list:
    """hybrid and nugget share retrieval (FTS + vector, RRF); they differ only in the snippet."""
    fts_results = await _read_db(db_path, lambda conn: search_fts(conn, query=q, limit=limit, paper_id=paper_id))
    with tracer.start_as_current_span("embed.query"):
        query_vector = await embedder.embed_single(q, mode="search")
    vector_results = await vector_store.asearch(query_vector=query_vector, limit=limit, paper_id_filter=paper_id)

    # Ingest never stores chunk_id in the Qdrant payload, so resolve every point's chunk_id
    # from the chunks table. Orphan points (no chunks row) are dropped in one place.
    resolved_vector_results = await _read_db(db_path, _resolve_vector_chunk_ids, vector_results, mode)

    merged_slice = rrf_merge(fts_results, resolved_vector_results)[:limit]
    merged_ids = [r["chunk_id"] for r in merged_slice]
    merged_page_map = await _read_db(db_path, _page_map, merged_ids, "id")
    if mode == "nugget" and nugget_embed_weight > 0.0:
        nugget_vecs = await _embed_nugget_sentences(embedder, merged_slice)
    else:
        nugget_vecs = [None] * len(merged_slice)

    results = []
    for rank, (result, chunk_sentence_vecs) in enumerate(zip(merged_slice, nugget_vecs), start=1):
        if mode == "nugget":
            snippet = extract_nuggets(
                q,
                result["text"],
                top_k=nuggets_per_chunk,
                embed_weight=nugget_embed_weight,
                query_vec=query_vector,
                sentence_vecs=chunk_sentence_vecs,
            )
        else:
            snippet = _make_snippet(result["text"], snippet_length)
        results.append(
            _result_row(
                rank=rank,
                score=result["rrf_score"],
                paper_id=result["paper_id"],
                chunk_index=result["chunk_index"],
                page_start=merged_page_map.get(result["chunk_id"]),
                snippet=snippet,
            )
        )
    return results


async def run_search(
    db_path: str,
    embedder: EmbedderClient,
    vector_store: QdrantStore,
    *,
    q: str,
    mode: str,
    limit: int,
    paper_id: int | None,
    snippet_length: int,
    nuggets_per_chunk: int,
    nugget_embed_weight: float,
) -> dict:
    """Run one of vector/keyword/hybrid/nugget search and return the /search response body.

    Reads the SQLite database at ``db_path`` on worker threads (#678). Raises ValueError for an unknown mode (the endpoint validates it first); otherwise
    whatever the underlying DB/embedder/vector_store calls raise. The caller (server.py's
    search() handler) is responsible for turning that into an HTTP response.
    """
    if mode == "keyword":
        results = await _search_keyword(db_path, q=q, limit=limit, paper_id=paper_id, snippet_length=snippet_length)
    elif mode == "vector":
        results = await _search_vector(
            db_path, embedder, vector_store, q=q, limit=limit, paper_id=paper_id, snippet_length=snippet_length
        )
    elif mode in ("hybrid", "nugget"):
        results = await _search_hybrid(
            db_path,
            embedder,
            vector_store,
            q=q,
            mode=mode,
            limit=limit,
            paper_id=paper_id,
            snippet_length=snippet_length,
            nuggets_per_chunk=nuggets_per_chunk,
            nugget_embed_weight=nugget_embed_weight,
        )
    else:
        raise ValueError(f"unknown search mode: {mode!r} (expected one of {', '.join(SEARCH_MODES)})")
    return {"mode": mode, "query": q, "results": results}
