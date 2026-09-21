"""Search logic for the /search endpoint's 4 modes, extracted from server.py (#357).

server.py's search() handler stays a thin FastAPI shim: validate query params,
call run_search(), let its own try/except turn exceptions into HTTP responses.
This module holds the actual retrieval logic so it's testable without booting
the FastAPI app.
"""

import logging
import sqlite3

from academic_paper.db import search_fts
from academic_paper.embedder import EmbedderClient
from academic_paper.hybrid import rrf_merge
from academic_paper.nugget import extract_nuggets, split_sentences
from academic_paper.telemetry import get_tracer
from academic_paper.vector_store import QdrantStore

logger = logging.getLogger(__name__)
tracer = get_tracer()


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


async def run_search(
    conn: sqlite3.Connection,
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

    Raises whatever the underlying DB/embedder/vector_store calls raise; the
    caller (server.py's search() handler) is responsible for turning that
    into an HTTP response.
    """
    cursor = conn.cursor()

    if mode == "keyword":
        fts_results = search_fts(conn, query=q, limit=limit, paper_id=paper_id)
        kw_ids = [r["chunk_id"] for r in fts_results]
        kw_meta_map = _fetch_chunk_meta(cursor, kw_ids, "id", "id, page_start, chunk_index")
        results = []
        for rank, result in enumerate(fts_results, start=1):
            chunk_id = result["chunk_id"]
            paper_id_res = result["paper_id"]
            meta = kw_meta_map.get(chunk_id)
            page_start = meta["page_start"] if meta else None
            chunk_index = meta["chunk_index"] if meta else 0
            full_text = result["text"]
            snippet = full_text[:snippet_length] if snippet_length > 0 else full_text
            results.append(
                {
                    "rank": rank,
                    "score": result["rank"],
                    "paper_id": paper_id_res,
                    "chunk_index": chunk_index,
                    "page_start": page_start,
                    "snippet": snippet,
                }
            )
        return {"mode": mode, "query": q, "results": results}

    if mode == "vector":
        with tracer.start_as_current_span("embed.query"):
            query_vector = await embedder.embed_single(q, mode="search")
        search_results = await vector_store.asearch(query_vector=query_vector, limit=limit, paper_id_filter=paper_id)
        vec_qids = [r["id"] for r in search_results]
        vec_meta = _fetch_chunk_meta(cursor, vec_qids, "qdrant_id", "qdrant_id, page_start")
        vec_page_map = {k: v["page_start"] for k, v in vec_meta.items()}
        results = []
        for result in search_results:
            qdrant_id = result["id"]
            payload = result.get("payload") or {}
            # Orphan points (partial ingest) lack paper_id; skip them like rrf_merge does
            # instead of surfacing a KeyError as a 500.
            if payload.get("paper_id") is None:
                logger.warning("search mode=vector: skipping orphan point %s (payload missing paper_id)", qdrant_id)
                continue
            full_text = payload.get("text", "")
            snippet = full_text[:snippet_length] if snippet_length > 0 else full_text
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": result["score"],
                    "paper_id": payload["paper_id"],
                    "chunk_index": payload.get("chunk_index", 0),
                    "page_start": vec_page_map.get(qdrant_id),
                    "snippet": snippet,
                }
            )
        return {"mode": mode, "query": q, "results": results}

    # hybrid or nugget (same retrieval, different snippet)
    fts_results = search_fts(conn, query=q, limit=limit, paper_id=paper_id)
    fts_ids = [r["chunk_id"] for r in fts_results]
    ci_map = {k: v["chunk_index"] for k, v in _fetch_chunk_meta(cursor, fts_ids, "id", "id, chunk_index").items()}
    if ci_map:
        for fts_result in fts_results:
            if fts_result["chunk_id"] in ci_map:
                fts_result["chunk_index"] = ci_map[fts_result["chunk_id"]]

    with tracer.start_as_current_span("embed.query"):
        query_vector = await embedder.embed_single(q, mode="search")
    vector_results = await vector_store.asearch(query_vector=query_vector, limit=limit, paper_id_filter=paper_id)

    missing_qids = [v["id"] for v in vector_results if "chunk_id" not in v["payload"]]
    qid_to_cid = {k: v["id"] for k, v in _fetch_chunk_meta(cursor, missing_qids, "qdrant_id", "id, qdrant_id").items()}
    if qid_to_cid:
        for vec_result in vector_results:
            if "chunk_id" not in vec_result["payload"] and vec_result["id"] in qid_to_cid:
                vec_result["payload"]["chunk_id"] = qid_to_cid[vec_result["id"]]

    merged = rrf_merge(fts_results, vector_results)
    # Drop orphaned Qdrant results that have no chunk_id — these can arise
    # from a partial ingest failure before compensation runs (#145).
    merged_slice = [r for r in merged[:limit] if "chunk_id" in r]
    merged_ids = [r["chunk_id"] for r in merged_slice]
    merged_meta = _fetch_chunk_meta(cursor, merged_ids, "id", "id, page_start")
    merged_page_map = {k: v["page_start"] for k, v in merged_meta.items()}
    # nugget mode: batch-embed all sentences across all chunks in a single
    # HTTP call instead of one call per chunk (#143).
    if mode == "nugget" and nugget_embed_weight > 0.0:
        chunk_sentences = [split_sentences(r["text"]) for r in merged_slice]
        flat_sentences = [s for sents in chunk_sentences for s in sents]
        if flat_sentences:
            with tracer.start_as_current_span("embed.nuggets"):
                flat_vecs = await embedder.embed(flat_sentences, mode="search")
        else:
            flat_vecs = []
        # Slice flat vectors back to per-chunk lists.
        nugget_vecs: list[list[list[float]] | None] = []
        offset = 0
        for sents in chunk_sentences:
            if sents:
                nugget_vecs.append(flat_vecs[offset : offset + len(sents)])
                offset += len(sents)
            else:
                nugget_vecs.append(None)
    else:
        nugget_vecs = [None] * len(merged_slice)

    results = []
    for rank, (result, chunk_sentence_vecs) in enumerate(zip(merged_slice, nugget_vecs), start=1):
        full_text = result["text"]
        if mode == "nugget":
            snippet = extract_nuggets(
                q,
                full_text,
                top_k=nuggets_per_chunk,
                embed_weight=nugget_embed_weight,
                query_vec=query_vector,
                sentence_vecs=chunk_sentence_vecs,
            )
        else:
            snippet = full_text[:snippet_length] if snippet_length > 0 else full_text
        results.append(
            {
                "rank": rank,
                "score": result["rrf_score"],
                "paper_id": result["paper_id"],
                "chunk_index": result["chunk_index"],
                "page_start": merged_page_map.get(result["chunk_id"]),
                "snippet": snippet,
            }
        )
    return {"mode": mode, "query": q, "results": results}
