"""Hybrid search using Reciprocal Rank Fusion (RRF) to combine FTS5 and vector results."""


def rrf_merge(
    fts_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion combines BM25 and vector search results.

    RRF formula: score += 1 / (k + rank_position)  (1-indexed)

    Args:
        fts_results: BM25 results from FTS5 search.
                    Each dict contains: chunk_id, paper_id, chunk_index, text, rank
        vector_results: Vector similarity results from Qdrant.
                       Each dict contains: id, score, payload (with chunk_id, paper_id, chunk_index, text)
        k: Constant for RRF formula (default 60).

    Returns:
        List of merged results sorted by rrf_score (descending).
        Each dict contains: chunk_id, paper_id, chunk_index, text, rrf_score
    """
    scores: dict[int, float] = {}
    meta: dict[int, dict] = {}

    # Process FTS5 results (rank_position is 1-indexed)
    for i, result in enumerate(fts_results):
        chunk_id = result["chunk_id"]
        rrf_score = 1.0 / (k + i + 1)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
        meta[chunk_id] = {
            "chunk_id": chunk_id,
            "paper_id": result["paper_id"],
            "chunk_index": result.get("chunk_index", 0),
            "text": result["text"],
        }

    # Process vector results (rank_position is 1-indexed)
    for i, result in enumerate(vector_results):
        payload = result["payload"]
        # Try to get chunk_id from payload
        chunk_id = payload.get("chunk_id")
        if chunk_id is None:
            continue

        rrf_score = 1.0 / (k + i + 1)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

        # Add or update metadata if not already present from FTS5
        if chunk_id not in meta:
            meta[chunk_id] = {
                "chunk_id": chunk_id,
                "paper_id": payload.get("paper_id"),
                "chunk_index": payload.get("chunk_index", 0),
                "text": payload.get("text", ""),
            }

    # Build result list with rrf_score and sort by score descending
    merged_results = [
        {"rrf_score": scores[chunk_id], **meta[chunk_id]}
        for chunk_id in scores.keys()
    ]
    merged_results.sort(key=lambda x: x["rrf_score"], reverse=True)

    return merged_results
