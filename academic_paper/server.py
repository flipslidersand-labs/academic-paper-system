"""FastAPI server for academic paper ingestion and retrieval."""

import sqlite3
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from academic_paper.chunker import chunk_pages
from academic_paper.config import settings
from academic_paper.db import (
    get_connection,
    init_db,
    save_chunks,
    save_paper,
    update_paper_status,
    list_papers,
    get_paper,
    get_chunks,
    search_fts,
)
from academic_paper.extractor import extract_text, hash_file
from academic_paper.embedder import EmbedderClient
from academic_paper.vector_store import QdrantStore, make_qdrant_id
from academic_paper.hybrid import rrf_merge


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and services on startup."""
    init_db(settings.academic_db)
    # Initialize EmbedderClient and QdrantStore
    app.state.embedder = EmbedderClient()
    app.state.vector_store = QdrantStore()
    yield


app = FastAPI(title="Academic Paper System", lifespan=lifespan)


@app.post("/papers/ingest")
async def ingest_paper(file: UploadFile = File(...)):
    """Ingest a PDF paper.

    Args:
        file: PDF file to ingest.

    Returns:
        JSON response with paper_id, file_name, chunks count, and status.

    Raises:
        HTTPException: If file already exists (409) or processing fails (400).
    """
    try:
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Check for duplicates using file hash
        file_hash = hash_file(tmp_path)
        conn = get_connection(settings.academic_db)

        # Check if file already exists
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM papers WHERE file_hash = ?", (file_hash,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=409, detail="File already ingested")

        # Extract text from PDF
        pages = extract_text(tmp_path)
        if not pages:
            conn.close()
            raise HTTPException(status_code=400, detail="No text extracted from PDF")

        # Chunk pages
        chunks_list = chunk_pages(
            pages,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        if not chunks_list:
            conn.close()
            raise HTTPException(status_code=400, detail="No chunks generated")

        # Save paper to database
        paper_id = save_paper(conn, file.filename or "unknown.pdf", file_hash)

        try:
            # Embed chunks using EmbedderClient
            chunk_texts = [chunk["text"] for chunk in chunks_list]
            embeddings = await app.state.embedder.embed(chunk_texts, mode="index")

            # Ensure Qdrant collection exists
            app.state.vector_store.ensure_collection()

            # Prepare points for Qdrant with qdrant_id based on file_hash
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks_list, embeddings)):
                qdrant_id = make_qdrant_id(file_hash, idx)
                chunk["qdrant_id"] = qdrant_id
                points.append({
                    "id": qdrant_id,
                    "vector": embedding,
                    "payload": {
                        "paper_id": paper_id,
                        "chunk_index": idx,
                        "text": chunk["text"],
                        "file_name": file.filename or "unknown.pdf",
                    }
                })

            # Upsert to Qdrant
            app.state.vector_store.upsert(points)

            # Save chunks with qdrant_id to database
            save_chunks(conn, paper_id, chunks_list)

            # Update status to indexed
            update_paper_status(conn, paper_id, "indexed")
            conn.close()

            return {
                "paper_id": paper_id,
                "file_name": file.filename or "unknown.pdf",
                "chunks": len(chunks_list),
                "status": "indexed",
            }

        except Exception as e:
            # If embedding/Qdrant fails, update status and return error
            update_paper_status(conn, paper_id, "failed")
            conn.close()
            raise HTTPException(status_code=400, detail=f"Embedding or Qdrant error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/papers")
def list_papers_endpoint(
    limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)
):
    """List papers with pagination.

    Args:
        limit: Number of papers to return (1-100, default 20).
        offset: Number of papers to skip (default 0).

    Returns:
        JSON response with total count and papers list.
    """
    conn = get_connection(settings.academic_db)

    # Get total count
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    total = cursor.fetchone()[0]

    # Get papers with pagination
    cursor.execute(
        """
        SELECT id, file_name, file_hash, status, ingested_at
        FROM papers
        ORDER BY ingested_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )
    papers = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return {"total": total, "papers": papers}


@app.get("/papers/{paper_id}")
def get_paper_endpoint(paper_id: int):
    """Get paper details by ID.

    Args:
        paper_id: ID of the paper.

    Returns:
        JSON response with paper details.

    Raises:
        HTTPException: If paper not found (404).
    """
    conn = get_connection(settings.academic_db)
    paper = get_paper(conn, paper_id)
    conn.close()

    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    return paper


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("hybrid", pattern="^(vector|keyword|hybrid)$"),
    limit: int = Query(10, ge=1, le=100),
    paper_id: int | None = Query(None),
):
    """Search papers using vector similarity, keyword search, or hybrid.

    Args:
        q: Search query text.
        mode: Search mode:
              - "hybrid": Combine FTS5 (BM25) and vector search using RRF
              - "keyword": FTS5 (BM25) search only
              - "vector": Vector similarity search only
        limit: Maximum number of results to return (1-100, default 10).
        paper_id: Optional paper ID to filter results to a single paper.

    Returns:
        JSON response with search results:
        {
            "mode": str,
            "query": str,
            "results": [
                {
                    "rank": int,
                    "score": float,
                    "paper_id": int,
                    "chunk_index": int,
                    "page_start": int | None,
                    "snippet": str
                }
            ]
        }

    Raises:
        HTTPException: If embedding or search fails (400).
    """
    try:
        conn = get_connection(settings.academic_db)
        cursor = conn.cursor()

        if mode == "keyword":
            # Keyword search only (FTS5 BM25)
            fts_results = search_fts(
                conn,
                query=q,
                limit=limit,
                paper_id=paper_id,
            )

            # Build results from FTS5
            results = []
            for rank, result in enumerate(fts_results, start=1):
                chunk_id = result["chunk_id"]
                paper_id_res = result["paper_id"]

                # Fetch chunk from database to get page_start
                cursor.execute(
                    "SELECT page_start FROM chunks WHERE id = ?",
                    (chunk_id,),
                )
                row = cursor.fetchone()
                page_start = row["page_start"] if row else None

                # Extract snippet
                snippet = result["text"][:200]

                results.append({
                    "rank": rank,
                    "score": result["rank"],  # FTS5 BM25 score
                    "paper_id": paper_id_res,
                    "chunk_index": result.get("chunk_index", 0),
                    "page_start": page_start,
                    "snippet": snippet,
                })

            conn.close()
            return {
                "mode": mode,
                "query": q,
                "results": results,
            }

        elif mode == "vector":
            # Vector search only
            query_vector = await app.state.embedder.embed_single(q, mode="search")
            search_results = app.state.vector_store.search(
                query_vector=query_vector,
                limit=limit,
                paper_id_filter=paper_id,
            )

            # Build results from vector search
            results = []
            for rank, result in enumerate(search_results, start=1):
                qdrant_id = result["id"]
                score = result["score"]
                payload = result["payload"]
                chunk_idx = payload["chunk_index"]
                paper_id_res = payload["paper_id"]

                # Fetch chunk from database to get page_start
                cursor.execute(
                    "SELECT page_start FROM chunks WHERE qdrant_id = ?",
                    (qdrant_id,),
                )
                row = cursor.fetchone()
                page_start = row["page_start"] if row else None

                # Extract snippet
                snippet = payload["text"][:200]

                results.append({
                    "rank": rank,
                    "score": score,
                    "paper_id": paper_id_res,
                    "chunk_index": chunk_idx,
                    "page_start": page_start,
                    "snippet": snippet,
                })

            conn.close()
            return {
                "mode": mode,
                "query": q,
                "results": results,
            }

        else:  # mode == "hybrid"
            # Hybrid search: combine FTS5 and vector using RRF
            # 1. FTS5 search
            fts_results = search_fts(
                conn,
                query=q,
                limit=limit,
                paper_id=paper_id,
            )

            # Enrich FTS5 results with chunk_index
            for fts_result in fts_results:
                cursor.execute(
                    "SELECT chunk_index FROM chunks WHERE id = ?",
                    (fts_result["chunk_id"],),
                )
                row = cursor.fetchone()
                if row:
                    fts_result["chunk_index"] = row["chunk_index"]

            # 2. Vector search
            query_vector = await app.state.embedder.embed_single(q, mode="search")
            vector_results = app.state.vector_store.search(
                query_vector=query_vector,
                limit=limit,
                paper_id_filter=paper_id,
            )

            # Prepare vector results for RRF (add chunk_id to payload if missing)
            for vec_result in vector_results:
                if "chunk_id" not in vec_result["payload"]:
                    # Try to get chunk_id from qdrant_id
                    cursor.execute(
                        "SELECT id FROM chunks WHERE qdrant_id = ?",
                        (vec_result["id"],),
                    )
                    row = cursor.fetchone()
                    if row:
                        vec_result["payload"]["chunk_id"] = row["id"]

            # 3. RRF merge
            merged = rrf_merge(fts_results, vector_results)

            # 4. Build results with rank
            results = []
            for rank, result in enumerate(merged[:limit], start=1):
                chunk_id = result["chunk_id"]
                paper_id_res = result["paper_id"]
                chunk_idx = result["chunk_index"]
                rrf_score = result["rrf_score"]

                # Fetch chunk from database to get page_start
                cursor.execute(
                    "SELECT page_start FROM chunks WHERE id = ?",
                    (chunk_id,),
                )
                row = cursor.fetchone()
                page_start = row["page_start"] if row else None

                # Extract snippet
                snippet = result["text"][:200]

                results.append({
                    "rank": rank,
                    "score": rrf_score,
                    "paper_id": paper_id_res,
                    "chunk_index": chunk_idx,
                    "page_start": page_start,
                    "snippet": snippet,
                })

            conn.close()
            return {
                "mode": mode,
                "query": q,
                "results": results,
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search error: {str(e)}")
