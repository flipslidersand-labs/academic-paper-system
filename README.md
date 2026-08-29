# academic-paper-system

Research paper knowledge base — PDF ingestion → hybrid search → structured summarization RAG (Python + FastAPI).

## Architecture

```
PDF upload
  → pdfplumber extraction
  → text chunking (512 tokens / 64 overlap)
  → e5-large-v2 embeddings (768-d) via embedding-svc
  → Qdrant vector store  +  SQLite FTS5 (BM25)
  → RRF hybrid retrieval
  → Gemini / Ollama structured summarization
```

**Stack:** FastAPI · pdfplumber · Qdrant · SQLite FTS5 · Google Generative AI · Ollama · OpenTelemetry

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/papers/ingest` | Upload a PDF; returns **202 + `job_id`** and indexes in the background (`?wait=true` for synchronous 200) |
| `GET` | `/papers` | List papers (paginated) |
| `GET` | `/papers/{id}` | Paper detail |
| `GET` | `/papers/{id}/summary` | LLM-generated structured summary (cached) |
| `GET` | `/jobs/{id}` | Poll a background job; `done` carries `result.paper_id`/`result.chunks` |
| `GET` | `/search` | Hybrid search (`mode=hybrid\|vector\|keyword`) |

### Ingestion (async)

`POST /papers/ingest` accepts the upload, deduplicates by file hash (**409** if
already ingested), then extracts → chunks → embeds → upserts in the background,
returning **202** immediately:

```json
{ "job_id": "…", "paper_id": 1, "status": "pending" }
```

Poll `GET /jobs/{job_id}` until `status` is `done` (or `failed`). On `done`:

```json
{ "status": "done", "result": { "paper_id": 1, "chunks": 42, "status": "indexed" } }
```

Pass `?wait=true` to process synchronously and receive the indexed result (200)
in one call — used by the collector scripts' `--wait`-free default via polling.

### Search modes

- **hybrid** — RRF fusion of BM25 (FTS5) + vector scores
- **keyword** — BM25 only (fast, offline)
- **vector** — semantic similarity only

### Summary response schema

```json
{
  "objective": "...",
  "method": "...",
  "results": "...",
  "limitations": "...",
  "keywords": ["deep learning", "..."],
  "cached": false
}
```

## Setup

```bash
cp .env.example .env   # configure embedding-svc, Qdrant, Gemini API key
pip install -e ".[dev]"
uvicorn academic_paper.server:app --reload --port 8020
```

Key env vars: `EMBEDDING_SVC_URL`, `QDRANT_URL`, `GOOGLE_API_KEY`, `OLLAMA_URL`  
See `.env.example` for the full list.

## Test

```bash
pytest
```

## License

MIT

