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

Every endpoint except `GET /health` requires the `X-API-Key` header when the
`API_KEY` env var is set (auth is disabled when it is unset).

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/papers/ingest` | yes | Upload a PDF; returns **202 + `job_id`** and indexes in the background (`?wait=true` for synchronous 200) |
| `GET` | `/papers` | yes | List papers (paginated; `author`/`category` filters, `sort=ingested_at\|score`) |
| `GET` | `/papers/{id}` | yes | Paper detail |
| `GET` | `/papers/{id}/summary` | yes | Cached structured summary only (never triggers LLM generation) |
| `POST` | `/papers/{id}/summary` | yes | Generate the summary (cached unless `?force=true`) |
| `POST` | `/papers/score-all` | yes | Compute and store relevance scores for all papers |
| `POST` | `/papers/{id}/score` | yes | Compute and store the relevance score for one paper |
| `GET` | `/summaries` | yes | List summaries with paper metadata (`limit`/`offset`) |
| `POST` | `/jobs/summarize-all` | yes | Start a background job that summarizes papers (**202 + `job_id`**) |
| `GET` | `/jobs` | yes | List background jobs |
| `GET` | `/jobs/{id}` | yes | Poll a background job; `done` carries `result.paper_id`/`result.chunks` |
| `GET` | `/search` | yes | Hybrid search (`mode=hybrid\|vector\|keyword`) |
| `GET` | `/health` | no | Qdrant / embedding-svc reachability (**503** when degraded) |
| `GET` | `/stats` | yes | Counts: `papers`, `chunks`, `qdrant_points` |

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

## Automated collection

Besides manual PDF upload, `scripts/` collects papers from external sources and
ingests them through `POST /papers/ingest` (the scripts only talk to the API
over HTTP):

| Script | Source |
|--------|--------|
| `scripts/arxiv_collect.py` | arXiv (`--categories`, `--max`, `--from-date`, `--until-date`) |
| `scripts/pubmed_collect.py` | PubMed |
| `scripts/openalex_collect.py` | OpenAlex |
| `scripts/semantic_scholar_collect.py` | Semantic Scholar |

All collectors accept `--api-url` (default `http://localhost:8020`) and
`--summary-file` (write a run summary JSON); run `python scripts/<name>.py --help`
for the full option list. When `API_KEY` is set on the server, export the same
value as `PAPER_API_KEY` for the scripts.

`.github/workflows/arxiv-daily.yml` runs them daily (cron `0 0 * * *`, 09:00 JST)
on a self-hosted runner that can reach `localhost:8020`, then triggers bulk
summarization (`POST /jobs/summarize-all`) and scoring (`POST /papers/score-all`).

## Scoring & Portfolio

- **Scoring** (`academic_paper/scorer.py`): `score = freshness (0-0.5) + category match (0-0.5)`.
  Freshness decays exponentially with a 30-day half-life; category match is the
  fraction of preferred categories present. Preferred categories come from the
  `PREFERRED_CATEGORIES` env var (default `cs.AI,cs.LG,cs.CL`).
- **Portfolio** (`scripts/generate_portfolio.py`): fetches papers sorted by score
  from the API and writes a static site (`index.html`, `papers.json`) to
  `--output-dir` (default `docs`):

  ```bash
  python scripts/generate_portfolio.py --api-url http://localhost:8020 --output-dir docs
  ```

  `.github/workflows/portfolio-publish.yml` regenerates and commits `docs/`
  weekly (Sunday 10:00 JST).

## Setup

```bash
cp .env.example .env   # configure embedding-svc, Qdrant, Gemini API key
pip install -e ".[dev]"
uvicorn academic_paper.server:app --reload --port 8020
```

Key env vars: `EMBEDDING_SVC_URL`, `QDRANT_URL`, `GOOGLE_API_KEY`, `OLLAMA_URL`, `SUMMARIZE_TOTAL_TIMEOUT`, `LOG_FORMAT`  
See `.env.example` for the full list.

### Docker Compose

The `paper-rag` container runs as a non-root user (uid `10001`, #426). Since
`docker-compose.yml` bind-mounts `./data:/data` for the SQLite DB, that host
directory must be writable by uid `10001` before the first `docker-compose up`:

```bash
mkdir -p data
sudo chown -R 10001:10001 data/
cp .env.example .env    # configure embedding-svc, Qdrant, Gemini API key
docker compose up -d
curl http://localhost:8020/health
```

`qdrant_storage`/`qdrant_snapshots` are created automatically by Compose on
first run. To reuse an existing volume on a given host instead, add a
`docker-compose.override.yml` (already gitignored):

```yaml
volumes:
  qdrant_storage:
    name: <existing-volume-name>
    external: true
  qdrant_snapshots:
    name: <existing-volume-name>
    external: true
```

## Test

```bash
pytest
```

## License

MIT

