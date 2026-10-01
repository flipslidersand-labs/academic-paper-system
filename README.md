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

## Scheduled workflow watchdog

`.github/workflows/scheduled-watchdog.yml` (GitHub-hosted `ubuntu-latest`, every 3 hours + manual dispatch) watches the
self-hosted-runner cron workflows. A job that never gets a runner runs no steps, so its own notifications cannot fire (#499).
It runs `scripts/watchdog_check.py` against `gh run list` output and fails when:

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| a run stays `queued` | > 3h | GitHub cancels queued jobs after 24h; 3h catches a dead runner early |
| latest completed run is `cancelled` / `failure` | - | the failure was not otherwise noticed |
| last `success` too old | `arxiv-daily` 26h, `portfolio-publish` ~170h | daily cron + 2h slack; weekly cron (Sun) + ~2h slack |

On a violation the job exits 1 (red run in the Actions tab) and posts the violation lines plus the run URL to Discord.
Read the message as `<workflow name>: <reason>`; run `gh run list --workflow <file>` to investigate.
The Discord step uses the `DISCORD_WEBHOOK_URL` repository secret (same one as `arxiv-daily.yml`); if it is unset the
notification is skipped and only the red run remains. `tests/test_watchdog_workflow.py` guards that the watchdog stays
on GitHub-hosted runners, keeps `actions: read` only, covers both workflows and has a timeout.

## License

MIT

