# academic-paper-system

Research paper knowledge base — PDF ingestion → hybrid search → structured summarization RAG (Python + FastAPI).

## Architecture

```
PDF upload
  → pdfplumber extraction
  → text chunking (512 tokens / 64 overlap)
  → multilingual-e5-base embeddings (768-d) via embedding-svc
  → Qdrant vector store  +  SQLite FTS5 (BM25)
  → RRF hybrid retrieval
  → Gemini / Ollama structured summarization
```

**Stack:** FastAPI · pdfplumber · Qdrant · SQLite FTS5 · Google Generative AI · Ollama · OpenTelemetry · Prometheus (`prometheus-fastapi-instrumentator`)

## API

Every endpoint except `GET /health` and the static UI at `/ui` requires the `X-API-Key` header when the
`API_KEY` env var is set (the server refuses to start when it is unset, unless `AUTH_DISABLED=true` is set explicitly).

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
| `GET` | `/jobs` | yes | List background jobs, newest first (`limit`/`offset`, in-memory jobs only; returns `total` + `jobs`) |
| `GET` | `/jobs/{id}` | yes | Poll a background job; `done` carries `result.paper_id`/`result.chunks` |
| `GET` | `/search` | yes | Hybrid search (`mode=hybrid\|vector\|keyword`) |
| `GET` | `/health` | no | Qdrant / embedding-svc reachability (**503** when degraded) |
| `GET` | `/stats` | yes | Counts: `papers`, `chunks`, `qdrant_points` |
| `GET` | `/metrics` | yes (**admin** scope) | Prometheus metrics (`prometheus-fastapi-instrumentator`). Requires a full-scope key (`API_KEY`/`API_KEYS`); `INGEST_API_KEY` gets 403 |
| `GET` | `/ui/` | no | Static frontend (`frontend/`, mounted only when that directory exists). Serves HTML/JS without a key; the page asks for the API key and sends it as `X-API-Key` on its API calls |

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
in one call. Intended for small inputs and tests. The collector scripts do not use
it: they submit without `wait`, receive 202, and poll `GET /jobs/{job_id}`
(see `submit_and_wait` in `scripts/ingest_client.py`).

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

The daily collection job is run by a private ops repo that invokes these scripts against the API
(see "Scheduled jobs" below); it then triggers bulk summarization (`POST /jobs/summarize-all`) and
scoring (`POST /papers/score-all`). The schedule and runner are defined there, not in this repo.

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

  The weekly publish job now lives in the private ops repo (see "Scheduled jobs" below);
  it no longer commits `docs/` here.

## CI workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `.github/workflows/ci.yml` | push / pull_request to `main`, manual | `test` (pytest + ruff via the shared reusable workflow, coverage threshold 71%), `docker-build` (compose validation and image build), `gitleaks` (secret scan). ruff is pinned, see "ruff version pin" |
| `.github/workflows/license-check.yml` | push, pull_request | `pip-licenses` allow-list check for runtime and dev installs; `fpdf2` is the only exception, see "Third-party licenses" |
| `.github/workflows/lock-update.yml` | weekly cron (Mon 03:00 JST) + manual | Runs `scripts/update-lock.sh --upgrade` and opens a PR with the new `requirements.lock`. CI is started explicitly via `workflow_dispatch`, because pushes made with `GITHUB_TOKEN` do not trigger workflows |
| `.github/workflows/vuln-scan.yml` | push, pull_request, weekly cron (Mon 03:00 UTC), manual | `pip-audit` on the hash-pinned `requirements.lock` plus a Trivy scan of the Docker image (#350) |

## Setup

```bash
cp .env.example .env   # configure embedding-svc, Qdrant, Gemini API key
pip install -e ".[dev]"
uvicorn academic_paper.server:app --reload --port 8020
```

Key env vars: `API_KEY` / `API_KEYS` (full-scope keys), `INGEST_API_KEY` (ingest-only key; rotate by swapping it after updating the collector's `PAPER_API_KEY`), `EMBEDDING_SVC_URL`, `QDRANT_URL`, `GOOGLE_API_KEY`, `OLLAMA_URL`, `SUMMARIZE_TOTAL_TIMEOUT`, `LOG_FORMAT`  
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

Qdrant runs with API-key auth enabled (#657). Set `QDRANT_API_KEY` in `.env`
(one value, used for both the Qdrant server and `paper-rag`); Compose refuses to
start Qdrant if it is empty. Generate one with e.g. `openssl rand -hex 32`.
The Qdrant image is pinned to a tag plus digest; Dependabot updates it.

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

### ruff version pin

ruff is pinned in three places that must match: `pyproject.toml` (dev extra
`ruff==X`), `.github/workflows/ci.yml` (`ruff-version`) and
`.pre-commit-config.yaml` (`rev: vX`). `tests/test_ruff_pin_sync.py` fails when
they drift (e.g. a Dependabot bump of only `pyproject.toml`), so bump all three
in the same PR.

## License

MIT

### Third-party licenses

- Runtime dependencies (`requirements.lock`, what the Docker image ships) are all
  permissive (MIT / BSD / Apache-2.0 / ISC / PSF; `certifi` is MPL-2.0).
- **Dev-only exception:** `fpdf2` (LGPL-3.0, optional `dev` extra) is used only by
  tests to generate sample PDFs. It is not in `requirements.lock`, not installed in
  the Docker image, and not part of the built wheel, so it is never distributed.
  `pip install academic-paper-system[dev]` does install it into your dev environment.
- `license-check.yml` (see "CI workflows") enforces an SPDX-style allow-list with
  `pip-licenses` for both runtime and dev installs; `fpdf2` is the sole listed exception.
  Adding another copyleft package makes the check fail.

## Scheduled jobs

The daily arXiv collection, the weekly portfolio publish and their watchdog run from a private ops repo on a
self-hosted runner. The watchdog check logic (`scripts/watchdog_check.py`) stays in this repo. This public repo has no self-hosted runner and no `runs-on` pointing at one: public CI uses
GitHub-hosted runners only (repo-compass ADR-010).

## 設計判断 / Architecture decisions

[ADR 一覧](docs/adr/README.md) — 設計判断、根拠資料、未決定事項。
