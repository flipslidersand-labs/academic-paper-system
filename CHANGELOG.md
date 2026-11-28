# Changelog

All notable changes to this project are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/)

> No version has been tagged yet. The entries below summarize the git history up to the first release.

---

## [Unreleased]

### Added

- Core ingestion pipeline: PDF text extraction (pdfplumber), chunking, embeddings through the embedding service, and storage in Qdrant and SQLite, behind a FastAPI server with `POST /papers/ingest` and `GET /papers`.
- Asynchronous ingestion: `POST /papers/ingest` returns 202 with a `job_id` and indexes in the background (`?wait=true` for synchronous use), with compensation of partial failures (orphan vectors removed, failed papers can be re-ingested), file-hash deduplication, and an `arxiv_id` column with arXiv watermark matching.
- Search: vector search, SQLite FTS5 keyword search (BM25), RRF hybrid search, and a `nugget` mode that extracts BM25 plus embedding-scored snippets.
- Structured summarization: RAG prompt construction with structured JSON output, Gemini and Ollama clients with fallback and explicit provider selection (`LLM_PROVIDER`, `GEMINI_MODEL`), cached summaries, and `POST /papers/{id}/summary` separated from the read-only `GET`.
- Background jobs: bulk summarization (`POST /jobs/summarize-all`), job listing with pagination, and jobs persisted in SQLite.
- Scoring and portfolio: relevance score (freshness plus category match), score endpoints, author/category filters and a summaries list endpoint, and a static portfolio site generator with a weekly publish workflow.
- Automated collection: arXiv, PubMed, OpenAlex and Semantic Scholar collector scripts with date-range options, shared helpers, a daily collection workflow with bulk summarization and notifications, and a scheduled-workflow watchdog.
- Web UI: an Alpine.js paper browser with hybrid search, sending `X-API-Key` on all API calls, with pinned assets (SRI) and bundled CSS.
- Authentication: `X-API-Key` auth, per-endpoint scopes via `require_scope`, multiple keys for rotation (`API_KEYS`), and an ingest-only `INGEST_API_KEY`.
- Observability: OpenTelemetry instrumentation, structured JSON logging (`LOG_FORMAT`, `LOG_LEVEL`), a Prometheus `/metrics` endpoint, a startup health probe, and `GET /health` and `GET /stats`.
- Deployment: multi-stage Dockerfile with a healthcheck and a non-root user, Docker Compose with Qdrant (API-key auth, pinned image digest), and hash-pinned lock files for runtime and development dependencies with an update script.
- Configuration: a cached `get_settings()` with numeric constraints and placeholder validation, and a documented `.env.example`.
- Tests and CI: unit and integration tests across the server, collectors, search, summarizer and vector store; a coverage threshold in CI; pre-commit with ruff; gitleaks, pip-audit, Trivy and license checks; a weekly lock-update workflow; and Dependabot configuration.
- Documentation: README with architecture, API table and search modes, a retroactive ADR for RRF search fusion, and developer notes in `CLAUDE.md`.

### Security

- Fixed a fail-open authentication path: the server refuses to start without `API_KEY` unless `AUTH_DISABLED=true` is set explicitly. Read endpoints, job endpoints, `/summaries`, `/stats` and `/metrics` now require authentication, keys are compared in constant time (as UTF-8 bytes), and `/metrics` requires an admin-scope key.
- Hardened PDF download in collectors: SSRF guard for unsafe URLs, closed redirect bypass and DNS-rebinding gaps, size limits, strict content-type and magic-number checks, and proxy environment variables ignored.
- Bounded resource use to limit denial of service: wall-clock timeout on PDF extraction, streaming upload size checks (50 MB default limit), a cap on search query length, and timeouts on all external clients and the summarizer.
- Used `defusedxml` for XML parsing to prevent entity-expansion attacks.
- Prompt-injection hardening for summarization: PDF text is wrapped in delimiter tags with a defensive instruction, length limits and output validation.
- Stopped exposing raw exception text in job and score errors; sanitized upload filenames and control characters in metadata; redacted API keys in error output and passed collector keys by environment instead of command-line arguments.
- Restricted the public portfolio output to an allow-list of fields, and stopped publishing partial data when a fetch fails.
- Stopped exposing API docs without authentication and added security headers to `/ui`.
- Fixed an argparse injection through unquoted workflow expansion and removed direct shell interpolation of workflow inputs.
- Container and dependency hardening: non-root container user, pinned image versions and digests, refreshed base images and dependencies (including `pillow`, `urllib3` and `certifi`) to clear audit findings, least-privilege workflow permissions, and SHA-pinned GitHub Actions.

### Fixed

- Search correctness: escaped FTS5 queries and `LIKE` wildcards in author filters, exact category matching, deterministic ordering with a primary-key tiebreaker, BM25 IDF computed from document frequency, punctuation-aware tokenization, and handling of chunks with missing payloads or orphan Qdrant points.
- Ingestion robustness: temporary PDF files are always removed, a pending ingest returns 409, jobs and papers no longer stick in running/pending after failures or cancellation, and embedding/chunk count mismatches raise errors instead of being silently truncated.
- Error classification: dependency failures map to 502/503, input errors to 400, timeouts to 504 and unexpected errors to 500, including Gemini API errors and invalid LLM or embedding responses; `/health` returns 503 when degraded.
- Concurrency and event-loop blocking: blocking SQLite, hashing and file operations moved off the event loop, serialized job persistence per job id, atomic check-then-create in the job store, WAL mode with a busy timeout, and fixed connection handling around LLM calls.
- Summarizer fixes: robust JSON extraction from LLM output, null fields no longer stored as the string "null", real embeddings instead of dummy vectors, a bounded overall timeout, and dedicated errors when no usable chunks exist.
- Collector fixes: search API failures no longer report success, retries on 429 and 5xx, `Retry-After` handling, PubMed date and error-body handling, correct arXiv detail links, and fixed one-sided date ranges.
- Scoring and validation: future publication dates clamp to a freshness of 0.5, out-of-range `published_date` values are rejected, and metadata is validated at ingest and in collector CLIs.
- Workflow fixes: daily collection continues after an ingest failure, failures are surfaced, workflow runs are bounded with concurrency groups and timeouts, and portfolio publishing no longer commits empty output.
- Refactoring for maintainability: the server was split into routers and service layers (ingest, summary, search), settings access moved to `get_settings()`, raw SQL moved into the database module, and duplicated collector and retry logic was consolidated.
