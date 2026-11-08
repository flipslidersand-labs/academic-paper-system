# Stage 1: Build — install pinned deps into a virtualenv
FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 AS builder
WORKDIR /build
COPY pyproject.toml README.md requirements.lock ./
COPY academic_paper/ ./academic_paper/
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --require-hashes -r requirements.lock && \
    /opt/venv/bin/pip install --no-cache-dir --no-deps .

# Stage 2: Runtime — slim image with only the venv, no build toolchain
FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 AS runtime
COPY --from=builder /opt/venv /opt/venv
# trivy-image (CRITICAL/HIGH, fixable only) で検出された zlib1g の修正版が
# debian-security に出ていてもベースイメージに未反映の間は、ここだけ更新する。
# ベース更新で解消したら不要(この RUN を削除)。
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends --only-upgrade zlib1g && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY frontend/ ./frontend/
ENV PATH="/opt/venv/bin:$PATH"
# Run as a dedicated non-root user (CIS Docker Benchmark 4.1, #426). /data is
# where the SQLite DB (ACADEMIC_DB) lives when ./data is bind-mounted via
# docker-compose; it must be owned by this user to be writable at runtime.
RUN useradd --system --uid 10001 --create-home app && \
    mkdir -p /data && chown app:app /data /app
USER app
EXPOSE 8020
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8020/health', timeout=4)"
CMD ["uvicorn", "academic_paper.server:app", "--host", "0.0.0.0", "--port", "8020"]
