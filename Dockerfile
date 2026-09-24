# Stage 1: Build — install pinned deps into a virtualenv
FROM python:3.12 AS builder
WORKDIR /build
COPY pyproject.toml README.md requirements.lock ./
COPY academic_paper/ ./academic_paper/
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.lock && \
    /opt/venv/bin/pip install --no-cache-dir --no-deps .

# Stage 2: Runtime — slim image with only the venv, no build toolchain
FROM python:3.12-slim AS runtime
COPY --from=builder /opt/venv /opt/venv
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
