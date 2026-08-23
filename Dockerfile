FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

COPY academic_paper/ ./academic_paper/
COPY frontend/ ./frontend/

EXPOSE 8020

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8020/health', timeout=4)"

CMD ["uvicorn", "academic_paper.server:app", "--host", "0.0.0.0", "--port", "8020"]
