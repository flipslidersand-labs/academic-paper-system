FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY academic_paper/ ./academic_paper/

EXPOSE 8020

CMD ["uvicorn", "academic_paper.server:app", "--host", "0.0.0.0", "--port", "8020"]
