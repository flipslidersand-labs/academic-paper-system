# Academic Paper System

Paper RAG system for academic paper processing and retrieval.

## Features

- PDF document processing with pdfplumber
- Vector embeddings with Qdrant
- FastAPI-based REST API
- OpenTelemetry instrumentation
- Google Generative AI integration
- Ollama integration for local LLM inference

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

See `.env.example` for environment variables configuration.

## Development

```bash
pytest
```
