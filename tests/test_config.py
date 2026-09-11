"""Tests for academic_paper/config.py — placeholder URL rejection (#200)."""

import pytest
from pydantic import ValidationError

from academic_paper.config import Settings


def test_placeholder_embedding_url_rejected():
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(embedding_svc_url="http://<internal-host>:9092", qdrant_url="http://localhost:6333")


def test_placeholder_qdrant_url_rejected():
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(embedding_svc_url="http://localhost:9092", qdrant_url="http://<internal-host>:6333")


def test_valid_urls_accepted():
    s = Settings(embedding_svc_url="http://localhost:9092", qdrant_url="http://localhost:6333")
    assert s.embedding_svc_url == "http://localhost:9092"
    assert s.qdrant_url == "http://localhost:6333"


def test_placeholder_api_key_rejected():
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(
            embedding_svc_url="http://localhost:9092",
            qdrant_url="http://localhost:6333",
            api_key="<your-api-key>",
        )


def test_placeholder_embedding_api_key_rejected():
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(
            embedding_svc_url="http://localhost:9092",
            qdrant_url="http://localhost:6333",
            embedding_api_key="<placeholder>",
        )


def test_placeholder_qdrant_api_key_rejected():
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(
            embedding_svc_url="http://localhost:9092",
            qdrant_url="http://localhost:6333",
            qdrant_api_key="<placeholder>",
        )


def test_placeholder_google_api_key_rejected():
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(
            embedding_svc_url="http://localhost:9092",
            qdrant_url="http://localhost:6333",
            google_api_key="<placeholder>",
        )


def test_empty_api_keys_accepted():
    s = Settings(embedding_svc_url="http://localhost:9092", qdrant_url="http://localhost:6333")
    assert s.api_key == ""
    assert s.embedding_api_key == ""
    assert s.qdrant_api_key == ""
    assert s.google_api_key == ""


def test_chunk_overlap_greater_equal_chunk_size_rejected():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(
            embedding_svc_url="http://localhost:9092",
            qdrant_url="http://localhost:6333",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunk_overlap_smaller_than_chunk_size_accepted():
    s = Settings(
        embedding_svc_url="http://localhost:9092",
        qdrant_url="http://localhost:6333",
        chunk_size=512,
        chunk_overlap=64,
    )
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64


@pytest.mark.parametrize(
    "field",
    [
        "embedding_timeout",
        "qdrant_timeout",
        "ollama_timeout",
        "gemini_timeout_ms",
        "max_upload_mb",
        "port",
        "chunk_size",
    ],
)
def test_non_positive_numeric_fields_rejected(field):
    with pytest.raises(ValidationError):
        Settings(
            embedding_svc_url="http://localhost:9092",
            qdrant_url="http://localhost:6333",
            **{field: 0},
        )
