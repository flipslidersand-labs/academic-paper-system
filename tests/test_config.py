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
