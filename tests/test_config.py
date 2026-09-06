"""Tests for Settings validator that rejects placeholder URLs."""

import pytest
from pydantic import ValidationError

from academic_paper.config import Settings


def test_settings_default_imports_without_error():
    """Default placeholder values are allowed (not set via env var)."""
    s = Settings()
    assert "<internal-host>" in s.embedding_svc_url
    assert "<internal-host>" in s.qdrant_url


def test_settings_rejects_placeholder_embedding_url_when_set(monkeypatch):
    """Explicit env-var placeholder in EMBEDDING_SVC_URL raises ValidationError."""
    monkeypatch.setenv("EMBEDDING_SVC_URL", "http://<internal-host>:9092")
    with pytest.raises(ValidationError, match="Set .* via environment variable"):
        Settings()


def test_settings_rejects_placeholder_qdrant_url_when_set(monkeypatch):
    """Explicit env-var placeholder in QDRANT_URL raises ValidationError."""
    monkeypatch.setenv("QDRANT_URL", "http://<internal-host>:6333")
    with pytest.raises(ValidationError, match="Set .* via environment variable"):
        Settings()


def test_settings_accepts_real_urls(monkeypatch):
    """Real URLs set via env var pass validation."""
    monkeypatch.setenv("EMBEDDING_SVC_URL", "http://localhost:9092")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    s = Settings()
    assert s.embedding_svc_url == "http://localhost:9092"
    assert s.qdrant_url == "http://localhost:6333"
