"""Tests for scripts/ingest_client.py auth header (#183)."""

import sys
from pathlib import Path

import httpx
import respx

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import ingest_client  # noqa: E402


def _submit(client: httpx.Client) -> dict:
    return ingest_client.submit_and_wait(
        client, "http://api.test", "x.pdf", b"%PDF-", {"source": "arxiv"}, poll_timeout=5
    )


@respx.mock
def test_submit_sends_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("PAPER_API_KEY", "sekrit")
    route = respx.post("http://api.test/papers/ingest").mock(
        return_value=httpx.Response(200, json={"paper_id": 1, "chunks": 1})
    )
    with httpx.Client() as client:
        result = _submit(client)

    assert result["status"] == "ingested"
    assert route.calls[0].request.headers["X-API-Key"] == "sekrit"


@respx.mock
def test_submit_omits_header_when_key_unset(monkeypatch):
    monkeypatch.delenv("PAPER_API_KEY", raising=False)
    route = respx.post("http://api.test/papers/ingest").mock(
        return_value=httpx.Response(200, json={"paper_id": 1, "chunks": 1})
    )
    with httpx.Client() as client:
        _submit(client)

    assert "X-API-Key" not in route.calls[0].request.headers
