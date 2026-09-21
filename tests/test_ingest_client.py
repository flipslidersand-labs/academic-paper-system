"""Tests for scripts/ingest_client.py: auth header (#183) and the real 202+poll contract (#420)."""

import io
import sys
from pathlib import Path

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import ingest_client  # noqa: E402


def _submit(client: httpx.Client) -> dict:
    return ingest_client.submit_and_wait(
        client,
        "http://api.test",
        "x.pdf",
        io.BytesIO(b"%PDF-"),
        {"source": "arxiv"},
        poll_timeout=5,
        poll_interval=0,
    )


def _mock_ingest_and_done_job(paper_id: int = 1, chunks: int = 1):
    """Mock the server's real contract: 202+job_id from POST, then GET /jobs/{id} -> done."""
    submit_route = respx.post("http://api.test/papers/ingest").mock(
        return_value=httpx.Response(202, json={"job_id": "job-1", "paper_id": paper_id})
    )
    respx.get("http://api.test/jobs/job-1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "job-1",
                "status": "done",
                "result": {"paper_id": paper_id, "chunks": chunks, "status": "indexed"},
            },
        )
    )
    return submit_route


@respx.mock
def test_submit_sends_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("PAPER_API_KEY", "sekrit")
    submit_route = _mock_ingest_and_done_job()

    with httpx.Client() as client:
        result = _submit(client)

    assert result == {"paper_id": 1, "chunks": 1, "status": "ingested"}
    assert submit_route.calls[0].request.headers["X-API-Key"] == "sekrit"


@respx.mock
def test_submit_omits_header_when_key_unset(monkeypatch):
    monkeypatch.delenv("PAPER_API_KEY", raising=False)
    submit_route = _mock_ingest_and_done_job()

    with httpx.Client() as client:
        _submit(client)

    assert "X-API-Key" not in submit_route.calls[0].request.headers


@respx.mock
def test_submit_returns_duplicate_on_409():
    respx.post("http://api.test/papers/ingest").mock(return_value=httpx.Response(409))

    with httpx.Client() as client:
        result = _submit(client)

    assert result == {"status": "duplicate"}


@respx.mock
def test_submit_raises_runtime_error_when_job_fails():
    respx.post("http://api.test/papers/ingest").mock(
        return_value=httpx.Response(202, json={"job_id": "job-1", "paper_id": 1})
    )
    respx.get("http://api.test/jobs/job-1").mock(
        return_value=httpx.Response(200, json={"id": "job-1", "status": "failed", "errors": ["boom"]})
    )

    with httpx.Client() as client, pytest.raises(RuntimeError, match="boom"):
        _submit(client)
