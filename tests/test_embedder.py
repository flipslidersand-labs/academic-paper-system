"""Tests for embedder client."""

import httpx
import pytest
import respx

from academic_paper.embedder import _BATCH_MAX, EmbedderClient


@pytest.mark.anyio
async def test_embed_returns_vectors():
    """embed() sends one /embed/batch request and returns all vectors."""
    client = EmbedderClient(base_url="http://localhost:9092", api_key="test-key")

    with respx.mock:
        respx.post("http://localhost:9092/embed/batch").mock(
            return_value=httpx.Response(200, json={"vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})
        )

        result = await client.embed(["hello", "world"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]


@pytest.mark.anyio
async def test_embed_single_returns_vector():
    """embed_single() delegates to embed() via /embed/batch."""
    client = EmbedderClient(base_url="http://localhost:9092", api_key="test-key")

    with respx.mock:
        respx.post("http://localhost:9092/embed/batch").mock(
            return_value=httpx.Response(200, json={"vectors": [[0.1, 0.2, 0.3]]})
        )

        result = await client.embed_single("hello")

        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]


@pytest.mark.anyio
async def test_embed_sends_correct_headers():
    """embed() includes X-API-Key header in /embed/batch request."""
    client = EmbedderClient(base_url="http://localhost:9092", api_key="test-api-key")

    with respx.mock:
        route = respx.post("http://localhost:9092/embed/batch").mock(
            return_value=httpx.Response(200, json={"vectors": [[0.1, 0.2, 0.3]]})
        )

        await client.embed(["hello"], mode="search")

        assert route.called
        request = route.calls[0].request
        assert request.headers["X-API-Key"] == "test-api-key"
        assert request.headers["Content-Type"] == "application/json"


@pytest.mark.anyio
async def test_embed_empty_list_returns_empty():
    """embed([]) returns [] without making any HTTP requests."""
    client = EmbedderClient(base_url="http://localhost:9092", api_key="test-key")

    with respx.mock:
        result = await client.embed([])
        assert result == []


@pytest.mark.anyio
async def test_embed_chunks_large_input():
    """embed() splits inputs larger than _BATCH_MAX into multiple requests."""
    client = EmbedderClient(base_url="http://localhost:9092", api_key="test-key")
    n = _BATCH_MAX + 10
    texts = [f"text_{i}" for i in range(n)]
    first_vectors = [[float(i)] * 3 for i in range(_BATCH_MAX)]
    second_vectors = [[float(i + _BATCH_MAX)] * 3 for i in range(10)]

    with respx.mock:
        route = respx.post("http://localhost:9092/embed/batch").mock(
            side_effect=[
                httpx.Response(200, json={"vectors": first_vectors}),
                httpx.Response(200, json={"vectors": second_vectors}),
            ]
        )

        result = await client.embed(texts)

        assert len(result) == n
        assert route.call_count == 2
        assert result[:_BATCH_MAX] == first_vectors
        assert result[_BATCH_MAX:] == second_vectors


@pytest.mark.anyio
async def test_embed_batch_request_body():
    """embed() sends texts, mode, and collection in the request body."""
    client = EmbedderClient(base_url="http://localhost:9092", api_key="test-key")

    with respx.mock:
        route = respx.post("http://localhost:9092/embed/batch").mock(
            return_value=httpx.Response(200, json={"vectors": [[0.1, 0.2]]})
        )

        await client.embed(["hello"], mode="index", collection="papers")

        import json

        body = json.loads(route.calls[0].request.content)
        assert body["texts"] == ["hello"]
        assert body["mode"] == "index"
        assert body["collection"] == "papers"


@pytest.mark.anyio
async def test_embed_reuses_injected_client():
    """Injected persistent client is used instead of creating a new one per call (#143)."""
    with respx.mock:
        respx.post("http://localhost:9092/embed/batch").mock(
            return_value=httpx.Response(200, json={"vectors": [[0.1, 0.2, 0.3]]})
        )
        persistent = httpx.AsyncClient()
        try:
            client = EmbedderClient(base_url="http://localhost:9092", api_key="test-key", client=persistent)
            result = await client.embed(["hello"])
        finally:
            await persistent.aclose()

    assert result == [[0.1, 0.2, 0.3]]


def test_embedding_timeout_default():
    """embedding_timeout defaults to 120s — the qdrant_timeout (10s) must not be reused for embed calls (#153)."""
    from academic_paper.config import Settings

    assert Settings(_env_file=None).embedding_timeout == 120


@pytest.mark.anyio
async def test_embed_fallback_client_uses_embedding_timeout(monkeypatch):
    """Per-call fallback client is created with settings.embedding_timeout, not a hardcoded value (#153)."""
    from academic_paper import embedder as embedder_module

    monkeypatch.setattr(embedder_module.settings, "embedding_timeout", 77)
    captured: dict = {}
    real_async_client = httpx.AsyncClient

    def spy_client(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(embedder_module.httpx, "AsyncClient", spy_client)

    with respx.mock:
        respx.post("http://localhost:9092/embed/batch").mock(
            return_value=httpx.Response(200, json={"vectors": [[0.1]]})
        )
        client = EmbedderClient(base_url="http://localhost:9092", api_key="test-key")
        await client.embed(["hello"])

    assert captured["timeout"] == 77
