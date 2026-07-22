"""Tests for embedder client."""

import pytest
import respx
import httpx

from academic_paper.embedder import EmbedderClient


@pytest.mark.anyio
async def test_embed_returns_vectors():
    """Test that embed returns list of vectors."""
    client = EmbedderClient(
        base_url="http://localhost:9092",
        api_key="test-key"
    )

    with respx.mock:
        # Mock the POST /embed endpoint
        respx.post("http://localhost:9092/embed").mock(
            return_value=httpx.Response(
                200,
                json={
                    "embeddings": [
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                    ]
                }
            )
        )

        result = await client.embed(["hello", "world"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]


@pytest.mark.anyio
async def test_embed_single_returns_vector():
    """Test that embed_single returns single vector."""
    client = EmbedderClient(
        base_url="http://localhost:9092",
        api_key="test-key"
    )

    with respx.mock:
        # Mock the POST /embed endpoint
        respx.post("http://localhost:9092/embed").mock(
            return_value=httpx.Response(
                200,
                json={
                    "embeddings": [
                        [0.1, 0.2, 0.3],
                    ]
                }
            )
        )

        result = await client.embed_single("hello")

        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]


@pytest.mark.anyio
async def test_embed_sends_correct_headers():
    """Test that embed sends correct headers including API key."""
    client = EmbedderClient(
        base_url="http://localhost:9092",
        api_key="test-api-key"
    )

    with respx.mock:
        # Create a route that captures the request
        route = respx.post("http://localhost:9092/embed").mock(
            return_value=httpx.Response(
                200,
                json={"embeddings": [[0.1, 0.2, 0.3]]}
            )
        )

        await client.embed(["hello"], mode="search")

        # Verify the request was made with correct headers
        assert route.called
        request = route.calls[0].request
        assert request.headers["X-API-Key"] == "test-api-key"
        assert request.headers["Content-Type"] == "application/json"
