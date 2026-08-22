"""HTTP client for embedding service."""

import httpx

from academic_paper.config import settings
from academic_paper.retry import async_with_retry

_EMBED_RETRYABLE = (httpx.NetworkError, httpx.TimeoutException)


class EmbedderClient:
    """Client for embedding service API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        """Initialize embedder client.

        Args:
            base_url: Base URL for embedding service (default: from settings)
            api_key: API key for embedding service (default: from settings)
        """
        self.base_url = base_url or settings.embedding_svc_url
        self.api_key = api_key or settings.embedding_api_key

    async def embed(self, texts: list[str], mode: str = "index", collection: str = "facts") -> list[list[float]]:
        """Embed texts using embedding service (one request per text).

        Args:
            texts: List of texts to embed
            mode: Embedding mode ("index" or "search")
            collection: Qdrant collection name

        Returns:
            List of embedding vectors

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        results = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for text in texts:
                vector = await async_with_retry(
                    self._embed_one,
                    client, text, mode, collection,
                    attempts=3,
                    base_delay=1.0,
                    exceptions=_EMBED_RETRYABLE,
                )
                results.append(vector)
        return results

    async def _embed_one(
        self, client: httpx.AsyncClient, text: str, mode: str, collection: str
    ) -> list[float]:
        response = await client.post(
            f"{self.base_url}/embed",
            json={"text": text, "mode": mode, "collection": collection},
            headers={"X-API-Key": self.api_key},
        )
        response.raise_for_status()
        return response.json()["vector"]

    async def embed_single(self, text: str, mode: str = "search", collection: str = "facts") -> list[float]:
        """Embed single text using embedding service."""
        results = await self.embed([text], mode=mode, collection=collection)
        return results[0]
