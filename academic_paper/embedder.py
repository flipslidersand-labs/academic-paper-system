"""HTTP client for embedding service."""

import httpx

from academic_paper.config import settings
from academic_paper.retry import async_with_retry

_EMBED_RETRYABLE = (httpx.NetworkError, httpx.TimeoutException)
_BATCH_MAX = 256  # embedding-svc /embed/batch hard limit


class EmbedderClient:
    """Client for embedding service API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url or settings.embedding_svc_url
        self.api_key = api_key or settings.embedding_api_key
        # Injected persistent client (managed by lifespan); None → per-call client.
        self._client = client

    async def embed(self, texts: list[str], mode: str = "index", collection: str = "facts") -> list[list[float]]:
        """Embed texts using /embed/batch, splitting into chunks of at most 256.

        Returns:
            List of embedding vectors in the same order as texts

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        if not texts:
            return []
        results: list[list[float]] = []
        if self._client is not None:
            for i in range(0, len(texts), _BATCH_MAX):
                chunk = texts[i : i + _BATCH_MAX]
                vectors = await async_with_retry(
                    self._embed_batch,
                    self._client,
                    chunk,
                    mode,
                    collection,
                    attempts=3,
                    base_delay=1.0,
                    exceptions=_EMBED_RETRYABLE,
                )
                results.extend(vectors)
        else:
            # Fallback: per-call client (tests / direct instantiation without lifespan).
            async with httpx.AsyncClient(timeout=settings.embedding_timeout) as client:
                for i in range(0, len(texts), _BATCH_MAX):
                    chunk = texts[i : i + _BATCH_MAX]
                    vectors = await async_with_retry(
                        self._embed_batch,
                        client,
                        chunk,
                        mode,
                        collection,
                        attempts=3,
                        base_delay=1.0,
                        exceptions=_EMBED_RETRYABLE,
                    )
                    results.extend(vectors)
        return results

    async def _embed_batch(
        self, client: httpx.AsyncClient, texts: list[str], mode: str, collection: str
    ) -> list[list[float]]:
        response = await client.post(
            f"{self.base_url}/embed/batch",
            json={"texts": texts, "mode": mode, "collection": collection},
            headers={"X-API-Key": self.api_key},
        )
        response.raise_for_status()
        return response.json()["vectors"]

    async def embed_single(self, text: str, mode: str = "search", collection: str = "facts") -> list[float]:
        """Embed single text using embedding service."""
        results = await self.embed([text], mode=mode, collection=collection)
        return results[0]
