"""HTTP client for embedding service."""

import httpx

from academic_paper.config import settings


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
            httpx.HTTPError: If request fails
        """
        results = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/embed",
                    json={"text": text, "mode": mode, "collection": collection},
                    headers={"X-API-Key": self.api_key},
                )
                response.raise_for_status()
                data = response.json()
                results.append(data["vector"])
        return results

    async def embed_single(self, text: str, mode: str = "search", collection: str = "facts") -> list[float]:
        """Embed single text using embedding service."""
        results = await self.embed([text], mode=mode, collection=collection)
        return results[0]
