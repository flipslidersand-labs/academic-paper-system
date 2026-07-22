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

    async def embed(self, texts: list[str], mode: str = "index") -> list[list[float]]:
        """Embed texts using embedding service.

        Args:
            texts: List of texts to embed
            mode: Embedding mode ("index" or "search")

        Returns:
            List of embedding vectors

        Raises:
            httpx.HTTPError: If request fails
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embed",
                json={"texts": texts, "mode": mode},
                headers={"X-API-Key": self.api_key},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    async def embed_single(self, text: str, mode: str = "search") -> list[float]:
        """Embed single text using embedding service.

        Args:
            text: Text to embed
            mode: Embedding mode ("index" or "search")

        Returns:
            Embedding vector

        Raises:
            httpx.HTTPError: If request fails
        """
        results = await self.embed([text], mode=mode)
        return results[0]
