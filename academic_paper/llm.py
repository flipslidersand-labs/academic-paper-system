import asyncio
from abc import ABC, abstractmethod

import httpx
from google.genai import errors as genai_errors

from academic_paper.config import settings
from academic_paper.retry import async_with_retry

_GEMINI_RETRYABLE = (genai_errors.ServerError,)
_OLLAMA_RETRYABLE = (httpx.NetworkError, httpx.TimeoutException)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate text using the LLM.

        Args:
            prompt: The prompt to send to the LLM
            system: Optional system message

        Returns:
            Generated text response
        """

    def close(self) -> None:
        """Release any underlying HTTP resources. Default: no-op (#262)."""

    async def aclose(self) -> None:
        """Async variant of close(). Default: no-op (#262)."""


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    def __init__(self, api_key: str | None = None):
        """Initialize Gemini client.

        Args:
            api_key: Google API key. If None, uses settings.google_api_key
        """
        self.api_key = api_key or settings.google_api_key
        from google import genai
        from google.genai.types import HttpOptions

        self.client = genai.Client(
            api_key=self.api_key,
            http_options=HttpOptions(timeout=settings.gemini_timeout_ms),
        )

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate text using Gemini API.

        Args:
            prompt: The prompt to send to the LLM
            system: Optional system message

        Returns:
            Generated text response
        """
        full_prompt = f"{system}\n{prompt}".strip() if system else prompt

        # generate_content is a sync blocking call; run in thread pool so the
        # event loop remains responsive during multi-second LLM generation (#149).
        # Retries only transient server-side errors (#234); ClientError (4xx) is not retried.
        response = await async_with_retry(
            asyncio.to_thread,
            self.client.models.generate_content,
            model="gemini-2.0-flash",
            contents=full_prompt,
            attempts=3,
            base_delay=1.0,
            exceptions=_GEMINI_RETRYABLE,
        )
        return response.text

    def close(self) -> None:
        """Close the underlying genai.Client HTTP session (#262)."""
        self.client.close()

    async def aclose(self) -> None:
        """Close the underlying genai.Client's async HTTP session (#262)."""
        await self.client.aio.aclose()


class OllamaClient(BaseLLMClient):
    """Client for Ollama HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama service URL. If None, uses settings.ollama_url
            model: Model name. If None, uses settings.ollama_model
            client: Injected persistent AsyncClient (managed by lifespan).
                    If None, a per-call client is created as fallback (tests /
                    direct instantiation without lifespan).
        """
        self.base_url = base_url or settings.ollama_url
        self.model = model or settings.ollama_model
        # Persistent client injected from lifespan; None → per-call fallback.
        self._client = client

    async def _post(self, client: httpx.AsyncClient, prompt: str, system: str) -> str:
        response = await client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate text using Ollama API.

        Args:
            prompt: The prompt to send to the LLM
            system: Optional system message

        Returns:
            Generated text response
        """
        if self._client is not None:
            return await async_with_retry(
                self._post,
                self._client,
                prompt,
                system,
                attempts=3,
                base_delay=1.0,
                exceptions=_OLLAMA_RETRYABLE,
            )
        # Fallback: per-call client (tests / direct instantiation without lifespan).
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            return await async_with_retry(
                self._post,
                client,
                prompt,
                system,
                attempts=3,
                base_delay=1.0,
                exceptions=_OLLAMA_RETRYABLE,
            )


def get_llm_client() -> BaseLLMClient | None:
    """Get appropriate LLM client based on configuration.

    Priority:
    1. If GOOGLE_API_KEY is set, return GeminiClient
    2. Else if OLLAMA_URL is set, return OllamaClient
    3. Otherwise return None

    Returns:
        LLMClient instance or None if no configuration available
    """
    if settings.google_api_key:
        return GeminiClient()
    if settings.ollama_url:
        return OllamaClient()
    return None
