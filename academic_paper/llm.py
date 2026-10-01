import asyncio
from abc import ABC, abstractmethod

import httpx
from google.genai import errors as genai_errors

from academic_paper.config import settings
from academic_paper.http_client import client_or_temporary
from academic_paper.retry import async_with_retry

_GEMINI_RETRYABLE = (genai_errors.ServerError, httpx.NetworkError, httpx.TimeoutException)
_OLLAMA_RETRYABLE = (httpx.NetworkError, httpx.TimeoutException)

GEMINI_MODEL = "gemini-2.0-flash"  # default for settings.gemini_model


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

    async def aclose(self) -> None:
        """Release any underlying HTTP resources. Default: no-op (#262)."""

    @property
    def display_name(self) -> str:
        """Model name to record alongside generated summaries (#347).

        Default falls back to the class name; concrete clients override this
        to report their actual configured model.
        """
        return self.__class__.__name__


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize Gemini client.

        Args:
            api_key: Google API key. If None, uses settings.google_api_key
            model: Model name. If None, uses settings.gemini_model
        """
        self.api_key = api_key or settings.google_api_key
        self.model = model or settings.gemini_model
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
        # Retries transient server-side errors and timeout/network errors (#234, #264);
        # ClientError (4xx) is not retried.
        response = await async_with_retry(
            asyncio.to_thread,
            self.client.models.generate_content,
            model=self.model,
            contents=full_prompt,
            attempts=3,
            base_delay=1.0,
            exceptions=_GEMINI_RETRYABLE,
        )
        return response.text

    @property
    def display_name(self) -> str:
        """Model name recorded alongside generated summaries (#347)."""
        return self.model

    async def aclose(self) -> None:
        """Close both the sync and async genai.Client HTTP sessions.

        generate() only ever uses the sync client (via asyncio.to_thread), so
        aclose() must also close it or its connection pool leaks until process
        exit (#304).
        """
        await asyncio.to_thread(self.client.close)
        await self.client.aio.aclose()


class OllamaClient(BaseLLMClient):
    """Client for Ollama HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        client: httpx.AsyncClient | None = None,
        owns_client: bool = False,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama service URL. If None, uses settings.ollama_url
            model: Model name. If None, uses settings.ollama_model
            client: Injected persistent AsyncClient (managed by lifespan).
                    If None, a per-call client is created as fallback (tests /
                    direct instantiation without lifespan).
            owns_client: If True, aclose() closes ``client``. Default False:
                    an injected client is closed by whoever injected it (#498).
        """
        self.base_url = base_url or settings.ollama_url
        self.model = model or settings.ollama_model
        # Persistent client injected from lifespan; None → per-call fallback.
        self._client = client
        self._owns_client = owns_client

    async def aclose(self) -> None:
        """Close the HTTP client only if this instance owns it (#498)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    @property
    def display_name(self) -> str:
        """Model name recorded alongside generated summaries (#347)."""
        return f"ollama/{self.model}"

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
        async with client_or_temporary(self._client, timeout=settings.ollama_timeout) as client:
            return await async_with_retry(
                self._post,
                client,
                prompt,
                system,
                attempts=3,
                base_delay=1.0,
                exceptions=_OLLAMA_RETRYABLE,
            )


def get_llm_client(http_client: httpx.AsyncClient | None = None) -> BaseLLMClient | None:
    """Get LLM client based on settings.llm_provider.

    - auto (default): GeminiClient if GOOGLE_API_KEY is set, else OllamaClient
      if OLLAMA_URL is set, else None
    - gemini: GeminiClient; raises ValueError if GOOGLE_API_KEY is empty
    - ollama: OllamaClient
    - none: None (LLM disabled)

    ``http_client`` is injected into OllamaClient (not owned: the caller closes it).
    """
    provider = settings.llm_provider
    if provider == "none":
        return None
    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("LLM_PROVIDER=gemini requires GOOGLE_API_KEY to be set")
        return GeminiClient()
    if provider == "ollama":
        return OllamaClient(client=http_client)
    if settings.google_api_key:
        return GeminiClient()
    if settings.ollama_url:
        return OllamaClient(client=http_client)
    return None
