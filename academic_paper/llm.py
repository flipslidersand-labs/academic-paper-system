from abc import ABC, abstractmethod
import httpx
from academic_paper.config import settings


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


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    def __init__(self, api_key: str | None = None):
        """Initialize Gemini client.

        Args:
            api_key: Google API key. If None, uses settings.google_api_key
        """
        self.api_key = api_key or settings.google_api_key
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate text using Gemini API.

        Args:
            prompt: The prompt to send to the LLM
            system: Optional system message

        Returns:
            Generated text response
        """
        full_prompt = f"{system}\n{prompt}".strip() if system else prompt

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
        )
        return response.text


class OllamaClient(BaseLLMClient):
    """Client for Ollama HTTP API."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        """Initialize Ollama client.

        Args:
            base_url: Ollama service URL. If None, uses settings.ollama_url
            model: Model name. If None, uses settings.ollama_model
        """
        self.base_url = base_url or settings.ollama_url
        self.model = model or settings.ollama_model

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate text using Ollama API.

        Args:
            prompt: The prompt to send to the LLM
            system: Optional system message

        Returns:
            Generated text response
        """
        async with httpx.AsyncClient() as client:
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
            data = response.json()
            return data.get("response", "")


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
