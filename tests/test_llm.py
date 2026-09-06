from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from academic_paper.llm import GeminiClient, OllamaClient, get_llm_client


@pytest.mark.anyio
async def test_gemini_client_generate():
    """Test GeminiClient.generate() returns a string response."""
    with patch("google.genai.Client") as mock_genai_client:
        # Mock the genai.Client instance
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        # Mock the generate_content response
        mock_response = MagicMock()
        mock_response.text = "Test response from Gemini"
        mock_client_instance.models.generate_content.return_value = mock_response

        # Create client and generate
        client = GeminiClient(api_key="test-key")
        result = await client.generate("Test prompt", system="System message")

        # Verify the result
        assert isinstance(result, str)
        assert result == "Test response from Gemini"
        mock_client_instance.models.generate_content.assert_called_once()


@pytest.mark.anyio
async def test_ollama_client_generate():
    """Test OllamaClient.generate() fallback path (per-call AsyncClient)."""
    with patch("academic_paper.llm.httpx.AsyncClient") as mock_async_client:
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test response from Ollama"}

        # Mock the async context manager
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance

        # Create client without injected client → exercises fallback path
        client = OllamaClient(base_url="http://localhost:11434", model="mistral")
        result = await client.generate("Test prompt", system="System message")

        # Verify the result
        assert isinstance(result, str)
        assert result == "Test response from Ollama"
        mock_client_instance.post.assert_called_once()


@pytest.mark.anyio
async def test_ollama_client_generate_with_injected_client():
    """Test OllamaClient.generate() reuses an injected persistent AsyncClient (#192)."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Test response from Ollama"}

    persistent_client = AsyncMock()
    persistent_client.post.return_value = mock_response

    # Inject the persistent client — no AsyncClient context manager should be opened
    with patch("academic_paper.llm.httpx.AsyncClient") as mock_async_client:
        client = OllamaClient(base_url="http://localhost:11434", model="mistral", client=persistent_client)
        result = await client.generate("Test prompt", system="System message")

    assert result == "Test response from Ollama"
    persistent_client.post.assert_called_once()
    # Persistent-client path must not open a new AsyncClient
    mock_async_client.assert_not_called()


def test_get_llm_client_returns_gemini_when_api_key_set(monkeypatch):
    """Test get_llm_client returns GeminiClient when GOOGLE_API_KEY is set."""
    # Mock settings to return api_key
    mock_settings = MagicMock()
    mock_settings.google_api_key = "test-api-key"
    mock_settings.ollama_url = ""

    with patch("academic_paper.llm.settings", mock_settings):
        with patch("google.genai.Client"):
            client = get_llm_client()
            assert isinstance(client, GeminiClient)


def test_get_llm_client_returns_ollama_when_url_set(monkeypatch):
    """Test get_llm_client returns OllamaClient when OLLAMA_URL is set."""
    # Mock settings to return ollama url
    mock_settings = MagicMock()
    mock_settings.google_api_key = ""
    mock_settings.ollama_url = "http://localhost:11434"
    mock_settings.ollama_model = "mistral"

    with patch("academic_paper.llm.settings", mock_settings):
        client = get_llm_client()
        assert isinstance(client, OllamaClient)


def test_get_llm_client_returns_none_when_no_config(monkeypatch):
    """Test get_llm_client returns None when no configuration is available."""
    # Mock settings with empty values
    mock_settings = MagicMock()
    mock_settings.google_api_key = ""
    mock_settings.ollama_url = ""

    with patch("academic_paper.llm.settings", mock_settings):
        client = get_llm_client()
        assert client is None
