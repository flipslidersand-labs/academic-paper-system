import pytest
from unittest.mock import AsyncMock, patch, MagicMock
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
    """Test OllamaClient.generate() returns a string response."""
    with patch("academic_paper.llm.httpx.AsyncClient") as mock_async_client:
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test response from Ollama"}

        # Mock the async context manager
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance

        # Create client and generate
        client = OllamaClient(base_url="http://localhost:11434", model="mistral")
        result = await client.generate("Test prompt", system="System message")

        # Verify the result
        assert isinstance(result, str)
        assert result == "Test response from Ollama"
        mock_client_instance.post.assert_called_once()


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
