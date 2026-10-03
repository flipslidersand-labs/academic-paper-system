from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from google.genai import errors as genai_errors

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
async def test_gemini_client_retries_on_server_error(monkeypatch):
    """A transient ServerError from generate_content is retried and succeeds (#234)."""
    monkeypatch.setattr("academic_paper.retry.asyncio.sleep", AsyncMock())

    with patch("google.genai.Client") as mock_genai_client:
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = "Test response from Gemini"
        mock_client_instance.models.generate_content.side_effect = [
            genai_errors.ServerError(503, {"error": "unavailable"}),
            mock_response,
        ]

        client = GeminiClient(api_key="test-key")
        result = await client.generate("Test prompt", system="System message")

        assert result == "Test response from Gemini"
        assert mock_client_instance.models.generate_content.call_count == 2


@pytest.mark.anyio
async def test_gemini_client_retries_on_timeout(monkeypatch):
    """A transient httpx.TimeoutException from generate_content is retried and succeeds (#264)."""
    monkeypatch.setattr("academic_paper.retry.asyncio.sleep", AsyncMock())

    with patch("google.genai.Client") as mock_genai_client:
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = "Test response from Gemini"
        mock_client_instance.models.generate_content.side_effect = [
            httpx.ReadTimeout("timed out"),
            mock_response,
        ]

        client = GeminiClient(api_key="test-key")
        result = await client.generate("Test prompt", system="System message")

        assert result == "Test response from Gemini"
        assert mock_client_instance.models.generate_content.call_count == 2


@pytest.mark.anyio
async def test_gemini_client_does_not_retry_on_client_error():
    """A ClientError (4xx) from generate_content is not retried (#234)."""
    with patch("google.genai.Client") as mock_genai_client:
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        mock_client_instance.models.generate_content.side_effect = genai_errors.ClientError(
            400, {"error": "bad request"}
        )

        client = GeminiClient(api_key="test-key")
        with pytest.raises(genai_errors.ClientError):
            await client.generate("Test prompt")

        assert mock_client_instance.models.generate_content.call_count == 1


@pytest.mark.anyio
async def test_ollama_client_retries_on_timeout(monkeypatch):
    """A ReadTimeout from Ollama is retried and succeeds on the second attempt (#234)."""
    monkeypatch.setattr("academic_paper.retry.asyncio.sleep", AsyncMock())
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Test response from Ollama"}

    persistent_client = AsyncMock()
    persistent_client.post.side_effect = [
        httpx.ReadTimeout("timed out"),
        mock_response,
    ]

    client = OllamaClient(base_url="http://localhost:11434", model="mistral", client=persistent_client)
    result = await client.generate("Test prompt", system="System message")

    assert result == "Test response from Ollama"
    assert persistent_client.post.call_count == 2


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
    mock_settings.llm_provider = "auto"

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
    mock_settings.llm_provider = "auto"
    mock_settings.ollama_model = "mistral"

    with patch("academic_paper.llm.settings", mock_settings):
        client = get_llm_client()
        assert isinstance(client, OllamaClient)


def test_gemini_client_close_and_aclose():
    """GeminiClient.close()/aclose() release the underlying genai.Client HTTP session (#262)."""
    with patch("google.genai.Client") as mock_genai_client:
        mock_client_instance = MagicMock()
        mock_client_instance.aio.aclose = AsyncMock()
        mock_genai_client.return_value = mock_client_instance

        client = GeminiClient(api_key="test-key")
        client.close()
        mock_client_instance.close.assert_called_once()

        import asyncio

        asyncio.run(client.aclose())
        mock_client_instance.aio.aclose.assert_awaited_once()


def test_gemini_client_aclose_also_closes_sync_client():
    """aclose() must also close the sync client since generate() only uses it via to_thread (#304)."""
    with patch("google.genai.Client") as mock_genai_client:
        mock_client_instance = MagicMock()
        mock_client_instance.aio.aclose = AsyncMock()
        mock_genai_client.return_value = mock_client_instance

        client = GeminiClient(api_key="test-key")

        import asyncio

        asyncio.run(client.aclose())
        mock_client_instance.close.assert_called_once()
        mock_client_instance.aio.aclose.assert_awaited_once()


def test_ollama_client_close_and_aclose_are_noop():
    """BaseLLMClient's default close()/aclose() are no-ops for clients without HTTP resources to release (#262)."""
    client = OllamaClient(base_url="http://localhost:11434", model="mistral")
    client.close()

    import asyncio

    asyncio.run(client.aclose())


def test_get_llm_client_returns_none_when_no_config(monkeypatch):
    """Test get_llm_client returns None when no configuration is available."""
    # Mock settings with empty values
    mock_settings = MagicMock()
    mock_settings.google_api_key = ""
    mock_settings.ollama_url = ""
    mock_settings.llm_provider = "auto"

    with patch("academic_paper.llm.settings", mock_settings):
        client = get_llm_client()
        assert client is None


def _provider_settings(provider, google_api_key="", gemini_model="gemini-2.0-flash"):
    s = MagicMock()
    s.llm_provider = provider
    s.google_api_key = google_api_key
    s.gemini_model = gemini_model
    s.ollama_url = "http://localhost:11434"
    s.ollama_model = "mistral"
    return s


def test_get_llm_client_provider_gemini_explicit():
    """llm_provider=gemini returns GeminiClient even when Ollama URL is set (#511)."""
    with patch("academic_paper.llm.settings", _provider_settings("gemini", "k")), patch("google.genai.Client"):
        assert isinstance(get_llm_client(), GeminiClient)


def test_get_llm_client_provider_gemini_without_key_raises():
    """llm_provider=gemini with an empty key is a configuration error (#511)."""
    with patch("academic_paper.llm.settings", _provider_settings("gemini", "")):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            get_llm_client()


def test_get_llm_client_provider_ollama_explicit_ignores_google_key():
    """llm_provider=ollama returns OllamaClient even when a Google key is set (#511)."""
    with patch("academic_paper.llm.settings", _provider_settings("ollama", "k")):
        assert isinstance(get_llm_client(), OllamaClient)


def test_get_llm_client_provider_none_returns_none():
    """llm_provider=none disables the LLM even when keys/URLs are configured (#511)."""
    with patch("academic_paper.llm.settings", _provider_settings("none", "k")):
        assert get_llm_client() is None


def test_get_llm_client_auto_prefers_gemini_over_ollama():
    """auto keeps the legacy priority: Google key wins over Ollama URL (#511)."""
    with patch("academic_paper.llm.settings", _provider_settings("auto", "k")), patch("google.genai.Client"):
        assert isinstance(get_llm_client(), GeminiClient)


@pytest.mark.anyio
async def test_gemini_model_setting_reflected_in_generate_and_display_name():
    """settings.gemini_model drives generate(model=...) and display_name (#511)."""
    with (
        patch("academic_paper.llm.settings", _provider_settings("gemini", "k", "gemini-custom")),
        patch("google.genai.Client") as mock_genai_client,
    ):
        instance = MagicMock()
        instance.models.generate_content.return_value = MagicMock(text="ok")
        mock_genai_client.return_value = instance
        client = GeminiClient()
        await client.generate("p")
        assert instance.models.generate_content.call_args.kwargs["model"] == "gemini-custom"
        assert client.display_name == "gemini-custom"


def test_settings_llm_provider_defaults():
    """Defaults preserve existing behavior (#511)."""
    from academic_paper.config import Settings

    s = Settings(_env_file=None)
    assert s.llm_provider == "auto"
    assert s.gemini_model == "gemini-2.0-flash"
