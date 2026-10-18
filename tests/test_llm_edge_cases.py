"""LLM client edge cases: Ollama HTTP errors, display_name, provider priority (#664)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from academic_paper.llm import BaseLLMClient, GeminiClient, OllamaClient, get_llm_client


def _response(status=200, json_body=None):
    request = httpx.Request("POST", "http://ollama/api/generate")
    return httpx.Response(status, json=json_body if json_body is not None else {}, request=request)


@pytest.mark.anyio
async def test_ollama_http_500_is_not_retried(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("academic_paper.retry.asyncio.sleep", sleep)
    http = AsyncMock()
    http.post.return_value = _response(500)
    client = OllamaClient(base_url="http://ollama", model="m", client=http)

    with pytest.raises(httpx.HTTPStatusError):
        await client.generate("p")

    assert http.post.call_count == 1
    sleep.assert_not_called()


@pytest.mark.anyio
async def test_ollama_missing_response_key_returns_empty_string():
    http = AsyncMock()
    http.post.return_value = _response(200, {})
    client = OllamaClient(base_url="http://ollama", model="m", client=http)
    assert await client.generate("p") == ""


@pytest.mark.anyio
async def test_ollama_request_payload():
    http = AsyncMock()
    http.post.return_value = _response(200, {"response": "ok"})
    client = OllamaClient(base_url="http://ollama", model="m", client=http)
    assert await client.generate("hello", system="sys") == "ok"
    args, kwargs = http.post.call_args
    assert args[0] == "http://ollama/api/generate"
    assert kwargs["json"] == {"model": "m", "prompt": "hello", "system": "sys", "stream": False}


def test_display_name_gemini():
    with patch("google.genai.Client"):
        assert GeminiClient(api_key="k", model="gemini-2.0-flash").display_name == "gemini-2.0-flash"


def test_display_name_ollama_is_prefixed():
    assert OllamaClient(base_url="http://o", model="mistral").display_name == "ollama/mistral"


def test_display_name_base_falls_back_to_class_name():
    class Dummy(BaseLLMClient):
        async def generate(self, prompt, system=""):
            return ""

    assert Dummy().display_name == "Dummy"


def test_auto_provider_prefers_gemini_when_both_configured():
    s = MagicMock()
    s.llm_provider = "auto"
    s.google_api_key = "key"
    s.ollama_url = "http://localhost:11434"
    s.gemini_model = "gemini-2.0-flash"
    s.ollama_model = "mistral"
    with patch("academic_paper.llm.get_settings", return_value=s), patch("google.genai.Client"):
        client = get_llm_client()
    assert isinstance(client, GeminiClient)
