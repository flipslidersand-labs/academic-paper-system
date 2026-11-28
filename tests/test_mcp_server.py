"""academic_paper_mcp: a read-only MCP server over the REST API.

The server is exercised through the SDK's in-memory client, with the paper API mocked by respx.
"""

import httpx
import pytest
import respx

pytest.importorskip("mcp", reason="install the mcp extra: pip install -e '.[mcp]'")

from mcp.client import Client  # noqa: E402

from academic_paper_mcp.server import build_server, make_client  # noqa: E402

BASE = "http://paper-api.test"
KEY = "sk-test-secret-key"

pytestmark = pytest.mark.anyio


def _server():
    return build_server(make_client(BASE, KEY))


async def _call(name, args=None):
    async with Client(_server()) as client:
        return await client.call_tool(name, args or {})


def _text(result) -> str:
    return " ".join(getattr(block, "text", "") for block in result.content)


async def test_exposes_exactly_the_read_only_tools():
    async with Client(_server()) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == {"search_papers", "get_paper", "get_summary", "list_papers", "library_stats"}
    for tool in tools:
        assert tool.annotations is not None and tool.annotations.read_only_hint is True, tool.name
        assert tool.annotations.destructive_hint is False, tool.name


@respx.mock
async def test_search_defaults_to_nugget_and_sends_the_key_header():
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"mode": "nugget", "query": "rag", "results": []})
    )
    result = await _call("search_papers", {"query": "rag"})

    assert not result.is_error
    request = route.calls.last.request
    assert request.headers["x-api-key"] == KEY
    assert dict(request.url.params) == {"q": "rag", "mode": "nugget", "limit": "5"}


@respx.mock
async def test_search_passes_paper_id_and_mode():
    route = respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json={"results": []}))
    await _call("search_papers", {"query": "x", "mode": "keyword", "limit": 3, "paper_id": 7})
    assert dict(route.calls.last.request.url.params) == {"q": "x", "mode": "keyword", "limit": "3", "paper_id": "7"}


@respx.mock
async def test_get_paper_and_summary_use_the_paper_routes():
    respx.get(f"{BASE}/papers/4").mock(return_value=httpx.Response(200, json={"id": 4, "status": "indexed"}))
    respx.get(f"{BASE}/papers/4/summary").mock(return_value=httpx.Response(200, json={"paper_id": 4, "cached": True}))
    assert "indexed" in _text(await _call("get_paper", {"paper_id": 4}))
    assert "cached" in _text(await _call("get_summary", {"paper_id": 4}))


@respx.mock
async def test_summary_is_read_only_and_never_posts():
    """A paper without a summary is reported as not found; the server must not ask the API to generate one."""
    respx.get(f"{BASE}/papers/9/summary").mock(return_value=httpx.Response(404, json={"detail": "no summary"}))
    post = respx.post(f"{BASE}/papers/9/summary").mock(return_value=httpx.Response(200, json={}))

    result = await _call("get_summary", {"paper_id": 9})

    assert result.is_error
    assert "not found" in _text(result)
    assert not post.called


@respx.mock
async def test_list_papers_omits_unset_filters():
    route = respx.get(f"{BASE}/papers").mock(return_value=httpx.Response(200, json={"total": 0, "papers": []}))
    await _call("list_papers", {"category": "cs.AI"})
    assert dict(route.calls.last.request.url.params) == {
        "limit": "20",
        "offset": "0",
        "sort": "ingested_at",
        "category": "cs.AI",
    }


@respx.mock
async def test_library_stats():
    respx.get(f"{BASE}/stats").mock(return_value=httpx.Response(200, json={"papers": 3, "chunks": 90}))
    assert "90" in _text(await _call("library_stats"))


@respx.mock
@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "missing or invalid"), (403, "lacks the read scope"), (422, "rejected the arguments"), (503, "HTTP 503")],
)
async def test_http_errors_become_short_tool_errors_without_leaking_the_body_or_key(status, expected):
    respx.get(f"{BASE}/stats").mock(
        return_value=httpx.Response(status, json={"detail": "internal-host-1 traceback secret-body"})
    )
    result = await _call("library_stats")

    assert result.is_error
    text = _text(result)
    assert expected in text
    assert KEY not in text and "secret-body" not in text and "internal-host-1" not in text


@respx.mock
async def test_unreachable_api_gives_an_actionable_error():
    respx.get(f"{BASE}/stats").mock(side_effect=httpx.ConnectError("boom: 10.0.0.1 refused"))
    result = await _call("library_stats")

    assert result.is_error
    assert "PAPER_API_URL" in _text(result)
    assert "10.0.0.1" not in _text(result)


def test_make_client_reads_the_environment_and_omits_an_empty_key(monkeypatch):
    monkeypatch.setenv("PAPER_API_URL", "http://example.test:9")
    monkeypatch.setenv("PAPER_API_KEY", "")
    client = make_client()
    assert str(client.base_url).startswith("http://example.test:9")
    assert "x-api-key" not in client.headers
