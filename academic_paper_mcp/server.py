"""Read-only MCP server for academic-paper-system.

Wraps the existing REST API so an MCP client (Claude Code, etc.) can search the paper
knowledge base and read stored summaries. It talks to a running API server with an
``X-API-Key``. It only issues GET requests (it never ingests, summarizes or scores), but the API has
no read-only key yet (api_key / api_keys carry every scope), so the key you give it is as powerful
as the one the API server was configured with.

Configuration (environment):
- ``PAPER_API_URL``  base URL of the API server (default ``http://localhost:8020``)
- ``PAPER_API_KEY``  X-API-Key (omit only when the server runs with AUTH_DISABLED)
"""

from __future__ import annotations

import os
from typing import Any, Literal

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

DEFAULT_API_URL = "http://localhost:8020"
REQUEST_TIMEOUT = 60.0  # hybrid / nugget search embeds the query, which can take a while

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

INSTRUCTIONS = (
    "Search a personal knowledge base of academic papers and read their stored summaries. "
    "Start with search_papers (mode 'nugget' returns only the sentences most relevant to the query, "
    "which keeps context small), then get_paper / get_summary for the papers it points to. "
    "Cite results by paper_id. Summaries exist only for papers that were summarized beforehand."
)


def make_client(base_url: str | None = None, api_key: str | None = None) -> httpx.AsyncClient:
    """Build the HTTP client from arguments or the environment. The key is sent only as a header."""
    base_url = base_url or os.environ.get("PAPER_API_URL") or DEFAULT_API_URL
    api_key = api_key if api_key is not None else os.environ.get("PAPER_API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}
    # trust_env=False: do not route requests for the API (often an internal host) through a proxy.
    return httpx.AsyncClient(base_url=base_url, headers=headers, timeout=REQUEST_TIMEOUT, trust_env=False)


async def _get(client: httpx.AsyncClient, path: str, *, what: str, params: dict[str, Any] | None = None) -> Any:
    """GET ``path`` and return the JSON body, or raise a ToolError with a short, actionable message.

    Messages never include the API key, the response body or the exception text, so nothing
    sensitive from either side reaches the model.
    """
    try:
        response = await client.get(path, params=params)
    except httpx.TimeoutException:
        raise ToolError(f"{what}: the paper API did not answer in time") from None
    except httpx.HTTPError:
        raise ToolError(f"{what}: cannot reach the paper API (check PAPER_API_URL and that it is running)") from None
    status = response.status_code
    if status == 401:
        raise ToolError(f"{what}: the API key is missing or invalid (set PAPER_API_KEY)")
    if status == 403:
        raise ToolError(f"{what}: the API key lacks the read scope")
    if status == 404:
        raise ToolError(f"{what}: not found")
    if status == 422:
        raise ToolError(f"{what}: the API rejected the arguments")
    if status >= 400:
        raise ToolError(f"{what}: the paper API failed (HTTP {status})")
    return response.json()


def build_server(client: httpx.AsyncClient | None = None) -> MCPServer:
    """Create the MCP server. ``client`` is injectable for tests; by default it is built from the environment."""
    http = client or make_client()
    server = MCPServer("academic-papers", instructions=INSTRUCTIONS)

    @server.tool(annotations=_READ_ONLY)
    async def search_papers(
        query: str,
        mode: Literal["nugget", "hybrid", "vector", "keyword"] = "nugget",
        limit: int = 5,
        paper_id: int | None = None,
    ) -> Any:
        """Search the papers. Returns ranked chunks with paper_id, page and text.

        mode: 'nugget' (default) keeps only the sentences most relevant to the query; 'hybrid' mixes
        keyword (BM25) and vector search and returns a 200-character snippet; 'vector' and 'keyword'
        use one method each. limit is 1-100. paper_id restricts the search to one paper.
        """
        params: dict[str, Any] = {"q": query, "mode": mode, "limit": limit}
        if paper_id is not None:
            params["paper_id"] = paper_id
        return await _get(http, "/search", what="search_papers", params=params)

    @server.tool(annotations=_READ_ONLY)
    async def get_paper(paper_id: int) -> Any:
        """Get one paper's metadata (title, authors, categories, published date, status)."""
        return await _get(http, f"/papers/{paper_id}", what="get_paper")

    @server.tool(annotations=_READ_ONLY)
    async def get_summary(paper_id: int) -> Any:
        """Get the stored structured summary (objective, method, results, limitations, keywords).

        Fails with 'not found' when the paper has no summary yet; this tool never generates one.
        """
        return await _get(http, f"/papers/{paper_id}/summary", what="get_summary")

    @server.tool(annotations=_READ_ONLY)
    async def list_papers(
        limit: int = 20,
        offset: int = 0,
        author: str | None = None,
        category: str | None = None,
        sort: Literal["ingested_at", "score"] = "ingested_at",
    ) -> Any:
        """List papers. author is a partial match; category is exact (e.g. cs.AI).

        sort: 'ingested_at' is newest first; 'score' is highest relevance first. limit is 1-100.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset, "sort": sort}
        if author:
            params["author"] = author
        if category:
            params["category"] = category
        return await _get(http, "/papers", what="list_papers", params=params)

    @server.tool(annotations=_READ_ONLY)
    async def library_stats() -> Any:
        """Counts for the knowledge base: papers, chunks and vector points."""
        return await _get(http, "/stats", what="library_stats")

    return server
