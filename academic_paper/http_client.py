"""Shared helper for the "injected persistent client or per-call fallback" pattern."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx


@asynccontextmanager
async def client_or_temporary(client: httpx.AsyncClient | None, *, timeout: float) -> AsyncIterator[httpx.AsyncClient]:
    """Yield the injected client as-is, or open a per-call AsyncClient when none was injected.

    The injected client is owned by the caller (lifespan) and is never closed here.
    """
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(timeout=timeout) as temporary:
        yield temporary
