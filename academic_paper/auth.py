"""API-key authentication with scope checks (#602, parent #355).

Behavior is unchanged from the single-scope era: every accepted key currently
holds every scope, so a valid key always passes and the 403 path is not yet
reachable from real keys. Per-key scope assignment is a later step.
"""

import hmac
from collections.abc import Awaitable, Callable
from enum import StrEnum

from fastapi import Header, HTTPException

from academic_paper.config import settings


class Scope(StrEnum):
    INGEST = "ingest"
    READ = "read"
    ADMIN = "admin"


ALL_SCOPES: frozenset[Scope] = frozenset(Scope)


def resolve_scopes(provided: str | None, configured: list[str]) -> frozenset[Scope] | None:
    """Return the scopes granted to ``provided``, or None if it matches no configured key.

    Compares as UTF-8 bytes so a non-ASCII header fails the comparison instead of
    raising TypeError in hmac.compare_digest (#425), and checks every candidate
    without short-circuiting (#601). Currently every key gets every scope.
    """
    value = (provided or "").encode("utf-8")
    matches = [hmac.compare_digest(value, key.encode("utf-8")) for key in configured]
    return ALL_SCOPES if any(matches) else None


def require_scope(*scopes: Scope) -> Callable[..., Awaitable[None]]:
    """FastAPI dependency factory: 401 for missing/invalid key, 403 if the key lacks a scope.

    Auth disabled (no api_key / api_keys configured) passes through.
    """
    required = frozenset(scopes)

    async def dependency(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        configured = settings.accepted_api_keys
        if not configured:
            return  # auth disabled
        granted = resolve_scopes(x_api_key, configured)
        if granted is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        if not required <= granted:
            raise HTTPException(status_code=403, detail="Insufficient scope")

    return dependency
