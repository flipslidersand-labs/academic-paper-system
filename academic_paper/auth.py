"""API-key authentication with scope checks (#602, parent #355).

api_key / api_keys hold every scope; ingest_api_key holds only the ingest scope
(#626). Endpoints do not yet request scopes, so existing behavior is unchanged.
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


def resolve_scopes(
    provided: str | None, configured: list[str], ingest_keys: list[str] | tuple[str, ...] = ()
) -> frozenset[Scope] | None:
    """Return the scopes granted to ``provided``, or None if it matches no configured key.

    Compares as UTF-8 bytes so a non-ASCII header fails the comparison instead of
    raising TypeError in hmac.compare_digest (#425), and checks every candidate
    without short-circuiting (#601). ``configured`` keys get every scope;
    ``ingest_keys`` get only the ingest scope (#626).
    """
    value = (provided or "").encode("utf-8")
    full = [hmac.compare_digest(value, key.encode("utf-8")) for key in configured]
    ingest = [hmac.compare_digest(value, key.encode("utf-8")) for key in ingest_keys]
    if any(full):
        return ALL_SCOPES
    if any(ingest):
        return frozenset({Scope.INGEST})
    return None


def require_scope(*scopes: Scope) -> Callable[..., Awaitable[None]]:
    """FastAPI dependency factory: 401 for missing/invalid key, 403 if the key lacks a scope.

    Auth disabled (no api_key / api_keys / ingest_api_key configured) passes through.
    """
    required = frozenset(scopes)

    async def dependency(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if not settings.auth_enabled:
            return  # auth disabled
        ingest_keys = [settings.ingest_api_key] if settings.ingest_api_key else []
        granted = resolve_scopes(x_api_key, settings.accepted_api_keys, ingest_keys)
        if granted is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        if not required <= granted:
            raise HTTPException(status_code=403, detail="Insufficient scope")

    return dependency


_require_authenticated = require_scope()  # any valid key, no scope required (#602)


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Require X-API-Key on write and read endpoints when API_KEY env var is set (#146).

    Applied to read endpoints too (#241): paper text/search snippets are at least
    as sensitive as the job metadata already gated behind auth (#190), so the
    boundary must not be asymmetric.

    Uses hmac.compare_digest for constant-time comparison to prevent
    timing attacks that could leak key length / prefix (#190).
    """
    await _require_authenticated(x_api_key)
