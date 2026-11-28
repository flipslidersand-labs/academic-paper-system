"""API-key authentication with scope checks (#602, parent #355).

api_key / api_keys hold every scope; ingest_api_key holds only the ingest scope
(#626). Endpoints do not yet request scopes, so existing behavior is unchanged.
"""

import hmac
from collections.abc import Awaitable, Callable
from enum import StrEnum

from fastapi import Header, HTTPException
from starlette.responses import JSONResponse

from academic_paper.config import get_settings


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


def auth_failure(provided: str | None, required: frozenset[Scope]) -> tuple[int, str] | None:
    """(status, detail) when ``provided`` does not satisfy ``required``; None when allowed.

    Shared by the ``require_scope`` dependency and ``PreAuthBodyGuard`` so both answer identically.
    Auth disabled (no api_key / api_keys / ingest_api_key configured) allows everything.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return None
    ingest_keys = [settings.ingest_api_key] if settings.ingest_api_key else []
    granted = resolve_scopes(provided, settings.accepted_api_keys, ingest_keys)
    if granted is None:
        return 401, "Invalid or missing API key"
    if not required <= granted:
        return 403, "Insufficient scope"
    return None


def require_scope(*scopes: Scope) -> Callable[..., Awaitable[None]]:
    """FastAPI dependency factory: 401 for missing/invalid key, 403 if the key lacks a scope.

    Auth disabled (no api_key / api_keys / ingest_api_key configured) passes through.
    """
    required = frozenset(scopes)

    async def dependency(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        failure = auth_failure(x_api_key, required)
        if failure is not None:
            raise HTTPException(status_code=failure[0], detail=failure[1])

    return dependency


class PreAuthBodyGuard:
    """ASGI middleware: authenticate body-carrying routes *before* the request body is read (#899).

    FastAPI parses a multipart body before it resolves route dependencies, so ``require_scope`` alone
    lets an unauthenticated client make the server read (and spool to disk) an arbitrarily large upload.
    This checks the X-API-Key header first and answers 401/403 without touching the body. The route's own
    ``require_scope`` stays in place as a second check. Only ``POST /papers/ingest`` takes a body.
    """

    GUARDED: dict[tuple[str, str], frozenset[Scope]] = {("POST", "/papers/ingest"): frozenset({Scope.INGEST})}

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        required = None
        if scope["type"] == "http":
            path, root = scope["path"], scope.get("root_path", "")
            if root and path.startswith(root + "/"):  # behind a prefix (--root-path): route on the path below it
                path = path[len(root) :]
            required = self.GUARDED.get((scope["method"], path))
        if required is not None:
            # Starlette decodes header bytes as latin-1; do the same so both checks see the same value.
            provided = next((v.decode("latin-1") for k, v in scope["headers"] if k == b"x-api-key"), None)
            failure = auth_failure(provided, required)
            if failure is not None:
                await JSONResponse({"detail": failure[1]}, status_code=failure[0])(scope, receive, send)
                return
        await self.app(scope, receive, send)
