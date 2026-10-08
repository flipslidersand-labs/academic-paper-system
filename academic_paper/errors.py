"""Exception -> HTTPException mapping (moved from server.py, #614)."""

import logging
import sqlite3
import uuid

import httpx
from fastapi import HTTPException
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from academic_paper.embedder import EmbeddingCountMismatchError

logger = logging.getLogger(__name__)

__all__ = ["_http_exc_for"]


def _http_exc_for(exc: Exception, fallback_msg: str) -> HTTPException:
    """Map an exception to an appropriate HTTPException (#148).

    - Dependency-unavailable (httpx connect/timeout, Qdrant HTTP errors) → 502/503
    - EmbeddingCountMismatchError (embedding-svc returned a mismatched vector count) → 502
    - sqlite3.IntegrityError (file_hash UNIQUE violation) → 409
    - ValueError from input validation (no text, no chunks) → 400
    - Everything else → 500 with opaque error-id (details go to logger only)
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return HTTPException(status_code=503, detail="Upstream service unavailable")
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return HTTPException(status_code=503, detail="Upstream service timeout or network error")
    if isinstance(exc, TimeoutError):
        # builtin TimeoutError (raised by asyncio.wait_for on extraction/summarization
        # deadlines, #238/#237/#269) is not a subclass of httpx.TimeoutException, so it
        # falls through to the catch-all below without this check. It's an expected
        # boundary condition the client should retry/split on, not an unclassified
        # error (#423).
        return HTTPException(status_code=504, detail=fallback_msg or "Processing timed out")
    if isinstance(exc, EmbeddingCountMismatchError):
        # Checked before the generic ValueError branch below (#336): this is an
        # upstream protocol failure, not bad client input, so it maps to 502.
        return HTTPException(status_code=502, detail="Embedding service returned a mismatched vector count")
    if isinstance(exc, (UnexpectedResponse, ResponseHandlingException)):
        return HTTPException(status_code=502, detail="Vector store returned an unexpected response")
    if isinstance(exc, sqlite3.IntegrityError):
        return HTTPException(status_code=409, detail="Conflict: duplicate record")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=fallback_msg)
    # Catch-all: 500 with opaque error id; real detail only in log.
    error_id = str(uuid.uuid4())[:8]
    logger.error("Unclassified error [%s]: %s", error_id, exc, exc_info=True)
    return HTTPException(status_code=500, detail=f"Internal error [{error_id}]")
