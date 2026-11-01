"""Exception classification (HTTP-independent) and its HTTPException mapping (#614, #741)."""

import logging
import sqlite3
import uuid
from dataclasses import dataclass

import httpx
from fastapi import HTTPException
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from academic_paper.embedder import EmbeddingCountMismatchError, EmbeddingResponseError
from academic_paper.llm import LLMOutputError

logger = logging.getLogger(__name__)

__all__ = ["ErrorInfo", "PdfExtractionError", "_http_exc_for", "_safe_error_message", "classify_error"]


class PdfExtractionError(ValueError):
    """pdfplumber/pdfminer could not parse the PDF (corrupt/encrypted): client input -> 400 (#729)."""


@dataclass(frozen=True)
class ErrorInfo:
    """HTTP-independent classification of an exception (#741).

    ``kind`` is a stable label; ``status`` is the HTTP status it maps to (used only by
    the HTTP layer); ``public_message`` is safe to show clients (never ``str(exc)``).
    """

    kind: str
    status: int
    public_message: str


def classify_error(exc: Exception, fallback_msg: str = "") -> ErrorInfo:
    """Classify an exception without logging or building an HTTPException (#148, #741).

    - embedding-svc 401/403 (key rejected) → 502 (#792)
    - Dependency-unavailable (httpx connect/timeout, Qdrant HTTP errors) → 502/503
    - EmbeddingCountMismatchError / EmbeddingResponseError / LLMOutputError → 502
    - builtin TimeoutError (asyncio.wait_for deadlines) → 504 (#423)
    - sqlite3.IntegrityError (file_hash UNIQUE violation) → 409
    - ValueError from input validation (no text, no chunks) → 400
    - Everything else → kind "internal" (500); the caller adds an opaque id and logs.
    """
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403):
        # A config problem the operator must fix, not an unclassified 500. The key value is never included.
        return ErrorInfo("upstream_auth", 502, "Embedding service rejected authentication (check EMBEDDING_API_KEY)")
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return ErrorInfo("upstream_unavailable", 503, "Upstream service unavailable")
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return ErrorInfo("upstream_timeout", 503, "Upstream service timeout or network error")
    if isinstance(exc, TimeoutError):
        # builtin TimeoutError is not an httpx.TimeoutException; it is an expected boundary
        # condition the client should retry/split on, not an unclassified error (#423).
        return ErrorInfo("timeout", 504, fallback_msg or "Processing timed out")
    # The three upstream-protocol errors below are checked before the generic ValueError
    # branch (#336, #730, #644): they are upstream failures, not bad client input.
    if isinstance(exc, EmbeddingCountMismatchError):
        return ErrorInfo("upstream_protocol", 502, "Embedding service returned a mismatched vector count")
    if isinstance(exc, LLMOutputError):
        return ErrorInfo("upstream_protocol", 502, "LLM returned an invalid response")
    if isinstance(exc, EmbeddingResponseError):
        return ErrorInfo("upstream_protocol", 502, "Embedding service returned an invalid response")
    if isinstance(exc, (UnexpectedResponse, ResponseHandlingException)):
        return ErrorInfo("upstream_protocol", 502, "Vector store returned an unexpected response")
    if isinstance(exc, sqlite3.IntegrityError):
        return ErrorInfo("conflict", 409, "Conflict: duplicate record")
    if isinstance(exc, ValueError):
        return ErrorInfo("invalid_input", 400, fallback_msg)
    return ErrorInfo("internal", 500, "")


def _internal_message(exc: Exception) -> str:
    """Opaque 'Internal error [id]' text; the real detail goes only to the log under the same id."""
    error_id = str(uuid.uuid4())[:8]
    logger.error("Unclassified error [%s]: %s", error_id, exc, exc_info=True)
    return f"Internal error [{error_id}]"


def _http_exc_for(exc: Exception, fallback_msg: str) -> HTTPException:
    """Map an exception to an HTTPException: thin status mapping over classify_error (#148, #741)."""
    info = classify_error(exc, fallback_msg)
    detail = _internal_message(exc) if info.kind == "internal" else info.public_message
    return HTTPException(status_code=info.status, detail=detail)


def _safe_error_message(exc: Exception) -> str:
    """Short, classified error text safe to expose in job/score errors (#474).

    Never includes str(exc): httpx/Qdrant/Ollama exception strings embed internal
    URLs, and the job errors reach public Actions logs/artifacts. HTTP-independent:
    builds no HTTPException (#741). Builtin TimeoutError has an empty str(), so it is
    reported by class name.
    """
    if isinstance(exc, TimeoutError):
        return "TimeoutError"
    if isinstance(exc, PdfExtractionError):
        return "Failed to extract text from PDF"
    info = classify_error(exc)
    if info.kind == "internal":
        return _internal_message(exc)
    if info.kind == "invalid_input":
        # Plain ValueError (input validation): the message may carry arbitrary content.
        return type(exc).__name__
    return info.public_message
