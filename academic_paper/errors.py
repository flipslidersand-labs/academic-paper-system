"""Exception classification (HTTP-independent) and its HTTPException mapping (#614, #741)."""

import logging
import sqlite3
import uuid
from dataclasses import dataclass

import httpx
from fastapi import HTTPException
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from academic_paper.config import get_settings
from academic_paper.embedder import EmbeddingCountMismatchError, EmbeddingResponseError
from academic_paper.llm import LLMOutputError

logger = logging.getLogger(__name__)

__all__ = [
    "DomainError",
    "ErrorInfo",
    "NotConfiguredError",
    "NotFoundError",
    "PdfExtractionError",
    "_http_exc_for",
    "_safe_error_message",
    "classify_error",
    "mapped_errors",
]


class DomainError(Exception):
    """Service-layer error with an author-written, client-safe message (#741).

    Raised by services instead of HTTPException; the HTTP layer maps it via classify_error.
    """


class NotFoundError(DomainError):
    """A requested resource does not exist (HTTP 404)."""


class NotConfiguredError(DomainError):
    """A required backend is not configured or not initialized (HTTP 503)."""


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


def _is_embedding_svc(exc: httpx.HTTPStatusError) -> bool:
    """True when the failed request went to embedding-svc (host:port match), not Ollama etc. (#818)."""
    return exc.request.url.netloc == httpx.URL(get_settings().embedding_svc_url).netloc


def _classify_http_status(exc: httpx.HTTPStatusError) -> ErrorInfo:
    """Classify an upstream HTTP error by status, regardless of which client raised it (#792, #818)."""
    status = exc.response.status_code
    if status in (401, 403):
        # A config problem the operator must fix. The key value is never included.
        if _is_embedding_svc(exc):
            return ErrorInfo(
                "upstream_auth", 502, "Embedding service rejected authentication (check EMBEDDING_API_KEY)"
            )
        return ErrorInfo("upstream_auth", 502, "Upstream service rejected authentication")
    if status >= 500 or status == 429:
        return ErrorInfo("upstream_unavailable", 503, "Upstream service unavailable")
    return ErrorInfo("upstream_protocol", 502, "Upstream service rejected the request")


def classify_error(exc: Exception, fallback_msg: str = "") -> ErrorInfo:
    """Classify an exception without logging or building an HTTPException (#148, #741).

    - NotFoundError → 404, NotConfiguredError → 503 (service-layer domain errors, #741)
    - httpx.HTTPStatusError: 401/403 → 502, 5xx/429 → 503, other 4xx → 502 (#792, #818)
    - Dependency-unavailable (httpx connect/timeout, Qdrant HTTP errors) → 502/503
    - EmbeddingCountMismatchError / EmbeddingResponseError / LLMOutputError → 502
    - builtin TimeoutError (asyncio.wait_for deadlines) → 504 (#423)
    - sqlite3.IntegrityError (file_hash UNIQUE violation) → 409
    - ValueError from input validation (no text, no chunks) → 400
    - Everything else → kind "internal" (500); the caller adds an opaque id and logs.
    """
    if isinstance(exc, NotFoundError):
        return ErrorInfo("not_found", 404, str(exc))
    if isinstance(exc, NotConfiguredError):
        return ErrorInfo("not_configured", 503, str(exc))
    if isinstance(exc, httpx.HTTPStatusError):
        return _classify_http_status(exc)
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


class _MappedErrors:
    """Context manager replacing the per-endpoint ``try/except -> _http_exc_for`` boilerplate (#747).

    ``with mapped_errors("Failed to list papers", "Failed to list papers"):`` logs any
    unexpected exception (``log_msg % log_args``) and re-raises it as the HTTPException
    ``_http_exc_for(exc, fallback_msg)`` would build. HTTPException and DomainError pass
    through untouched (the app's DomainError handler maps the latter).
    """

    def __init__(self, fallback_msg: str, log_msg: str, *log_args: object) -> None:
        self.fallback_msg = fallback_msg
        self.log_msg = log_msg
        self.log_args = log_args

    def __enter__(self) -> "_MappedErrors":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None or not isinstance(exc, Exception) or isinstance(exc, (HTTPException, DomainError)):
            return False
        logger.error(self.log_msg, *self.log_args, exc_info=exc)
        raise _http_exc_for(exc, self.fallback_msg) from exc


def mapped_errors(fallback_msg: str, log_msg: str, *log_args: object) -> _MappedErrors:
    """Build the context manager described on ``_MappedErrors`` (#747)."""
    return _MappedErrors(fallback_msg, log_msg, *log_args)
