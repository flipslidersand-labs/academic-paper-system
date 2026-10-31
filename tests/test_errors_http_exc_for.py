"""Table-driven tests for errors._http_exc_for / _safe_error_message (#671)."""

import logging
import sqlite3

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from academic_paper.embedder import EmbeddingCountMismatchError, EmbeddingResponseError
from academic_paper.errors import _http_exc_for, _safe_error_message
from academic_paper.llm import LLMOutputError

_REQ = httpx.Request("GET", "http://internal-host:9999/x")


def _unexpected():
    return UnexpectedResponse(status_code=500, reason_phrase="err", content=b"{}", headers={})


@pytest.mark.parametrize(
    "exc,status,detail",
    [
        (httpx.ConnectError("refused", request=_REQ), 503, "Upstream service unavailable"),
        (httpx.ConnectTimeout("t", request=_REQ), 503, "Upstream service unavailable"),
        (httpx.ReadTimeout("t", request=_REQ), 503, "Upstream service timeout or network error"),
        (httpx.WriteTimeout("t", request=_REQ), 503, "Upstream service timeout or network error"),
        (httpx.ReadError("r", request=_REQ), 503, "Upstream service timeout or network error"),
        (TimeoutError(), 504, "fb"),
        (EmbeddingCountMismatchError("x"), 502, "Embedding service returned a mismatched vector count"),
        (EmbeddingResponseError("x"), 502, "Embedding service returned an invalid response"),
        (LLMOutputError("empty"), 502, "LLM returned an invalid response"),
        (_unexpected(), 502, "Vector store returned an unexpected response"),
        (ResponseHandlingException(Exception("boom")), 502, "Vector store returned an unexpected response"),
        (sqlite3.IntegrityError("UNIQUE constraint failed"), 409, "Conflict: duplicate record"),
        (ValueError("bad input"), 400, "fb"),
    ],
)
def test_http_exc_for_mapping(exc, status, detail):
    out = _http_exc_for(exc, "fb")
    assert (out.status_code, out.detail) == (status, detail)


def test_timeout_error_without_fallback_uses_default_detail():
    assert _http_exc_for(TimeoutError(), "").detail == "Processing timed out"


def test_connect_timeout_takes_connect_branch_before_generic_timeout():
    # ConnectTimeout is both a ConnectTimeout and a TimeoutException; the
    # "unavailable" message (first branch) must win over "timeout or network error".
    out = _http_exc_for(httpx.ConnectTimeout("t", request=_REQ), "fb")
    assert out.detail == "Upstream service unavailable"


def test_embedding_count_mismatch_wins_over_valueerror_400():
    assert isinstance(EmbeddingCountMismatchError("x"), ValueError)
    assert _http_exc_for(EmbeddingCountMismatchError("x"), "fb").status_code == 502


def test_catch_all_500_is_opaque_and_logs_detail(caplog):
    secret = "postgres://user:pw@internal-host/db"
    with caplog.at_level(logging.ERROR, logger="academic_paper.errors"):
        out = _http_exc_for(RuntimeError(secret), "fb")
    assert out.status_code == 500
    assert out.detail.startswith("Internal error [") and out.detail.endswith("]")
    assert len(out.detail) == len("Internal error [") + 8 + 1
    assert secret not in out.detail
    # The real message goes to the log only, tagged with the same id.
    error_id = out.detail[len("Internal error [") : -1]
    assert any(secret in r.getMessage() and error_id in r.getMessage() for r in caplog.records)


def test_catch_all_500_ids_differ_between_calls():
    a = _http_exc_for(RuntimeError("x"), "fb").detail
    b = _http_exc_for(RuntimeError("x"), "fb").detail
    assert a != b


def test_safe_error_message_hides_internals():
    assert _safe_error_message(TimeoutError()) == "TimeoutError"
    assert _safe_error_message(ValueError("secret text")) == "ValueError"
    assert _safe_error_message(httpx.ConnectError("http://internal:1", request=_REQ)) == "Upstream service unavailable"
    msg = _safe_error_message(RuntimeError("http://internal:1"))
    assert msg.startswith("Internal error [") and "internal" not in msg


def test_llm_output_error_is_upstream_failure_not_client_error():
    """#730: LLMOutputError (a ValueError subclass) maps to 502; plain ValueError stays 400."""
    assert isinstance(LLMOutputError("x"), ValueError)
    assert _http_exc_for(LLMOutputError("x"), "fb").status_code == 502
    assert _http_exc_for(ValueError("x"), "fb").status_code == 400
    # job/score error text is the classified detail, not the bare class name
    assert _safe_error_message(LLMOutputError("secret http://internal")) == "LLM returned an invalid response"


@pytest.mark.parametrize("code", [401, 403])
def test_embedding_svc_auth_rejection_maps_to_502(code):
    """embedding-svc 401/403 (key rejected) -> 502 with a cause, not an opaque 500 (#792)."""
    exc = httpx.HTTPStatusError("denied", request=_REQ, response=httpx.Response(code, request=_REQ))
    http_exc = _http_exc_for(exc, "fb")
    assert http_exc.status_code == 502
    assert "authentication" in http_exc.detail
    assert "EMBEDDING_API_KEY" in http_exc.detail
