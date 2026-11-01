"""classify_error is HTTP-independent and side-effect free (#741)."""

import logging
import sqlite3

import httpx
import pytest

from academic_paper.errors import ErrorInfo, _safe_error_message, classify_error
from academic_paper.llm import LLMOutputError

_REQ = httpx.Request("GET", "http://internal:1")


@pytest.mark.parametrize(
    ("exc", "kind", "status"),
    [
        (httpx.HTTPStatusError("x", request=_REQ, response=httpx.Response(401, request=_REQ)), "upstream_auth", 502),
        (httpx.ConnectError("x", request=_REQ), "upstream_unavailable", 503),
        (httpx.ReadTimeout("x", request=_REQ), "upstream_timeout", 503),
        (TimeoutError(), "timeout", 504),
        (LLMOutputError("x"), "upstream_protocol", 502),
        (sqlite3.IntegrityError("dup"), "conflict", 409),
        (ValueError("x"), "invalid_input", 400),
        (RuntimeError("x"), "internal", 500),
    ],
)
def test_kind_and_status(exc, kind, status):
    info = classify_error(exc, "fb")
    assert isinstance(info, ErrorInfo)
    assert (info.kind, info.status) == (kind, status)


def test_classify_error_never_logs_or_leaks(caplog):
    with caplog.at_level(logging.DEBUG):
        info = classify_error(RuntimeError("secret http://internal:1"), "fb")
    assert caplog.records == []
    assert "internal" not in info.public_message


def test_safe_error_message_logs_only_unclassified(caplog):
    with caplog.at_level(logging.DEBUG, logger="academic_paper.errors"):
        _safe_error_message(httpx.ConnectError("x", request=_REQ))
        assert caplog.records == []
        msg = _safe_error_message(RuntimeError("boom"))
    assert msg.startswith("Internal error [")
    # the id in the job message is findable in the log
    assert msg[len("Internal error [") : -1] in caplog.text


def test_domain_errors_map_to_404_and_503_with_their_message():
    from academic_paper.errors import NotConfiguredError, NotFoundError, _http_exc_for

    nf = classify_error(NotFoundError("Paper not found"))
    nc = classify_error(NotConfiguredError("LLM not configured"))
    assert (nf.kind, nf.status, nf.public_message) == ("not_found", 404, "Paper not found")
    assert (nc.kind, nc.status, nc.public_message) == ("not_configured", 503, "LLM not configured")
    # the same mapping is reused by the generic HTTP path and the job-error text
    assert _http_exc_for(NotFoundError("x"), "fb").status_code == 404
    assert _safe_error_message(NotConfiguredError("LLM not configured")) == "LLM not configured"


def test_app_returns_detail_body_for_domain_error():
    """The app-level handler yields the same {"detail": ...} body as HTTPException (#741)."""
    from fastapi.testclient import TestClient

    from academic_paper.errors import NotFoundError
    from academic_paper.server import app

    @app.get("/__domain_error_probe")
    def _probe():
        raise NotFoundError("Paper not found")

    try:
        r = TestClient(app).get("/__domain_error_probe")
    finally:
        app.router.routes[:] = [x for x in app.router.routes if getattr(x, "path", "") != "/__domain_error_probe"]
    assert (r.status_code, r.json()) == (404, {"detail": "Paper not found"})
