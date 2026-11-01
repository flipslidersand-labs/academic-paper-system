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
