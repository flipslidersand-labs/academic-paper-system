"""classify_error / _safe_error_message classify google-genai APIError by status code (#861)."""

import pytest
from google.genai import errors as genai_errors

from academic_paper.errors import _safe_error_message, classify_error

_SECRET = "AIzaSyFAKEKEY-secret-detail"


def _client(code: int):
    return genai_errors.ClientError(code, {"error": {"message": _SECRET, "status": "X"}})


def _server(code: int):
    return genai_errors.ServerError(code, {"error": {"message": _SECRET, "status": "X"}})


@pytest.mark.parametrize(
    ("exc", "kind", "status"),
    [
        (_client(401), "upstream_auth", 502),
        (_client(403), "upstream_auth", 502),
        (_client(429), "upstream_unavailable", 503),
        (_server(500), "upstream_unavailable", 503),
        (_server(503), "upstream_unavailable", 503),
        (_client(400), "upstream_protocol", 502),
        (_client(404), "upstream_protocol", 502),
    ],
)
def test_genai_api_error_classification(exc, kind, status):
    info = classify_error(exc, "fb")
    assert (info.kind, info.status) == (kind, status)
    assert _SECRET not in info.public_message
    assert str(exc.code) not in info.public_message
    assert info.public_message


@pytest.mark.parametrize("exc", [_client(401), _client(429), _server(502), _client(400)])
def test_safe_error_message_uses_classification(exc):
    msg = _safe_error_message(exc)
    assert msg == classify_error(exc).public_message
    assert _SECRET not in msg
    assert not msg.startswith("Internal error")
