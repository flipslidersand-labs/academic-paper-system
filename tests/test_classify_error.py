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


# --- mapped_errors (#747) ------------------------------------------------------


def test_mapped_errors_maps_logs_and_passes_through(caplog):
    from fastapi import HTTPException

    from academic_paper.errors import NotFoundError, mapped_errors

    with pytest.raises(HTTPException) as ei, caplog.at_level(logging.ERROR, logger="academic_paper.errors"):
        with mapped_errors("Failed to x", "x failed for id=%s", 7):
            raise ValueError("boom")
    assert (ei.value.status_code, ei.value.detail) == (400, "Failed to x")  # fallback_msg used as before
    assert "x failed for id=7" in caplog.text

    existing = HTTPException(status_code=404, detail="gone")
    for passthrough in (existing, NotFoundError("nf")):
        with pytest.raises(type(passthrough)) as ei2:
            with mapped_errors("fb", "log"):
                raise passthrough
        assert ei2.value is passthrough


def test_mapped_errors_no_exception_is_noop():
    from academic_paper.errors import mapped_errors

    with mapped_errors("fb", "log"):
        value = 1
    assert value == 1


def test_no_handwritten_exception_to_http_boilerplate():
    """#747: endpoints/services use mapped_errors, not `except Exception` -> `_http_exc_for`.

    Allowed: routers/papers.py ingest_paper (its cleanup/compensation needs bespoke handling).
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "academic_paper"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "errors.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for h in ast.walk(fn):
                if isinstance(h, ast.ExceptHandler) and any(
                    isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_http_exc_for" for n in ast.walk(h)
                ):
                    offenders.append(f"{path.name}:{fn.name}")
    assert sorted(set(offenders)) == ["papers.py:ingest_paper"], offenders


def _status_error(status: int, url: str) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", url)
    return httpx.HTTPStatusError("x", request=req, response=httpx.Response(status, request=req))


def _embed_url() -> str:
    from academic_paper.config import get_settings

    return f"{get_settings().embedding_svc_url}/embed/batch"


@pytest.mark.parametrize(
    ("status", "kind", "http_status"),
    [
        (503, "upstream_unavailable", 503),
        (500, "upstream_unavailable", 503),
        (429, "upstream_unavailable", 503),
        (404, "upstream_protocol", 502),
        (400, "upstream_protocol", 502),
    ],
)
def test_http_status_error_classified_by_status(status, kind, http_status):
    """5xx/429 -> 503, other 4xx -> 502 instead of an opaque 500 (#818)."""
    info = classify_error(_status_error(status, _embed_url()))
    assert (info.kind, info.status) == (kind, http_status)


def test_auth_error_mentions_embedding_key_only_for_embedding_svc():
    """An Ollama 401 must not point the operator at EMBEDDING_API_KEY (#818)."""
    emb = classify_error(_status_error(401, _embed_url()))
    assert (emb.kind, emb.status) == ("upstream_auth", 502)
    assert "EMBEDDING_API_KEY" in emb.public_message

    for status in (401, 403):
        llm = classify_error(_status_error(status, "http://ollama.invalid:11434/api/generate"))
        assert (llm.kind, llm.status) == ("upstream_auth", 502)
        assert "EMBEDDING_API_KEY" not in llm.public_message


def test_safe_error_message_for_http_status_error_is_not_opaque():
    msg = _safe_error_message(_status_error(503, _embed_url()))
    assert not msg.startswith("Internal error")
