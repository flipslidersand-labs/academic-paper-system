"""PreAuthBodyGuard: POST /papers/ingest is authenticated before the multipart body is read (#899)."""

import asyncio
from unittest.mock import patch

import httpx
import pytest
from fastapi import Depends, FastAPI, File, UploadFile

from academic_paper.auth import PreAuthBodyGuard, Scope, require_scope
from academic_paper.config import get_settings

_BOUNDARY = "xBOUNDARYx"
_CHUNK = b"\0" * (64 * 1024)
_CHUNKS = 160  # ~10 MiB


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.post("/papers/ingest", dependencies=[Depends(require_scope(Scope.INGEST))])
    async def ingest(file: UploadFile = File(...)):
        return {"ok": True, "filename": file.filename}

    @app.post("/other", dependencies=[Depends(require_scope(Scope.ADMIN))])
    async def other():
        return {"ok": True}

    app.add_middleware(PreAuthBodyGuard)
    return app


class _Counting:
    """ASGI wrapper that counts request-body bytes the app consumed before it answered."""

    def __init__(self, app):
        self.app = app
        self.read = 0
        self.read_at_response = None

    async def __call__(self, scope, receive, send):
        async def rx():
            msg = await receive()
            if msg["type"] == "http.request":
                self.read += len(msg.get("body", b""))
            return msg

        async def tx(msg):
            if msg["type"] == "http.response.start" and self.read_at_response is None:
                self.read_at_response = self.read
            await send(msg)

        await self.app(scope, rx, tx)


async def _body():
    yield f'--{_BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n\r\n%PDF-'.encode()
    for _ in range(_CHUNKS):
        yield _CHUNK
    yield f"\r\n--{_BOUNDARY}--\r\n".encode()


async def _post_async(path: str, key: str | None, counting: _Counting, root_path: str = ""):
    headers: dict[str, str | bytes] = {"content-type": f"multipart/form-data; boundary={_BOUNDARY}"}
    if key is not None:
        headers["X-API-Key"] = key.encode("latin-1")  # bytes: httpx rejects non-ASCII str headers
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=counting, root_path=root_path), base_url="http://t"
    ) as c:
        return await c.post(path, content=_body(), headers=headers)


def _post(path: str, key: str | None, counting: _Counting, root_path: str = ""):
    return asyncio.run(_post_async(path, key, counting, root_path))


@pytest.fixture
def keys():
    s = get_settings()
    with patch.object(s, "api_key", "full"), patch.object(s, "api_keys", ""), patch.object(s, "ingest_api_key", "ing"):
        yield


@pytest.mark.parametrize("key", [None, "wrong", "ï"])
def test_bad_key_is_rejected_without_reading_the_body(keys, key):
    counting = _Counting(_make_app())
    r = _post("/papers/ingest", key, counting)
    assert r.status_code == 401
    assert r.json() == {"detail": "Invalid or missing API key"}
    assert counting.read_at_response is not None
    assert counting.read_at_response < len(_CHUNK) * _CHUNKS // 10  # not the ~10 MiB upload


def test_bad_key_is_rejected_before_the_body_behind_a_root_path(keys):
    """--root-path: scope["path"] carries the prefix; the guard must still match /papers/ingest."""
    counting = _Counting(_make_app())
    r = _post("/api/papers/ingest", None, counting, root_path="/api")
    assert r.status_code == 401
    assert counting.read_at_response < len(_CHUNK) * _CHUNKS // 10


@pytest.mark.parametrize("key", ["full", "ing"])
def test_valid_key_reaches_the_route(keys, key):
    counting = _Counting(_make_app())
    r = _post("/papers/ingest", key, counting)
    assert r.status_code == 200
    assert r.json()["filename"] == "a.pdf"


def test_ingest_scope_is_not_demanded_of_other_routes(keys):
    """The guard only covers POST /papers/ingest; /other keeps its own require_scope (403 for an ingest key)."""
    counting = _Counting(_make_app())
    r = _post("/other", "ing", counting)
    assert r.status_code == 403


def test_auth_disabled_passes_through():
    s = get_settings()
    with patch.object(s, "api_key", ""), patch.object(s, "api_keys", ""), patch.object(s, "ingest_api_key", ""):
        r = _post("/papers/ingest", None, _Counting(_make_app()))
    assert r.status_code == 200


def test_real_app_rejects_before_the_body_and_keeps_security_headers(keys):
    from academic_paper.server import app

    counting = _Counting(app)
    r = _post("/papers/ingest", None, counting)
    assert r.status_code == 401
    assert counting.read_at_response < len(_CHUNK) * _CHUNKS // 10
    assert r.headers["x-content-type-options"] == "nosniff"
