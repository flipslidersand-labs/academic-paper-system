"""require_scope dependency on a minimal app (#602)."""

from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from academic_paper.auth import ALL_SCOPES, Scope, require_scope, resolve_scopes
from academic_paper.config import settings

_app = FastAPI()


@_app.get("/read", dependencies=[Depends(require_scope(Scope.READ))])
def _read():
    return {"ok": True}


@_app.get("/any", dependencies=[Depends(require_scope())])
def _any():
    return {"ok": True}


@pytest.fixture
def c():
    return TestClient(_app)


def _h(key):
    return {"X-API-Key": key}


def test_missing_key_401(c):
    with patch.object(settings, "api_key", "k"), patch.object(settings, "api_keys", ""):
        assert c.get("/read").status_code == 401


def test_invalid_key_401(c):
    with patch.object(settings, "api_key", "k"), patch.object(settings, "api_keys", ""):
        assert c.get("/read", headers=_h("bad")).status_code == 401


def test_valid_key_passes_both_keys(c):
    with patch.object(settings, "api_key", "k"), patch.object(settings, "api_keys", "k2"):
        assert c.get("/read", headers=_h("k")).status_code == 200
        assert c.get("/any", headers=_h("k2")).status_code == 200


def test_auth_disabled_passes_through(c):
    with patch.object(settings, "api_key", ""), patch.object(settings, "api_keys", ""):
        assert c.get("/read").status_code == 200


def test_insufficient_scope_403(c):
    with (
        patch.object(settings, "api_key", "k"),
        patch.object(settings, "api_keys", ""),
        patch("academic_paper.auth.resolve_scopes", return_value=frozenset({Scope.INGEST})),
    ):
        assert c.get("/read", headers=_h("k")).status_code == 403
        # no scope required: still allowed
        assert c.get("/any", headers=_h("k")).status_code == 200


def test_non_ascii_header_401_not_500(c):
    with patch.object(settings, "api_key", "k"), patch.object(settings, "api_keys", ""):
        r = c.get("/read", headers={b"x-api-key": "caf\xe9".encode("latin-1")})
        assert r.status_code == 401


def test_resolve_scopes_all_for_valid_key_none_otherwise():
    assert resolve_scopes("a", ["a", "b"]) == ALL_SCOPES
    assert resolve_scopes("c", ["a", "b"]) is None
    assert resolve_scopes(None, ["a"]) is None
