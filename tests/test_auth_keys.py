"""Multi-key auth: API_KEYS alongside API_KEY (#601)."""

from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from academic_paper.config import Settings, settings
from academic_paper.server import verify_api_key

_app = FastAPI()


@_app.get("/protected", dependencies=[Depends(verify_api_key)])
def _protected():
    return {"ok": True}


@pytest.fixture
def c():
    return TestClient(_app)


def _get(c, key=None):
    return c.get("/protected", headers={"X-API-Key": key} if key is not None else {})


def test_api_key_only_unchanged(c):
    with patch.object(settings, "api_key", "a"), patch.object(settings, "api_keys", ""):
        assert _get(c, "a").status_code == 200
        assert _get(c, "b").status_code == 401
        assert _get(c).status_code == 401


def test_api_keys_accept_each_and_reject_others(c):
    with patch.object(settings, "api_key", ""), patch.object(settings, "api_keys", "a,b"):
        assert _get(c, "a").status_code == 200
        assert _get(c, "b").status_code == 200
        assert _get(c, "c").status_code == 401
        assert _get(c).status_code == 401


def test_api_key_and_api_keys_combined(c):
    with patch.object(settings, "api_key", "old"), patch.object(settings, "api_keys", "new"):
        assert _get(c, "old").status_code == 200
        assert _get(c, "new").status_code == 200
        assert _get(c, "other").status_code == 401


@pytest.mark.parametrize("keys", ["a,,b", "a, ,b", ",a,"])
def test_empty_elements_do_not_disable_auth_or_match_empty(c, keys):
    with patch.object(settings, "api_key", ""), patch.object(settings, "api_keys", keys):
        assert _get(c, "").status_code == 401
        assert _get(c, "   ").status_code == 401
        assert _get(c).status_code == 401
        assert _get(c, "zzz").status_code == 401


@pytest.mark.parametrize("keys", [",", "  ,  "])
def test_only_blank_api_keys_means_auth_disabled(c, keys):
    # Blank-only config is equivalent to unset (documented); never an accept-empty-key hole
    with patch.object(settings, "api_key", ""), patch.object(settings, "api_keys", keys):
        assert settings.accepted_api_keys == []
        assert _get(c).status_code == 200


def test_whitespace_around_elements_is_stripped(c):
    with patch.object(settings, "api_key", ""), patch.object(settings, "api_keys", " a , b "):
        assert _get(c, "a").status_code == 200
        assert _get(c, "b").status_code == 200


def test_both_empty_auth_disabled(c):
    with patch.object(settings, "api_key", ""), patch.object(settings, "api_keys", ""):
        assert _get(c).status_code == 200


def test_non_ascii_header_returns_401_not_500(c):
    """Regression (#425) with multiple keys configured."""
    with patch.object(settings, "api_key", "k"), patch.object(settings, "api_keys", "a,b"):
        r = c.get("/protected", headers={b"x-api-key": "caf\xe9".encode("latin-1")})
        assert r.status_code == 401


def test_placeholder_api_keys_rejected():
    with pytest.raises(ValidationError):
        Settings(api_keys="good,<your-api-key>")
    assert Settings(api_keys="a,b").api_keys_list == ["a", "b"]
