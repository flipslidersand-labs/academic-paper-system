"""PubMed esearch が 200 でエラー本文を返しても偽成功にしない (#854)."""

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pubmed_collect  # noqa: E402

_KEY = "SECRET-KEY-123"


def _client(body: dict) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)))


def _call(body: dict, api_key: str = _KEY):
    with _client(body) as c:
        return pubmed_collect.fetch_pmc_ids(["x"], 5, client=c, api_key=api_key)


@pytest.mark.parametrize(
    "body",
    [
        {"error": "API rate limit exceeded", "api-key": _KEY},
        {"esearchresult": {"ERROR": "Invalid query", "idlist": []}},
    ],
)
def test_error_bodies_raise_without_leaking_key(body):
    with pytest.raises(RuntimeError) as ei:
        _call(body)
    assert _KEY not in str(ei.value)
    assert _KEY not in pubmed_collect._safe_error(ei.value)


def test_result_error_text_is_included_truncated():
    with pytest.raises(pubmed_collect.EsearchError) as ei:
        _call({"esearchresult": {"ERROR": "Invalid query " + "x" * 500}})
    assert "Invalid query" in str(ei.value)
    assert len(str(ei.value)) < 300


def test_top_level_error_text_not_included():
    with pytest.raises(pubmed_collect.EsearchError) as ei:
        _call({"error": f"bad {_KEY}", "api-key": _KEY})
    assert "bad" not in str(ei.value)


def test_errorlist_with_idlist_is_not_fatal(capsys):
    body = {"esearchresult": {"idlist": ["1", "2"], "errorlist": {"phrasesnotfound": ["zzqqxxnotaword"]}}}
    assert _call(body) == ["1", "2"]
    assert "zzqqxxnotaword" in capsys.readouterr().err


def test_normal_empty_idlist_returns_empty():
    assert _call({"esearchresult": {"idlist": []}}) == []


def test_normal_idlist_returned():
    assert _call({"esearchresult": {"idlist": ["1", "2"]}}) == ["1", "2"]


def test_empty_errorlist_is_not_error():
    body = {"esearchresult": {"idlist": ["1"], "errorlist": {"phrasesnotfound": [], "fieldsnotfound": []}}}
    assert _call(body) == ["1"]


def test_main_writes_fetch_error_and_exits_1(monkeypatch, tmp_path, capsys):
    real_client = httpx.Client

    def _client_factory(*a, **k):
        return real_client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"error": "bad query", "api-key": _KEY}))
        )

    monkeypatch.setattr(pubmed_collect.httpx, "Client", _client_factory)
    out = tmp_path / "s.json"
    monkeypatch.setattr(sys, "argv", ["pubmed_collect.py", "--summary-file", str(out)])
    with pytest.raises(SystemExit) as ei:
        pubmed_collect.main()
    assert ei.value.code == 1
    data = json.loads(out.read_text())
    assert data["fetched"] == 0
    assert data["fetch_error"]
    assert _KEY not in out.read_text()
    assert _KEY not in capsys.readouterr().err
