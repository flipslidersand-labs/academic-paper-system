"""arXiv 429/5xx are retried; other 4xx fail immediately (#724)."""

import sys
import time
from pathlib import Path

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import arxiv_collect  # noqa: E402

FEED = (
    '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
    "<id>http://arxiv.org/abs/2501.00001v1</id><title>T</title>"
    "<published>2025-01-01T00:00:00Z</published></entry></feed>"
)


@pytest.fixture
def sleeps(monkeypatch):
    rec: list[float] = []
    monkeypatch.setattr(time, "sleep", rec.append)
    return rec


@respx.mock
def test_503_503_200_succeeds(sleeps):
    route = respx.get(arxiv_collect.ARXIV_API).mock(
        side_effect=[httpx.Response(503), httpx.Response(503), httpx.Response(200, text=FEED)]
    )
    papers = arxiv_collect.fetch_papers(["cs.AI"], 5)
    assert route.call_count == 3
    assert len(papers) == 1


@respx.mock
def test_400_fails_after_single_call(sleeps):
    route = respx.get(arxiv_collect.ARXIV_API).mock(return_value=httpx.Response(400))
    with pytest.raises(httpx.HTTPStatusError):
        arxiv_collect.fetch_papers(["cs.AI"], 5)
    assert route.call_count == 1
    assert sleeps == []


@respx.mock
def test_429_honors_retry_after(sleeps):
    respx.get(arxiv_collect.ARXIV_API).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, text=FEED)]
    )
    assert len(arxiv_collect.fetch_papers(["cs.AI"], 5)) == 1
    assert 7.0 in sleeps


@respx.mock
def test_persistent_503_gives_up_after_three_attempts(sleeps):
    route = respx.get(arxiv_collect.ARXIV_API).mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        arxiv_collect.fetch_papers(["cs.AI"], 5)
    assert route.call_count == 3
