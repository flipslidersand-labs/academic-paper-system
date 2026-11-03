"""esearch needs mindate AND maxdate together; one-sided input is ignored (#823)."""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pubmed_collect  # noqa: E402


def _params(**dates) -> dict:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"esearchresult": {"idlist": ["1"]}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        pubmed_collect.fetch_pmc_ids(["x"], 5, client=client, **dates)
    return dict(seen[0].url.params)


def test_from_date_only_fills_maxdate():
    p = _params(from_date="2024-01-01")
    assert p["datetype"] == "pdat"
    assert p["mindate"] == "2024-01-01"
    assert p["maxdate"] == "2099-12-31"


def test_until_date_only_fills_mindate():
    p = _params(until_date="2020-01-01")
    assert p["datetype"] == "pdat"
    assert p["mindate"] == "1800-01-01"
    assert p["maxdate"] == "2020-01-01"


def test_both_dates_unchanged():
    p = _params(from_date="2020-01-01", until_date="2021-01-01")
    assert (p["mindate"], p["maxdate"]) == ("2020-01-01", "2021-01-01")


def test_no_dates_sends_no_date_params():
    p = _params()
    assert not {"datetype", "mindate", "maxdate"} & p.keys()
