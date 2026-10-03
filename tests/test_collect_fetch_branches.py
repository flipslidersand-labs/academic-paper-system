"""Error-handling / pagination branch tests for the openalex, pubmed and semantic_scholar collectors (#388).

Complements tests/test_collect_scripts.py. No network: httpx is mocked with respx and time.sleep is patched.
"""

import time
from urllib.parse import parse_qs

import httpx
import openalex_collect
import pubmed_collect
import pytest
import respx
import semantic_scholar_collect


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Replace time.sleep everywhere (module attribute access) and record the requested delays."""
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


# ---------------------------------------------------------------- OpenAlex fetch_papers


def _oa_work(n: int, pdf: bool = True) -> dict:
    work = {"id": f"https://openalex.org/W{n}", "title": f"Paper {n}", "ids": {}, "open_access": {}}
    if pdf:
        work["ids"] = {"arxiv": f"https://arxiv.org/abs/2501.{n:05d}"}
    return work


def _oa_page(works: list[dict], next_cursor: str | None) -> httpx.Response:
    return httpx.Response(200, json={"results": works, "meta": {"next_cursor": next_cursor}})


@respx.mock
def test_openalex_paginates_with_cursor_until_max(sleeps):
    route = respx.get(openalex_collect.OPENALEX_API).mock(
        side_effect=[
            _oa_page([_oa_work(1), _oa_work(2)], "cur-2"),
            _oa_page([_oa_work(3), _oa_work(4)], "cur-3"),
        ]
    )

    papers, err = openalex_collect.fetch_papers("rag", max_results=3)

    assert err is None
    assert [p["id"].rsplit("/", 1)[-1] for p in papers] == ["W1", "W2", "W3"]  # truncated at max
    assert route.call_count == 2
    cursors = [parse_qs(c.request.url.query.decode())["cursor"][0] for c in route.calls]
    assert cursors == ["*", "cur-2"]
    assert all("_pdf_url" in p for p in papers)
    assert sleeps == [0.5]  # polite-pool delay between the two pages only


@respx.mock
def test_openalex_stops_when_next_cursor_missing(sleeps):
    route = respx.get(openalex_collect.OPENALEX_API).mock(return_value=_oa_page([_oa_work(1)], None))

    papers, err = openalex_collect.fetch_papers("rag", max_results=10)

    assert err is None
    assert len(papers) == 1
    assert route.call_count == 1
    assert sleeps == []


@respx.mock
def test_openalex_stops_on_empty_results_page(sleeps):
    route = respx.get(openalex_collect.OPENALEX_API).mock(
        side_effect=[_oa_page([_oa_work(1)], "cur-2"), _oa_page([], "cur-3")]
    )

    papers, err = openalex_collect.fetch_papers("rag", max_results=10)

    assert err is None
    assert len(papers) == 1
    assert route.call_count == 2


@respx.mock
def test_openalex_http_error_sets_fetch_error_and_keeps_earlier_pages(sleeps):
    respx.get(openalex_collect.OPENALEX_API).mock(side_effect=[_oa_page([_oa_work(1)], "cur-2"), httpx.Response(503)])

    papers, err = openalex_collect.fetch_papers("rag", max_results=10)

    assert len(papers) == 1  # page 1 results are kept
    assert err is not None and "503" in err


@respx.mock
def test_openalex_first_page_failure_returns_empty_with_error(sleeps):
    respx.get(openalex_collect.OPENALEX_API).mock(return_value=httpx.Response(500))

    papers, err = openalex_collect.fetch_papers("rag")

    assert papers == []
    assert err


@respx.mock
def test_openalex_transport_exception_with_empty_message_falls_back_to_type_name(sleeps):
    respx.get(openalex_collect.OPENALEX_API).mock(side_effect=httpx.ConnectError(""))

    papers, err = openalex_collect.fetch_papers("rag")

    assert papers == []
    assert err == "ConnectError"


@respx.mock
def test_openalex_dedupes_work_ids_across_pages_and_skips_pdfless_and_idless(sleeps):
    no_id = {"title": "no id", "ids": {"arxiv": "https://arxiv.org/abs/2501.99999"}}
    respx.get(openalex_collect.OPENALEX_API).mock(
        side_effect=[
            _oa_page([_oa_work(1), _oa_work(1), _oa_work(2, pdf=False), no_id], "cur-2"),
            _oa_page([_oa_work(1), _oa_work(3)], None),
        ]
    )

    papers, err = openalex_collect.fetch_papers("rag", max_results=10)

    assert err is None
    assert [p["id"].rsplit("/", 1)[-1] for p in papers] == ["W1", "W3"]


@respx.mock
def test_openalex_sends_date_filter_and_per_page_cap(sleeps):
    route = respx.get(openalex_collect.OPENALEX_API).mock(return_value=_oa_page([], None))

    openalex_collect.fetch_papers("rag", from_date="2025-02-01", until_date="2025-08-01", max_results=500)

    q = parse_qs(route.calls[0].request.url.query.decode())
    assert q["filter"] == ["title.search:rag,from_publication_date:2025-02-01,to_publication_date:2025-08-01"]
    assert q["per-page"] == ["200"]  # min(200, max*3)


# ---------------------------------------------------------------- PubMed fetch_paper_metadata

_PMC_XML = b"""<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <front><article-meta>
      <article-id pub-id-type="pmc">123</article-id>
      <title-group><article-title>Good Paper</article-title></title-group>
      <pub-date pub-type="epub"><year>2025</year><month>3</month><day>7</day></pub-date>
    </article-meta></front>
  </article>
  <article>
    <front><article-meta><title-group><article-title>No PMC id</article-title></title-group></article-meta></front>
  </article>
</pmc-articleset>"""


@respx.mock
def test_pubmed_metadata_parses_valid_xml_and_skips_articles_without_pmc_id():
    route = respx.get(pubmed_collect.EFETCH_URL).mock(return_value=httpx.Response(200, content=_PMC_XML))

    with httpx.Client() as client:
        papers = pubmed_collect.fetch_paper_metadata(["123", "456"], client=client, api_key="k")

    assert [(p["pmc_id"], p["title"], p["pub_date"]) for p in papers] == [("123", "Good Paper", "2025-03-07")]
    q = parse_qs(route.calls[0].request.url.query.decode())
    assert q["id"] == ["123,456"]
    assert q["api_key"] == ["k"]


@respx.mock
def test_pubmed_metadata_malformed_xml_returns_empty_and_logs(capsys):
    respx.get(pubmed_collect.EFETCH_URL).mock(return_value=httpx.Response(200, content=b"<pmc-articleset><article>"))

    with httpx.Client() as client:
        papers = pubmed_collect.fetch_paper_metadata(["1"], client=client)

    assert papers == []
    assert "XML parse error" in capsys.readouterr().err


@respx.mock
def test_pubmed_metadata_empty_ids_makes_no_request():
    route = respx.get(pubmed_collect.EFETCH_URL).mock(return_value=httpx.Response(200, content=_PMC_XML))

    with httpx.Client() as client:
        assert pubmed_collect.fetch_paper_metadata([], client=client) == []

    assert not route.called


@respx.mock
def test_pubmed_metadata_http_error_propagates():
    respx.get(pubmed_collect.EFETCH_URL).mock(return_value=httpx.Response(500))

    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        pubmed_collect.fetch_paper_metadata(["1"], client=client)


@respx.mock
def test_pubmed_metadata_omits_api_key_param_when_unset():
    route = respx.get(pubmed_collect.EFETCH_URL).mock(return_value=httpx.Response(200, content=_PMC_XML))

    with httpx.Client() as client:
        pubmed_collect.fetch_paper_metadata(["1"], client=client)

    assert "api_key" not in parse_qs(route.calls[0].request.url.query.decode())


# ---------------------------------------------------------------- Semantic Scholar 429 handling


def _s2_paper(n: int) -> dict:
    return {"paperId": f"p{n}", "title": f"T{n}", "openAccessPdf": {"url": f"https://x/{n}.pdf"}}


def _s2_page(papers: list[dict], total: int | None = None) -> httpx.Response:
    return httpx.Response(200, json={"data": papers, "total": len(papers) if total is None else total})


@respx.mock
def test_s2_429_then_200_sleeps_retry_after_and_retries_once(sleeps):
    route = respx.get(semantic_scholar_collect.S2_API).mock(
        side_effect=[httpx.Response(429, headers={"retry-after": "7"}), _s2_page([_s2_paper(1)])]
    )

    papers, err = semantic_scholar_collect.fetch_papers("rag", max_results=5)

    assert err is None
    assert [p["paperId"] for p in papers] == ["p1"]
    assert route.call_count == 2
    assert sleeps == [7]


@respx.mock
def test_s2_429_without_retry_after_header_defaults_to_15s(sleeps):
    respx.get(semantic_scholar_collect.S2_API).mock(side_effect=[httpx.Response(429), _s2_page([_s2_paper(1)])])

    papers, err = semantic_scholar_collect.fetch_papers("rag", max_results=5)

    assert err is None
    assert len(papers) == 1
    assert sleeps == [15]


@respx.mock
def test_s2_429_twice_reports_fetch_error_after_single_retry(sleeps):
    route = respx.get(semantic_scholar_collect.S2_API).mock(
        side_effect=[httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(429)]
    )

    papers, err = semantic_scholar_collect.fetch_papers("rag", max_results=5)

    assert papers == []
    assert err is not None and "429" in err
    assert route.call_count == 2  # exactly one retry, no loop
    assert sleeps == [1]


@respx.mock
def test_s2_non_numeric_retry_after_is_reported_as_fetch_error_not_crash(sleeps):
    """HTTP-date Retry-After makes int() raise; it is swallowed into fetch_error (no retry happens).

    Characterization test: the collector does not crash, but it also does not wait/retry.
    """
    route = respx.get(semantic_scholar_collect.S2_API).mock(
        return_value=httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    )

    papers, err = semantic_scholar_collect.fetch_papers("rag", max_results=5)

    assert papers == []
    assert err
    assert route.call_count == 1
    assert sleeps == []


@respx.mock
def test_s2_sends_api_key_header_only_when_given(sleeps):
    route = respx.get(semantic_scholar_collect.S2_API).mock(return_value=_s2_page([]))

    semantic_scholar_collect.fetch_papers("rag", api_key="abc")
    semantic_scholar_collect.fetch_papers("rag")

    assert route.calls[0].request.headers["x-api-key"] == "abc"
    assert "x-api-key" not in route.calls[1].request.headers


@respx.mock
def test_s2_paginates_by_offset_and_uses_short_sleep_with_api_key(sleeps):
    # max_results=1 -> limit=3. A full page of papers without openAccessPdf is filtered out,
    # so the loop must advance the offset and fetch the next page.
    no_pdf = [{"paperId": f"n{i}", "title": "x", "openAccessPdf": None} for i in range(3)]
    route = respx.get(semantic_scholar_collect.S2_API).mock(
        side_effect=[_s2_page(no_pdf, total=10), _s2_page([_s2_paper(10)], total=10)]
    )

    papers, err = semantic_scholar_collect.fetch_papers("rag", max_results=1, api_key="k")

    assert err is None
    assert [p["paperId"] for p in papers] == ["p10"]
    offsets = [parse_qs(c.request.url.query.decode())["offset"][0] for c in route.calls]
    assert offsets == ["0", "3"]
    assert sleeps == [0.05]


@respx.mock
def test_s2_stops_when_offset_reaches_total(sleeps):
    route = respx.get(semantic_scholar_collect.S2_API).mock(return_value=_s2_page([_s2_paper(1)], total=1))

    papers, err = semantic_scholar_collect.fetch_papers("rag", max_results=1000)

    assert err is None
    assert len(papers) == 1
    assert route.call_count == 1
