"""OpenAlex filter-syntax chars in --query must not leak into the filter param (#744)."""

import sys
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import openalex_collect  # noqa: E402


@respx.mock
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("retrieval, augmented", "retrieval augmented"),
        ("cats|dogs", "cats dogs"),
        ("a,,b | c", "a b c"),
    ],
)
def test_query_separators_stripped(query, expected):
    route = respx.get(openalex_collect.OPENALEX_API).mock(
        return_value=httpx.Response(200, json={"results": [], "meta": {"next_cursor": None}})
    )

    openalex_collect.fetch_papers(query, from_date="2025-02-01", until_date="2025-08-01")

    flt = parse_qs(route.calls[0].request.url.query.decode())["filter"][0]
    assert flt == f"title.search:{expected},from_publication_date:2025-02-01,to_publication_date:2025-08-01"
    assert flt.count(",") == 2 and "|" not in flt
