"""#821: pubmed pub-date assembly must never yield a date the server rejects (422)."""

from datetime import date, timedelta

import pubmed_collect
import pytest

from academic_paper.validators import _validate_published_date


def _pub_date(inner: str) -> str | None:
    xml = (
        "<article><front><article-meta>"
        "<article-id pub-id-type='pmc'>1</article-id>"
        f"<pub-date pub-type='epub'>{inner}</pub-date>"
        "</article-meta></front></article>"
    )
    meta = pubmed_collect._parse_article(pubmed_collect.ET.fromstring(xml))
    assert meta is not None
    return meta["pub_date"]


@pytest.mark.parametrize(
    ("inner", "expected"),
    [
        ("<year>2024</year><month>3</month><day>7</day>", "2024-03-07"),
        ("<year>2024</year>", "2024-01-01"),
        ("<year>2024</year><month/><day/>", "2024-01-01"),
        ("<year>2024</year><month>Jan</month><day>05</day>", "2024-01-01"),
        ("<year>2024</year><month>13</month><day>1</day>", "2024-01-01"),
        ("<year>2024</year><month>2</month><day>31</day>", "2024-02-01"),
        ("<year>2024</year><season>Spring</season>", "2024-01-01"),
        ("<year/><month>3</month>", None),
        ("<year>abcd</year>", None),
    ],
)
def test_pubdate_falls_back_to_valid_date(inner, expected):
    result = _pub_date(inner)
    assert result == expected
    _validate_published_date(result)  # must not raise 422


def test_pubdate_future_is_none():
    future = date.today() + timedelta(days=400)
    assert _pub_date(f"<year>{future.year}</year><month>{future.month}</month><day>1</day>") is None
