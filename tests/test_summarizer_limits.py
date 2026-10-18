"""LLM output size caps (#737) and field normalization / no-embedder 4xx (#669)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from academic_paper.summarizer import MAX_FIELD_CHARS, MAX_KEYWORD_CHARS, MAX_KEYWORDS, RAGSummarizer

_TEXT_FIELDS = ("objective", "method", "results", "limitations")


def _summarizer(payload: dict) -> RAGSummarizer:
    llm = AsyncMock()
    llm.generate.return_value = json.dumps(payload, ensure_ascii=False)
    qdrant = MagicMock()
    qdrant.asearch = AsyncMock(return_value=[{"payload": {"page_start": 1, "text": "t"}}])
    return RAGSummarizer(llm, qdrant)


@pytest.mark.anyio
@pytest.mark.parametrize("field", _TEXT_FIELDS)
async def test_text_field_truncated_to_max_chars(field):
    result = await _summarizer({field: "a" * (MAX_FIELD_CHARS + 1)}).summarize(paper_id=1, file_hash="h")
    assert len(result[field]) == MAX_FIELD_CHARS
    # other fields untouched
    assert all(result[f] == "" for f in _TEXT_FIELDS if f != field)


@pytest.mark.anyio
@pytest.mark.parametrize("field", _TEXT_FIELDS)
async def test_text_field_exactly_at_limit_is_unchanged(field):
    value = "b" * MAX_FIELD_CHARS
    result = await _summarizer({field: value}).summarize(paper_id=1, file_hash="h")
    assert result[field] == value


@pytest.mark.anyio
async def test_truncation_keeps_prefix_not_suffix():
    value = "H" * MAX_FIELD_CHARS + "TAIL"
    result = await _summarizer({"objective": value}).summarize(paper_id=1, file_hash="h")
    assert result["objective"] == "H" * MAX_FIELD_CHARS


@pytest.mark.anyio
async def test_keywords_capped_and_leading_ones_kept():
    kws = [f"kw{i}" for i in range(MAX_KEYWORDS + 1)]
    result = await _summarizer({"keywords": kws}).summarize(paper_id=1, file_hash="h")
    assert len(result["keywords"]) == MAX_KEYWORDS
    assert result["keywords"] == kws[:MAX_KEYWORDS]
    assert kws[-1] not in result["keywords"]


@pytest.mark.anyio
async def test_keywords_exactly_at_limit_unchanged():
    kws = [f"kw{i}" for i in range(MAX_KEYWORDS)]
    result = await _summarizer({"keywords": kws}).summarize(paper_id=1, file_hash="h")
    assert result["keywords"] == kws


@pytest.mark.anyio
@pytest.mark.parametrize(
    "length,expected",
    [
        (MAX_KEYWORD_CHARS - 1, MAX_KEYWORD_CHARS - 1),
        (MAX_KEYWORD_CHARS, MAX_KEYWORD_CHARS),
        (MAX_KEYWORD_CHARS + 1, MAX_KEYWORD_CHARS),
    ],
)
async def test_keyword_length_capped(length, expected):
    result = await _summarizer({"keywords": ["k" * length]}).summarize(paper_id=1, file_hash="h")
    assert [len(k) for k in result["keywords"]] == [expected]


@pytest.mark.anyio
async def test_dict_field_serialized_as_json_keeping_non_ascii():
    result = await _summarizer({"objective": {"a": "日本語"}}).summarize(paper_id=1, file_hash="h")
    assert result["objective"] == '{"a": "日本語"}'
    assert "\\u" not in result["objective"]


@pytest.mark.anyio
async def test_no_embedder_qdrant_404_propagates_without_db_fallback():
    qdrant = MagicMock()
    qdrant.vector_size = 4
    qdrant.asearch = AsyncMock(
        side_effect=UnexpectedResponse(status_code=404, reason_phrase="Not Found", content=b"{}", headers={})
    )
    llm = AsyncMock()
    summarizer = RAGSummarizer(llm, qdrant)  # embedder=None
    with (
        patch.object(summarizer, "_chunks_from_db") as chunks_from_db,
        pytest.raises(UnexpectedResponse) as ei,
    ):
        await summarizer.summarize(paper_id=1, file_hash="h")
    assert ei.value.status_code == 404
    chunks_from_db.assert_not_called()
    llm.generate.assert_not_called()
