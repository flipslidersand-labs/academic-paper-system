"""Error-handling / normalization branch tests for RAGSummarizer (#354).

Complements tests/test_summarizer.py. No network, no real Qdrant: embedder/Qdrant/LLM are mocks;
_chunks_from_db tests use a real temporary SQLite DB.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from academic_paper.config import settings
from academic_paper.db import db_connection, save_chunks, save_paper
from academic_paper.summarizer import RAGSummarizer

_OK_SUMMARY = json.dumps({"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]})
_DB_CHUNKS = [{"payload": {"page_start": 1, "text": "db text"}}]


def _qdrant_error(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(status_code=status, reason_phrase="err", content=b"{}", headers={})


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://embedding-svc/embed")
    return httpx.HTTPStatusError(f"{status}", request=request, response=httpx.Response(status, request=request))


def _llm(response: str = _OK_SUMMARY) -> AsyncMock:
    llm = AsyncMock()
    llm.generate.return_value = response
    return llm


def _embedder() -> AsyncMock:
    embedder = AsyncMock()
    embedder.embed_single.return_value = [0.1] * 768
    return embedder


# --- embed_single 5xx -> DB fallback (embedder present) ---------------------------------------


@pytest.mark.anyio
async def test_summarize_embed_500_falls_back_to_db(caplog):
    embedder = AsyncMock()
    embedder.embed_single.side_effect = _http_status_error(500)
    qdrant = MagicMock()
    qdrant.asearch = AsyncMock()
    summarizer = RAGSummarizer(_llm(), qdrant, embedder=embedder)

    with (
        caplog.at_level(logging.WARNING, logger="academic_paper.summarizer"),
        patch.object(summarizer, "_chunks_from_db", return_value=_DB_CHUNKS) as mock_db,
    ):
        result = await summarizer.summarize(paper_id=1, file_hash="abc", title="Test")

    assert result["objective"] == "o"
    mock_db.assert_called_once_with(1, 5)
    qdrant.asearch.assert_not_called()  # never reached the vector search
    assert "server error 500" in caplog.text


# --- no-embedder path: UnexpectedResponse 4xx propagates / 5xx falls back ---------------------


@pytest.mark.anyio
async def test_summarize_no_embedder_qdrant_404_propagates():
    qdrant = MagicMock()
    qdrant.asearch = AsyncMock(side_effect=_qdrant_error(404))
    summarizer = RAGSummarizer(_llm(), qdrant)

    with (
        patch.object(summarizer, "_chunks_from_db") as mock_db,
        pytest.raises(UnexpectedResponse) as exc_info,
    ):
        await summarizer.summarize(paper_id=1, file_hash="abc")

    assert exc_info.value.status_code == 404
    mock_db.assert_not_called()


@pytest.mark.anyio
async def test_summarize_no_embedder_qdrant_500_falls_back_to_db(caplog):
    qdrant = MagicMock()
    qdrant.asearch = AsyncMock(side_effect=_qdrant_error(500))
    summarizer = RAGSummarizer(_llm(), qdrant)

    with (
        caplog.at_level(logging.WARNING, logger="academic_paper.summarizer"),
        patch.object(summarizer, "_chunks_from_db", return_value=_DB_CHUNKS) as mock_db,
    ):
        result = await summarizer.summarize(paper_id=3, file_hash="abc", top_k=2)

    assert result["method"] == "m"
    mock_db.assert_called_once_with(3, 2)
    assert "server error 500" in caplog.text
    # no-embedder path searches with the zero vector
    assert qdrant.asearch.call_args.kwargs["query_vector"] == [0.0] * 768


@pytest.mark.anyio
async def test_summarize_no_embedder_qdrant_499_boundary_propagates_and_600_falls_back():
    for status, should_raise in ((499, True), (600, False)):
        qdrant = MagicMock()
        qdrant.asearch = AsyncMock(side_effect=_qdrant_error(status))
        summarizer = RAGSummarizer(_llm(), qdrant)
        with patch.object(summarizer, "_chunks_from_db", return_value=_DB_CHUNKS) as mock_db:
            if should_raise:
                with pytest.raises(UnexpectedResponse):
                    await summarizer.summarize(paper_id=1, file_hash="abc")
                mock_db.assert_not_called()
            else:
                await summarizer.summarize(paper_id=1, file_hash="abc")
                mock_db.assert_called_once()


# --- LLM nested-value normalization (json.dumps) ----------------------------------------------


@pytest.mark.anyio
async def test_summarize_normalizes_nested_dict_and_list_fields_to_json_strings():
    nested = {"primary": "日本語 goal", "secondary": ["a", "b"]}
    response = json.dumps(
        {
            "objective": nested,
            "method": ["step1", "step2"],
            "results": 95,
            "limitations": None,
            "keywords": ["x", 2],
        }
    )
    qdrant = MagicMock()
    qdrant.asearch = AsyncMock(return_value=[{"payload": {"page_start": 1, "text": "t"}}])
    summarizer = RAGSummarizer(_llm(response), qdrant)

    result = await summarizer.summarize(paper_id=1, file_hash="abc")

    assert result["objective"] == json.dumps(nested, ensure_ascii=False)
    assert "日本語" in result["objective"]  # ensure_ascii=False keeps non-ASCII readable
    assert result["method"] == '["step1", "step2"]'
    assert result["results"] == "95"
    assert result["limitations"] == "null"
    assert result["keywords"] == ["x", "2"]


@pytest.mark.anyio
async def test_summarize_missing_fields_default_to_empty_string():
    qdrant = MagicMock()
    qdrant.asearch = AsyncMock(return_value=[{"payload": {"page_start": 1, "text": "t"}}])
    summarizer = RAGSummarizer(_llm(json.dumps({"objective": "only"})), qdrant)

    result = await summarizer.summarize(paper_id=1, file_hash="abc")

    assert result["objective"] == "only"
    # absent string fields are not backfilled by the summarizer (callers use .get); only keywords is
    assert result.get("method", "") == ""
    assert result.get("results", "") == ""
    assert result.get("limitations", "") == ""
    assert result["keywords"] == []


# --- _chunks_from_db against a real temporary SQLite DB ---------------------------------------


@pytest.fixture
def db_with_chunks(temp_db, monkeypatch):
    monkeypatch.setattr(settings, "academic_db", temp_db)

    def _make(n_chunks: int, page_starts: list | None = None) -> int:
        with db_connection(temp_db) as conn:
            paper_id = save_paper(conn, "p.pdf", f"hash-{n_chunks}-{id(page_starts)}")
            save_chunks(
                conn,
                paper_id,
                [
                    {
                        "chunk_index": i,
                        "page_start": page_starts[i] if page_starts else i + 1,
                        "page_end": i + 1,
                        "text": f"chunk {i}",
                        "token_count": 1,
                        "qdrant_id": f"q-{paper_id}-{i}",
                    }
                    for i in range(n_chunks)
                ],
            )
        return paper_id

    return _make


def _summarizer() -> RAGSummarizer:
    return RAGSummarizer(_llm(), MagicMock())


def test_chunks_from_db_slices_to_top_k_in_index_order(db_with_chunks):
    paper_id = db_with_chunks(5)

    chunks = _summarizer()._chunks_from_db(paper_id, 3)

    assert [c["payload"]["text"] for c in chunks] == ["chunk 0", "chunk 1", "chunk 2"]
    assert [c["payload"]["page_start"] for c in chunks] == [1, 2, 3]
    assert all(c["payload"]["paper_id"] == paper_id for c in chunks)


def test_chunks_from_db_top_k_larger_than_available_returns_all(db_with_chunks):
    paper_id = db_with_chunks(2)

    assert len(_summarizer()._chunks_from_db(paper_id, 10)) == 2


def test_chunks_from_db_defaults_missing_page_start_to_unknown(db_with_chunks):
    paper_id = db_with_chunks(3, page_starts=[None, 0, 7])

    chunks = _summarizer()._chunks_from_db(paper_id, 5)

    # NULL and 0 are both falsy -> "unknown"; a real page is kept
    assert [c["payload"]["page_start"] for c in chunks] == ["unknown", "unknown", 7]


def test_chunks_from_db_empty_for_unknown_paper_and_zero_top_k(db_with_chunks):
    paper_id = db_with_chunks(2)

    assert _summarizer()._chunks_from_db(9999, 5) == []
    assert _summarizer()._chunks_from_db(paper_id, 0) == []


def test_chunks_from_db_only_returns_requested_paper(db_with_chunks):
    first = db_with_chunks(1)
    second = db_with_chunks(2)

    assert len(_summarizer()._chunks_from_db(first, 5)) == 1
    assert len(_summarizer()._chunks_from_db(second, 5)) == 2


@pytest.mark.anyio
async def test_summarize_db_fallback_end_to_end_uses_real_chunks(db_with_chunks):
    """Qdrant 500 -> real _chunks_from_db (unpatched) -> chunk text reaches the LLM prompt."""
    paper_id = db_with_chunks(3, page_starts=[None, 2, 3])
    qdrant = MagicMock()
    qdrant.asearch = AsyncMock(side_effect=_qdrant_error(500))
    llm = _llm()
    summarizer = RAGSummarizer(llm, qdrant, embedder=_embedder())

    await summarizer.summarize(paper_id=paper_id, file_hash="abc", top_k=2, title="T")

    prompt = llm.generate.call_args.args[0]
    assert "Page unknown: chunk 0" in prompt
    assert "Page 2: chunk 1" in prompt
    assert "chunk 2" not in prompt  # top_k=2 slice
