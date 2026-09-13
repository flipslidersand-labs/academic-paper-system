"""Tests for academic_paper.summarizer module."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from academic_paper.config import settings
from academic_paper.summarizer import SYSTEM_PROMPT, RAGSummarizer


@pytest.mark.anyio
async def test_summarize_returns_structured_dict():
    """Test that summarize returns a dict with all required keys."""
    # Setup mocks
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()

    # Valid JSON response from LLM
    valid_response = json.dumps(
        {
            "objective": "To investigate the effects of deep learning on image classification",
            "method": "We used convolutional neural networks and trained on ImageNet dataset with multiple augmentations",
            "results": "Achieved 95% accuracy on test set, 3% improvement over baseline methods",
            "limitations": "Limited to RGB images, requires significant computational resources",
            "keywords": [
                "deep learning",
                "CNN",
                "image classification",
                "neural networks",
                "ImageNet",
                "computer vision",
            ],
        }
    )
    mock_llm.generate.return_value = valid_response

    # Mock Qdrant chunks
    chunks = [
        {"id": "1", "score": 0.9, "payload": {"paper_id": 1, "page_start": 1, "text": "Sample chunk text for page 1"}},
        {"id": "2", "score": 0.85, "payload": {"paper_id": 1, "page_start": 2, "text": "Sample chunk text for page 2"}},
    ]
    mock_qdrant.asearch = AsyncMock(return_value=chunks)

    # Create summarizer and test
    summarizer = RAGSummarizer(mock_llm, mock_qdrant)
    result = await summarizer.summarize(paper_id=1, file_hash="abc123")

    # Assert required keys exist
    assert "objective" in result
    assert "method" in result
    assert "results" in result
    assert "limitations" in result
    assert "keywords" in result

    # Assert values are non-empty
    assert isinstance(result["objective"], str)
    assert isinstance(result["method"], str)
    assert isinstance(result["results"], str)
    assert isinstance(result["limitations"], str)
    assert isinstance(result["keywords"], list)


@pytest.mark.anyio
async def test_summarize_raises_on_invalid_json():
    """Test that summarize raises ValueError when LLM returns invalid JSON."""
    # Setup mocks
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()

    # Invalid JSON response
    mock_llm.generate.return_value = "This is not valid JSON {]"

    # Mock chunks
    chunks = [{"id": "1", "score": 0.9, "payload": {"paper_id": 1, "page_start": 1, "text": "Sample chunk text"}}]
    mock_qdrant.asearch = AsyncMock(return_value=chunks)

    # Create summarizer and test
    summarizer = RAGSummarizer(mock_llm, mock_qdrant)

    # Should raise ValueError
    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        await summarizer.summarize(paper_id=1, file_hash="abc123")


@pytest.mark.anyio
async def test_summarize_calls_llm_with_context():
    """Test that generate() is called with SYSTEM_PROMPT."""
    # Setup mocks
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()

    # Valid response
    valid_response = json.dumps(
        {
            "objective": "Research goal",
            "method": "Methodology approach",
            "results": "Key findings",
            "limitations": "Future work",
            "keywords": ["term1", "term2", "term3"],
        }
    )
    mock_llm.generate.return_value = valid_response

    # Mock chunks
    chunks = [
        {
            "id": "1",
            "score": 0.9,
            "payload": {"paper_id": 1, "page_start": 1, "text": "Important paper content for context"},
        }
    ]
    mock_qdrant.asearch = AsyncMock(return_value=chunks)

    # Create summarizer and call
    summarizer = RAGSummarizer(mock_llm, mock_qdrant)
    await summarizer.summarize(paper_id=1, file_hash="abc123", top_k=8)

    # Assert generate was called with SYSTEM_PROMPT
    assert mock_llm.generate.called
    call_kwargs = mock_llm.generate.call_args[1]
    assert call_kwargs["system"] == SYSTEM_PROMPT

    # Assert the prompt contains context
    call_args = mock_llm.generate.call_args[0]
    prompt = call_args[0]
    assert "Please summarize" in prompt
    assert "Page 1:" in prompt
    assert "Important paper content" in prompt


@pytest.mark.anyio
async def test_summarize_qdrant_attr_error_propagates():
    """Regression (#139): AttributeError from qdrant.search is a real bug and must propagate."""
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.asearch = AsyncMock(side_effect=AttributeError("no search"))
    mock_embedder = AsyncMock()
    mock_embedder.embed_single.return_value = [0.1] * 768

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    with pytest.raises(AttributeError):
        await summarizer.summarize(paper_id=1, file_hash="hash1", title="Test")


@pytest.mark.anyio
async def test_summarize_qdrant_unavailable_falls_back_to_db():
    """Qdrant connection errors (not bugs) fall back to DB chunk order (#139)."""
    from unittest.mock import patch

    from qdrant_client.http.exceptions import ResponseHandlingException

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )
    mock_qdrant = MagicMock()
    mock_qdrant.asearch = AsyncMock(side_effect=ResponseHandlingException("connection refused"))
    mock_embedder = AsyncMock()
    mock_embedder.embed_single.return_value = [0.1] * 768

    db_chunks = [{"text": "chunk text from db", "page_start": 1}]

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    with patch.object(
        summarizer,
        "_chunks_from_db",
        return_value=[{"payload": {"page_start": 1, "text": c["text"]}} for c in db_chunks],
    ):
        result = await summarizer.summarize(paper_id=1, file_hash="hash1", title="Test")

    assert "objective" in result


@pytest.mark.anyio
async def test_summarize_db_fallback_runs_off_event_loop_thread():
    """_chunks_from_db is offloaded via asyncio.to_thread, not called on the event loop (#232).

    Regression guard: a naive synchronous call would run on the current
    thread, which for the running event loop's thread is exactly the bug
    #232 reports. Recording the executing thread's identity proves the
    fallback runs elsewhere.
    """
    import threading
    from unittest.mock import patch

    from qdrant_client.http.exceptions import ResponseHandlingException

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )
    mock_qdrant = MagicMock()
    mock_qdrant.asearch = AsyncMock(side_effect=ResponseHandlingException("connection refused"))
    mock_embedder = AsyncMock()
    mock_embedder.embed_single.return_value = [0.1] * 768

    calling_thread_ids: list[int] = []

    def fake_chunks_from_db(paper_id, top_k):
        calling_thread_ids.append(threading.get_ident())
        return [{"payload": {"page_start": 1, "text": "db text"}}]

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    with patch.object(summarizer, "_chunks_from_db", side_effect=fake_chunks_from_db):
        await summarizer.summarize(paper_id=1, file_hash="hash1", title="Test")

    assert calling_thread_ids
    assert calling_thread_ids[0] != threading.get_ident()


@pytest.mark.anyio
async def test_summarize_raises_when_no_chunks():
    """ValueError raised when chunks list is empty."""
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.asearch = AsyncMock(return_value=[])

    summarizer = RAGSummarizer(mock_llm, mock_qdrant)
    with pytest.raises(ValueError, match="No chunks found"):
        await summarizer.summarize(paper_id=1, file_hash="abc")


@pytest.mark.anyio
async def test_summarize_raises_when_no_valid_text_in_chunks():
    """ValueError raised when chunks exist but all have empty text."""
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.asearch = AsyncMock(return_value=[{"id": "1", "payload": {"paper_id": 1, "page_start": 1, "text": ""}}])

    summarizer = RAGSummarizer(mock_llm, mock_qdrant)
    with pytest.raises(ValueError, match="No valid content"):
        await summarizer.summarize(paper_id=1, file_hash="abc")


@pytest.mark.anyio
async def test_summarize_keywords_not_list_normalized():
    """When LLM returns keywords as a string it is wrapped in a list."""
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.asearch = AsyncMock(
        return_value=[{"id": "1", "score": 0.9, "payload": {"paper_id": 1, "page_start": 1, "text": "sample text"}}]
    )
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": "single keyword"}
    )

    summarizer = RAGSummarizer(mock_llm, mock_qdrant)
    result = await summarizer.summarize(paper_id=1, file_hash="abc")

    assert isinstance(result["keywords"], list)
    assert result["keywords"] == ["single keyword"]


@pytest.mark.anyio
async def test_summarize_uses_embedder_with_title():
    """When embedder is provided, embed_single is called with the paper title."""
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_embedder = AsyncMock()

    query_vec = [0.5] * 768
    mock_embedder.embed_single.return_value = query_vec

    chunks = [{"id": "1", "score": 0.9, "payload": {"paper_id": 1, "page_start": 1, "text": "paper text"}}]
    mock_qdrant.asearch = AsyncMock(return_value=chunks)
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    await summarizer.summarize(paper_id=1, file_hash="abc", title="My Paper Title")

    mock_embedder.embed_single.assert_called_once_with("My Paper Title", mode="search")
    mock_qdrant.asearch.assert_called_once_with(query_vector=query_vec, limit=5, paper_id_filter=1)


@pytest.mark.anyio
async def test_summarize_uses_filename_when_no_title():
    """When embedder is provided and title is None, file_name (sans .pdf) is used."""
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed_single.return_value = [0.1] * 768

    chunks = [{"id": "1", "score": 0.9, "payload": {"paper_id": 1, "page_start": 1, "text": "text"}}]
    mock_qdrant.asearch = AsyncMock(return_value=chunks)
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    await summarizer.summarize(paper_id=1, file_hash="abc", title=None, file_name="my_paper.pdf")

    mock_embedder.embed_single.assert_called_once_with("my_paper", mode="search")


@pytest.mark.anyio
async def test_summarize_falls_back_to_db_on_embed_http_failure():
    """When embed_single raises httpx.HTTPError, summarizer falls back to DB chunks (#187).

    httpx.HTTPError covers network/HTTP errors from the embedding-svc (the
    expected transient failure mode). A zero-vector search would cache a
    degraded summary, so DB chunk order is used instead.
    """
    from unittest.mock import patch

    import httpx

    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed_single.side_effect = httpx.ConnectError("embedding-svc down")

    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    with patch.object(
        summarizer, "_chunks_from_db", return_value=[{"payload": {"page_start": 1, "text": "db text"}}]
    ) as mock_db:
        result = await summarizer.summarize(paper_id=1, file_hash="abc", title="Test")

    assert "objective" in result
    mock_db.assert_called_once_with(1, 5)
    mock_qdrant.search.assert_not_called()


@pytest.mark.anyio
async def test_summarize_embed_runtime_error_propagates():
    """Non-httpx exceptions from embed_single (e.g. RuntimeError) must propagate (#187).

    Before the fix, bare `except Exception:` silently swallowed these and
    cached a low-quality summary. Now they surface as real errors.
    """
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed_single.side_effect = RuntimeError("client misconfigured")

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    with pytest.raises(RuntimeError, match="client misconfigured"):
        await summarizer.summarize(paper_id=1, file_hash="abc", title="Test")


@pytest.mark.anyio
async def test_summarize_falls_back_to_db_on_embed_timeout(monkeypatch):
    """A hung embed_single (no HTTP error, no response) must not block forever (#237).

    embedding_timeout bounds the wait; on expiry the summarizer falls back to
    DB chunk order, same as the httpx.HTTPError path (#187).
    """
    from unittest.mock import patch

    monkeypatch.setattr(settings, "embedding_timeout", 0.05)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )
    mock_qdrant = MagicMock()
    mock_embedder = AsyncMock()

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)

    mock_embedder.embed_single.side_effect = _hang

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    with patch.object(
        summarizer, "_chunks_from_db", return_value=[{"payload": {"page_start": 1, "text": "db text"}}]
    ) as mock_db:
        result = await summarizer.summarize(paper_id=1, file_hash="abc", title="Test")

    assert "objective" in result
    mock_db.assert_called_once_with(1, 5)
    mock_qdrant.asearch.assert_not_called()


@pytest.mark.anyio
async def test_summarize_falls_back_to_db_on_qdrant_timeout(monkeypatch):
    """A hung Qdrant search must not block forever; falls back to DB chunk order (#237)."""
    from unittest.mock import patch

    monkeypatch.setattr(settings, "qdrant_timeout", 0.05)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )
    mock_qdrant = MagicMock()

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)

    mock_qdrant.asearch = AsyncMock(side_effect=_hang)
    mock_embedder = AsyncMock()
    mock_embedder.embed_single.return_value = [0.1] * 768

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    with patch.object(
        summarizer, "_chunks_from_db", return_value=[{"payload": {"page_start": 1, "text": "db text"}}]
    ) as mock_db:
        result = await summarizer.summarize(paper_id=1, file_hash="abc", title="Test")

    assert "objective" in result
    mock_db.assert_called_once_with(1, 5)


@pytest.mark.anyio
async def test_summarize_no_embedder_falls_back_to_db_on_qdrant_error():
    """Without an embedder, Qdrant failures must also fall back to DB chunk order (#267).

    Previously only the embedder-present path wrapped asearch in
    QDRANT_UNAVAILABLE_ERRORS/_chunks_from_db; the no-embedder (test
    convenience) path let the exception propagate unhandled.
    """
    from unittest.mock import patch

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )
    mock_qdrant = MagicMock()
    mock_qdrant.asearch = AsyncMock(side_effect=ConnectionError("qdrant down"))

    summarizer = RAGSummarizer(mock_llm, mock_qdrant)
    with patch.object(
        summarizer, "_chunks_from_db", return_value=[{"payload": {"page_start": 1, "text": "db text"}}]
    ) as mock_db:
        result = await summarizer.summarize(paper_id=1, file_hash="abc", title="Test")

    assert "objective" in result
    mock_db.assert_called_once_with(1, 5)


@pytest.mark.anyio
async def test_summarize_llm_timeout_propagates(monkeypatch):
    """A hung LLM backend must not block forever; TimeoutError propagates so the

    caller (server.py) records the job as failed (#237). Unlike embed/Qdrant,
    there is no fallback path for LLM generation.
    """
    monkeypatch.setattr(settings, "llm_generate_timeout", 0.05)

    mock_llm = AsyncMock()

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)

    mock_llm.generate.side_effect = _hang
    mock_qdrant = MagicMock()
    mock_qdrant.asearch = AsyncMock(
        return_value=[{"id": "1", "score": 0.9, "payload": {"paper_id": 1, "page_start": 1, "text": "sample text"}}]
    )

    summarizer = RAGSummarizer(mock_llm, mock_qdrant)
    with pytest.raises(TimeoutError):
        await summarizer.summarize(paper_id=1, file_hash="abc")


@pytest.mark.anyio
async def test_summarize_overall_timeout_bounds_stacked_individual_timeouts(monkeypatch):
    """The 3 inner wait_for calls (embedding/qdrant/llm) stack sequentially in the

    worst case, so summarize()'s real worst-case latency is their sum, not any one
    of them. summarize_total_timeout must cut the whole call short even when each
    individual timeout is generous enough to not fire on its own (#269).
    """
    monkeypatch.setattr(settings, "embedding_timeout", 10)
    monkeypatch.setattr(settings, "qdrant_timeout", 10)
    monkeypatch.setattr(settings, "llm_generate_timeout", 10)
    monkeypatch.setattr(settings, "summarize_total_timeout", 0.05)

    mock_embedder = AsyncMock()
    mock_embedder.embed_single = AsyncMock(return_value=[0.1] * 768)
    mock_qdrant = MagicMock()

    async def _slow_search(*args, **kwargs):
        await asyncio.sleep(10)

    mock_qdrant.asearch = AsyncMock(side_effect=_slow_search)
    mock_llm = AsyncMock()

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    with pytest.raises(TimeoutError):
        await summarizer.summarize(paper_id=1, file_hash="abc", title="paper")
