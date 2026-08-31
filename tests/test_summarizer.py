"""Tests for academic_paper.summarizer module."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

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
async def test_summarize_falls_back_to_db_on_embed_failure():
    """When embed_single raises, summarizer takes DB chunks instead of a zero-vector search (#139)."""
    from unittest.mock import patch

    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed_single.side_effect = RuntimeError("embedding-svc down")

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
