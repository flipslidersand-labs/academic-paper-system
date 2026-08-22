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
    mock_qdrant.search.return_value = chunks

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
    mock_qdrant.search.return_value = chunks

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
    mock_qdrant.search.return_value = chunks

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
async def test_summarize_uses_embedder_when_provided():
    """With embedder set, embed_single() is called instead of using dummy vector."""
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_embedder = AsyncMock()
    real_vector = [0.5] * 768
    mock_embedder.embed_single.return_value = real_vector

    chunks = [{"id": "1", "score": 0.9, "payload": {"paper_id": 1, "page_start": 1, "text": "chunk text"}}]
    mock_qdrant.search.return_value = chunks
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    await summarizer.summarize(paper_id=1, file_hash="abc", title="Deep Learning Paper")

    mock_embedder.embed_single.assert_called_once_with(
        "Deep Learning Paper", mode="search", collection="academic-papers"
    )
    # The real vector (not dummy) was passed to Qdrant
    call_kwargs = mock_qdrant.search.call_args[1]
    assert call_kwargs["query_vector"] == real_vector


@pytest.mark.anyio
async def test_summarize_embedder_fallback_on_no_title():
    """Without title, a generic fallback string is embedded."""
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed_single.return_value = [0.3] * 768

    chunks = [{"id": "1", "score": 0.9, "payload": {"paper_id": 1, "page_start": 1, "text": "text"}}]
    mock_qdrant.search.return_value = chunks
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )

    summarizer = RAGSummarizer(mock_llm, mock_qdrant, embedder=mock_embedder)
    await summarizer.summarize(paper_id=1, file_hash="abc")  # no title

    mock_embedder.embed_single.assert_called_once_with(
        "academic paper overview", mode="search", collection="academic-papers"
    )


@pytest.mark.anyio
async def test_summarize_without_embedder_uses_dummy_vector():
    """Without embedder, dummy vector [0.1]*768 is used (backward compat)."""
    mock_llm = AsyncMock()
    mock_qdrant = MagicMock()

    chunks = [{"id": "1", "score": 0.9, "payload": {"paper_id": 1, "page_start": 1, "text": "text"}}]
    mock_qdrant.search.return_value = chunks
    mock_llm.generate.return_value = json.dumps(
        {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}
    )

    summarizer = RAGSummarizer(mock_llm, mock_qdrant)  # no embedder
    await summarizer.summarize(paper_id=1, file_hash="abc")

    call_kwargs = mock_qdrant.search.call_args[1]
    assert call_kwargs["query_vector"] == [0.1] * 768
