"""Tests for hybrid search RRF merge functionality."""

import pytest
from academic_paper.hybrid import rrf_merge


def test_rrf_merge_combines_results():
    """Test that RRF merge combines FTS and vector results with rrf_score."""
    fts_results = [
        {"chunk_id": 1, "paper_id": 1, "chunk_index": 0, "text": "Machine learning basics", "rank": -5.0},
        {"chunk_id": 2, "paper_id": 1, "chunk_index": 1, "text": "Neural networks", "rank": -3.0},
    ]
    vector_results = [
        {"id": "vec-1", "score": 0.95, "payload": {"chunk_id": 2, "paper_id": 1, "chunk_index": 1, "text": "Neural networks"}},
        {"id": "vec-2", "score": 0.80, "payload": {"chunk_id": 3, "paper_id": 1, "chunk_index": 2, "text": "Deep learning models"}},
    ]

    result = rrf_merge(fts_results, vector_results, k=60)

    assert len(result) == 3
    assert result[0]["chunk_id"] == 2  # chunk 2 appears in both, should be ranked first
    assert "rrf_score" in result[0]
    assert all("rrf_score" in r for r in result)


def test_rrf_merge_both_sources_boost_score():
    """Test that results appearing in both FTS and vector have higher scores."""
    fts_results = [
        {"chunk_id": 1, "paper_id": 1, "chunk_index": 0, "text": "Test chunk", "rank": -5.0},
    ]
    vector_results = [
        {"id": "vec-1", "score": 0.95, "payload": {"chunk_id": 1, "paper_id": 1, "chunk_index": 0, "text": "Test chunk"}},
        {"id": "vec-2", "score": 0.80, "payload": {"chunk_id": 2, "paper_id": 1, "chunk_index": 1, "text": "Other chunk"}},
    ]

    result = rrf_merge(fts_results, vector_results, k=60)

    # Find scores for chunk 1 (in both) and chunk 2 (in vector only)
    chunk_1_score = next(r["rrf_score"] for r in result if r["chunk_id"] == 1)
    chunk_2_score = next(r["rrf_score"] for r in result if r["chunk_id"] == 2)

    assert chunk_1_score > chunk_2_score


def test_rrf_merge_empty_inputs():
    """Test that RRF merge handles empty FTS or vector results."""
    # Empty FTS, non-empty vector
    fts_results = []
    vector_results = [
        {"id": "vec-1", "score": 0.95, "payload": {"chunk_id": 1, "paper_id": 1, "chunk_index": 0, "text": "Test"}},
    ]

    result = rrf_merge(fts_results, vector_results)
    assert len(result) == 1
    assert result[0]["chunk_id"] == 1

    # Non-empty FTS, empty vector
    fts_results = [
        {"chunk_id": 1, "paper_id": 1, "chunk_index": 0, "text": "Test", "rank": -5.0},
    ]
    vector_results = []

    result = rrf_merge(fts_results, vector_results)
    assert len(result) == 1
    assert result[0]["chunk_id"] == 1

    # Both empty
    result = rrf_merge([], [])
    assert len(result) == 0


def test_rrf_merge_sorted_by_score():
    """Test that results are sorted by rrf_score in descending order."""
    fts_results = [
        {"chunk_id": 1, "paper_id": 1, "chunk_index": 0, "text": "Text 1", "rank": -5.0},
        {"chunk_id": 2, "paper_id": 1, "chunk_index": 1, "text": "Text 2", "rank": -10.0},
        {"chunk_id": 3, "paper_id": 1, "chunk_index": 2, "text": "Text 3", "rank": -15.0},
    ]
    vector_results = []

    result = rrf_merge(fts_results, vector_results, k=60)

    # Verify descending order
    for i in range(len(result) - 1):
        assert result[i]["rrf_score"] >= result[i + 1]["rrf_score"]

    # Verify specific order (chunk_id 1 > 2 > 3 from FTS ranks)
    assert result[0]["chunk_id"] == 1
    assert result[1]["chunk_id"] == 2
    assert result[2]["chunk_id"] == 3
