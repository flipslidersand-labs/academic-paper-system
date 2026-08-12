"""Tests for nugget extraction module."""

import pytest

from academic_paper.nugget import bm25_scores, extract_nuggets, split_sentences


def test_split_sentences_basic():
    text = "This is sentence one. This is sentence two. And three!"
    parts = split_sentences(text)
    assert len(parts) == 3


def test_split_sentences_empty():
    assert split_sentences("") == []


def test_split_sentences_single():
    assert split_sentences("Only one sentence") == ["Only one sentence"]


def test_bm25_scores_length():
    sentences = ["KV cache reuse", "Diffusion models", "RAG retrieval"]
    scores = bm25_scores("KV cache", sentences)
    assert len(scores) == 3


def test_bm25_relevant_scores_higher():
    sentences = ["KV cache reuse reduces inference cost", "Diffusion models generate images"]
    scores = bm25_scores("KV cache", sentences)
    assert scores[0] > scores[1]


def test_bm25_empty_sentences():
    assert bm25_scores("query", []) == []


def test_extract_nuggets_returns_top_k():
    text = (
        "KV cache reuse reduces inference cost in transformers. "
        "Diffusion models iteratively denoise images. "
        "RAG combines retrieval with generation. "
        "KV caching is essential for long-context LLMs."
    )
    nugget = extract_nuggets("KV cache transformer", text, top_k=2)
    assert "KV" in nugget or "kv" in nugget.lower()
    # Should not be the full text (nugget is shorter)
    assert len(nugget.split()) < len(text.split())


def test_extract_nuggets_fallback_empty():
    assert extract_nuggets("query", "") == ""


def test_extract_nuggets_single_sentence():
    text = "KV cache reuse is the core mechanism."
    result = extract_nuggets("KV cache", text, top_k=3)
    assert result == text


def test_extract_nuggets_top_k_respected():
    text = (
        "Sentence one about KV cache. "
        "Sentence two about RAG. "
        "Sentence three about diffusion. "
        "Sentence four about transformers."
    )
    nugget = extract_nuggets("KV cache RAG", text, top_k=2)
    # top_k=2 → at most 2 sentences returned
    sentences = [s.strip() for s in nugget.split(".") if s.strip()]
    assert len(sentences) <= 2
