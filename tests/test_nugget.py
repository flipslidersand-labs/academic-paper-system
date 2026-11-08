"""Tests for nugget extraction module."""

import pytest

from academic_paper.nugget import (
    bm25_scores,
    cosine_similarity,
    extract_nuggets,
    split_sentences,
)


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


# --- hybrid (BM25 + embedding) scoring (#84) ---


def test_cosine_similarity_basic():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0


def test_cosine_similarity_empty_returns_zero():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], []) == 0.0


def test_cosine_similarity_length_mismatch_truncates_dot_keeps_full_norms():
    # 内積は短い方に切り詰め、ノルムは各ベクトル全体で計算する (#820 以前からの挙動)。
    assert cosine_similarity([3.0, 4.0], [1.0]) == pytest.approx(0.6)
    assert cosine_similarity([1.0], [3.0, 4.0]) == pytest.approx(0.6)
    # 共通部分が直交なら 0、片方がゼロベクトルなら 0。
    assert cosine_similarity([1.0, 0.0, 5.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_sign_and_scale():
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)
    assert cosine_similarity([3.0, 4.0], [3.0, 4.0]) == 1.0
    assert cosine_similarity([3.0, 4.0], [300.0, 400.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(
        32.0 / (14.0**0.5 * 77.0**0.5), rel=1e-12
    )


def test_extract_nuggets_embedding_shifts_ranking():
    # BM25 favors the lexical-overlap sentence; embedding vectors are crafted so
    # the *second* sentence is closest to the query vector. With embed_weight=1.0
    # the embedding-favored sentence must win.
    text = "Alpha beta gamma. Delta epsilon zeta."
    query = "alpha"  # lexically overlaps only sentence one
    query_vec = [0.0, 1.0]
    sentence_vecs = [[1.0, 0.0], [0.0, 1.0]]  # sentence two aligns with query
    nugget = extract_nuggets(
        query,
        text,
        top_k=1,
        embed_weight=1.0,
        query_vec=query_vec,
        sentence_vecs=sentence_vecs,
    )
    assert nugget == "Delta epsilon zeta."


def test_extract_nuggets_weight_zero_is_bm25():
    # embed_weight=0.0 must ignore vectors entirely → identical to BM25-only.
    text = "Alpha beta gamma. Delta epsilon zeta."
    query = "alpha"
    query_vec = [0.0, 1.0]
    sentence_vecs = [[1.0, 0.0], [0.0, 1.0]]
    hybrid = extract_nuggets(query, text, top_k=1, embed_weight=0.0, query_vec=query_vec, sentence_vecs=sentence_vecs)
    bm25_only = extract_nuggets(query, text, top_k=1)
    assert hybrid == bm25_only == "Alpha beta gamma."


def test_extract_nuggets_mismatched_vecs_falls_back_to_bm25():
    # Wrong number of sentence vectors → safe BM25 fallback, no crash.
    text = "Alpha beta gamma. Delta epsilon zeta."
    result = extract_nuggets(
        "alpha",
        text,
        top_k=1,
        embed_weight=0.7,
        query_vec=[0.0, 1.0],
        sentence_vecs=[[1.0, 0.0]],  # only 1 vec for 2 sentences
    )
    assert result == "Alpha beta gamma."


def test_bm25_idf_uses_document_frequency():
    # "common" appears in every sentence (df=N), "rare" in one sentence (df=1).
    # With a document-frequency idf the rare term must outweigh the common one
    # for sentences of equal length and tf=1 (#383).
    sentences = ["common rare x", "common y z", "common w v"]
    rare = bm25_scores("rare", sentences)
    common = bm25_scores("common", sentences)
    assert rare[0] > common[0]
    # Common term: identical tf/length -> identical score across sentences.
    assert common[0] == common[1] == common[2]
    # Rare term only scores the sentence that contains it.
    assert rare[1] == rare[2] == 0.0


# --- BM25 tokenization strips punctuation (#824) ---


def test_tokenize_strips_punctuation():
    from academic_paper.nugget import _tokenize

    assert _tokenize("Results. Accuracy, (RAG) rag?") == ["results", "accuracy", "rag", "rag"]


def test_tokenize_keeps_non_ascii():
    from academic_paper.nugget import _tokenize

    assert _tokenize("日本語の検索、テスト。 café") == ["日本語の検索", "テスト", "café"]


def test_bm25_punctuated_word_matches_query_term():
    # 'results.' (sentence-final) must match query 'results'.
    scores = bm25_scores("results", ["We report results.", "Nothing here"])
    assert scores[0] > 0.0
    assert scores[1] == 0.0


def test_extract_nuggets_selects_sentence_by_punctuated_word():
    # Only the sentence-final / parenthesised / comma-suffixed words match the query.
    text = "Intro text here. We improve accuracy, (RAG) greatly. Other filler words follow. The final results."
    assert extract_nuggets("results", text, top_k=1, embed_weight=0.0) == "The final results."
    assert extract_nuggets("rag", text, top_k=1, embed_weight=0.0) == "We improve accuracy, (RAG) greatly."
