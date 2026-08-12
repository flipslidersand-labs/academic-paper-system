"""Nugget extraction: select query-relevant sentences from a chunk.

Based on the CoinRAG idea: instead of returning full chunks to the LLM,
return only the top-N sentences most relevant to the query (nuggets).
This reduces context length while maintaining or improving Recall@k.

No external dependencies — uses BM25 and sentence splitting only.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on '. ', '? ', '! ' boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_scores(query: str, sentences: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Return BM25 score for each sentence relative to query."""
    if not sentences:
        return []
    tokenized = [_tokenize(s) for s in sentences]
    avgdl = sum(len(t) for t in tokenized) / len(tokenized)
    q_terms = _tokenize(query)
    scores = []
    for doc in tokenized:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            idf = math.log(1 + (len(sentences) - f + 0.5) / (f + 0.5))
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores


def extract_nuggets(query: str, text: str, top_k: int = 3) -> str:
    """Return the top-k most query-relevant sentences from text joined as a string.

    Falls back to the full text if it cannot be split into sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text
    scores = bm25_scores(query, sentences)
    ranked = sorted(zip(scores, sentences), reverse=True)
    return " ".join(s for _, s in ranked[:top_k])
