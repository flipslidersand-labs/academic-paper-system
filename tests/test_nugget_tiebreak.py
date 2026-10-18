"""Regression tests for extract_nuggets tie-breaking (#723).

Equal scores must resolve by document order (earlier sentence first), not by
string comparison, and the selected sentences are joined in document order.
"""

from academic_paper.nugget import extract_nuggets


def test_no_match_query_returns_leading_sentences_in_doc_order():
    # Query matches nothing -> all BM25 scores equal -> the first top_k sentences.
    text = "Alpha one. Beta two. Gamma three. Delta four."
    assert extract_nuggets("zzz", text, top_k=2) == "Alpha one. Beta two."


def test_tied_scores_keep_document_order_with_mixed_matches():
    # Only the middle sentence matches; the two zero-score sentences tie.
    # The tie must go to the earlier sentence ("Alpha"), not the later one ("Gamma").
    text = "Alpha one. Kv cache two. Gamma three."
    assert extract_nuggets("kv", text, top_k=2) == "Alpha one. Kv cache two."


def test_selected_sentences_are_joined_in_document_order():
    # The highest-scoring sentence is last in the document; output must still be in doc order.
    # Alpha/Beta tie at 0, so the earlier one (Alpha) fills the second slot.
    text = "Beta two. Alpha one. Kv cache three."
    assert extract_nuggets("kv", text, top_k=2) == "Beta two. Kv cache three."


def test_top_k_larger_than_sentence_count_returns_all_in_doc_order():
    text = "Alpha one. Beta two. Gamma three."
    assert extract_nuggets("zzz", text, top_k=10) == "Alpha one. Beta two. Gamma three."


def test_top_k_zero_returns_empty_string():
    text = "Alpha one. Beta two."
    assert extract_nuggets("alpha", text, top_k=0) == ""


def test_zero_embed_weight_is_deterministic_and_ignores_vectors():
    text = "Alpha one. Beta two. Gamma three. Delta four."
    vecs = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    first = extract_nuggets("zzz", text, top_k=2, embed_weight=0.0, query_vec=[1.0, 0.0], sentence_vecs=vecs)
    second = extract_nuggets("zzz", text, top_k=2, embed_weight=0.0, query_vec=[1.0, 0.0], sentence_vecs=vecs)
    assert first == second == extract_nuggets("zzz", text, top_k=2) == "Alpha one. Beta two."
