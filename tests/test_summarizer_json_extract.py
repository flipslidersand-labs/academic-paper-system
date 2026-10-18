"""Regression tests for LLM reply JSON extraction in the summarizer (#726).

The reply must yield the first complete JSON object even when prose around it
contains braces or several objects follow.
"""

import pytest

from academic_paper.summarizer import _parse_llm_json


def test_braces_in_trailing_prose_do_not_break_parsing():
    resp = '{"objective": "o"}\n\nNote: {see above}'
    assert _parse_llm_json(resp) == {"objective": "o"}


def test_code_fence_with_braces_inside_value_and_prose():
    resp = 'Here is the summary:\n```json\n{"objective": "a {b} c"}\n```\nDone {x}.'
    assert _parse_llm_json(resp) == {"objective": "a {b} c"}


def test_multiple_objects_returns_first_complete_object():
    resp = '{"objective": "first"} {"objective": "second"}'
    assert _parse_llm_json(resp) == {"objective": "first"}


def test_invalid_brace_span_before_valid_object_is_skipped():
    resp = 'pre {bad json} then {"objective": "ok"}'
    assert _parse_llm_json(resp) == {"objective": "ok"}


def test_reply_without_object_raises_value_error():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_llm_json("no json here")
