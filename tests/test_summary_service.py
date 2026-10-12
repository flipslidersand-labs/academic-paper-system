"""Unit tests for academic_paper/services/summary_service.py (FastAPI not booted, #618)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from academic_paper.config import settings
from academic_paper.db import get_connection, save_paper
from academic_paper.services.summary_service import generate_summary, get_cached_summary

SUMMARY = {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": ["k"]}


@pytest.fixture(autouse=True)
def _db(temp_db):
    with patch.object(settings, "academic_db", temp_db):
        yield temp_db


@pytest.fixture
def paper_id(_db):
    conn = get_connection(_db)
    pid = save_paper(conn, "p.pdf", "h618")
    conn.close()
    return pid


@pytest.fixture
def llm():
    m = MagicMock()
    m.display_name = "test-model"
    return m


@pytest.fixture
def summarizer():
    s = MagicMock()
    s.summarize = AsyncMock(return_value=SUMMARY)
    return s


def _generate(pid, force, llm, summarizer):
    return asyncio.run(generate_summary(pid, force, llm, summarizer))


def test_get_cached_missing_paper_404():
    with pytest.raises(HTTPException) as ei:
        get_cached_summary(999)
    assert ei.value.status_code == 404


def test_get_cached_not_generated_404(paper_id):
    with pytest.raises(HTTPException) as ei:
        get_cached_summary(paper_id)
    assert ei.value.status_code == 404
    assert "not generated" in ei.value.detail


def test_generate_then_cache_hit(paper_id, llm, summarizer):
    first = _generate(paper_id, False, llm, summarizer)
    second = _generate(paper_id, False, llm, summarizer)
    assert first["cached"] is False and second["cached"] is True
    assert second["model"] == "test-model"
    assert summarizer.summarize.await_count == 1
    assert get_cached_summary(paper_id)["cached"] is True


def test_force_regenerates(paper_id, llm, summarizer):
    _generate(paper_id, False, llm, summarizer)
    again = _generate(paper_id, True, llm, summarizer)
    assert again["cached"] is False
    assert summarizer.summarize.await_count == 2


def test_missing_paper_404(llm, summarizer):
    with pytest.raises(HTTPException) as ei:
        _generate(999, False, llm, summarizer)
    assert ei.value.status_code == 404


def test_llm_none_503(paper_id, summarizer):
    with pytest.raises(HTTPException) as ei:
        _generate(paper_id, False, None, summarizer)
    assert (ei.value.status_code, ei.value.detail) == (503, "LLM not configured")


def test_summarizer_none_503(paper_id, llm):
    with pytest.raises(HTTPException) as ei:
        _generate(paper_id, False, llm, None)
    assert (ei.value.status_code, ei.value.detail) == (503, "Summarizer not initialized")


def test_timeout_maps_to_504(paper_id, llm, summarizer):
    summarizer.summarize = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(HTTPException) as ei:
        _generate(paper_id, False, llm, summarizer)
    assert ei.value.status_code == 504


# --- scoring / list (#619) ---------------------------------------------------


def _mk_papers(db, n):
    conn = get_connection(db)
    ids = [save_paper(conn, f"s{i}.pdf", f"hs{i}") for i in range(n)]
    conn.close()
    return ids


def test_score_all_zero_papers(_db):
    from academic_paper.services.summary_service import score_all

    r = score_all()
    assert (r["total"], r["scored"], r["failed"], r["errors"]) == (0, 0, 0, [])
    assert r["preferred_categories"] == settings.preferred_categories_list


def test_score_all_success_and_partial_failure(_db):
    from academic_paper.services import summary_service as svc

    _mk_papers(_db, 3)
    ok = svc.score_all()
    assert (ok["total"], ok["scored"], ok["failed"]) == (3, 3, 0)

    real = svc.compute_score
    calls = {"n": 0}

    def flaky(paper, preferred):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom http://10.0.0.5")
        return real(paper, preferred)

    with patch.object(svc, "compute_score", flaky):
        r = svc.score_all()
    assert (r["scored"], r["failed"]) == (2, 1)
    assert len(r["errors"]) == 1 and "10.0.0.5" not in r["errors"][0]


def test_score_one_ok_and_404(_db):
    from academic_paper.services.summary_service import score_one

    (pid,) = _mk_papers(_db, 1)
    r = score_one(pid)
    assert r["paper_id"] == pid and isinstance(r["score"], float)
    with pytest.raises(HTTPException) as ei:
        score_one(9999)
    assert ei.value.status_code == 404


def test_list_all_summaries_empty(_db):
    from academic_paper.services.summary_service import list_all_summaries

    assert list_all_summaries(20, 0) == {"total": 0, "summaries": []}
