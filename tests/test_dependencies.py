"""Depends providers (#360): handlers are testable via app.dependency_overrides, no app.state / lifespan."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from academic_paper import dependencies as deps
from academic_paper.routers import jobs, summary


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(jobs.router)
    app.include_router(summary.router)
    return app


@pytest.fixture
def app():
    return _make_app()


def test_providers_read_request_app_state():
    req = MagicMock()
    req.app.state = MagicMock(embedder="e", vector_store="v", llm="l", summarizer="s", active_ingest_tasks={1})
    assert deps.get_embedder(req) == "e"
    assert deps.get_vector_store(req) == "v"
    assert deps.get_llm(req) == "l"
    assert deps.get_summarizer(req) == "s"
    assert deps.get_active_ingest_tasks(req) == {1}


def test_summarize_all_503_when_llm_overridden_to_none(app):
    app.dependency_overrides[deps.get_llm] = lambda: None
    app.dependency_overrides[deps.get_summarizer] = lambda: MagicMock()
    r = TestClient(app).post("/jobs/summarize-all")
    assert r.status_code == 503
    assert r.json()["detail"] == "LLM not configured"


def test_generate_summary_receives_overridden_llm_and_summarizer(app):
    llm, summarizer = MagicMock(), MagicMock()
    app.dependency_overrides[deps.get_llm] = lambda: llm
    app.dependency_overrides[deps.get_summarizer] = lambda: summarizer
    with patch.object(summary, "generate_summary", AsyncMock(return_value={"ok": 1})) as gen:
        r = TestClient(app).post("/papers/7/summary?force=true")
    assert r.status_code == 200
    assert r.json() == {"ok": 1}
    gen.assert_awaited_once_with(7, True, llm, summarizer)
