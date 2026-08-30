"""Tests for GET /papers/{id}/summary endpoint."""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.db import get_connection, init_db, save_chunks, save_paper
from academic_paper.server import app


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    # Initialize the database schema
    init_db(db_path)
    yield db_path


@pytest.fixture
def client(temp_db):
    """Create a test client with patched settings and mocked services."""
    with patch.object(settings, "academic_db", temp_db):
        # Create mock instances for EmbedderClient and QdrantStore
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])  # 768-dim vector

        mock_qdrant = MagicMock()

        # Patch and create client
        with (
            patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
            patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
        ):
            client = TestClient(app)
            # Manually set the mocked services since lifespan is patched
            client.app.state.embedder = mock_embedder
            client.app.state.vector_store = mock_qdrant
            yield client


def _make_chunks():
    return [
        {
            "text": "test content",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "qdrant_id": "qid1",
            "token_count": 10,
        }
    ]


def test_summary_404_for_missing_paper(client):
    """Test GET /papers/{paper_id}/summary returns 404 for missing paper."""
    response = client.get("/papers/999/summary")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_summary_503_when_no_llm(client, temp_db):
    """Test POST /papers/{paper_id}/summary returns 503 when LLM not configured."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "test.pdf", "hash123")
    save_chunks(conn, paper_id, _make_chunks())
    conn.close()

    client.app.state.llm = None
    client.app.state.summarizer = None

    response = client.post(f"/papers/{paper_id}/summary")
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"].lower()


def test_summary_returns_structured_response(client, temp_db):
    """Test POST /papers/{paper_id}/summary returns structured response."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "test.pdf", "hash123")
    save_chunks(conn, paper_id, _make_chunks())
    conn.close()

    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "GeminiClient"
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(
        return_value={
            "objective": "Test objective",
            "method": "Test method",
            "results": "Test results",
            "limitations": "Test limitations",
            "keywords": ["test", "keyword"],
        }
    )
    client.app.state.llm = mock_llm
    client.app.state.summarizer = mock_summarizer

    response = client.post(f"/papers/{paper_id}/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["paper_id"] == paper_id
    assert data["model"] == "gemini-2.0-flash"
    assert data["objective"] == "Test objective"
    assert data["method"] == "Test method"
    assert data["results"] == "Test results"
    assert data["limitations"] == "Test limitations"
    assert data["keywords"] == ["test", "keyword"]
    assert data["cached"] is False


def test_summary_cached_on_second_call(client, temp_db):
    """Test POST then GET: generation caches, GET serves cached=True (#140)."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "test.pdf", "hash123")
    save_chunks(conn, paper_id, _make_chunks())
    conn.close()

    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "GeminiClient"
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(
        return_value={
            "objective": "Test objective",
            "method": "Test method",
            "results": "Test results",
            "limitations": "Test limitations",
            "keywords": ["test", "keyword"],
        }
    )
    client.app.state.llm = mock_llm
    client.app.state.summarizer = mock_summarizer

    response1 = client.post(f"/papers/{paper_id}/summary")
    assert response1.status_code == 200
    assert response1.json()["cached"] is False

    # GET now serves the cached summary without generating (#140)
    response2 = client.get(f"/papers/{paper_id}/summary")
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cached"] is True
    assert data2["objective"] == response1.json()["objective"]
    assert mock_summarizer.summarize.call_count == 1


# --- New tests ---


def test_summary_force_regenerate(client, temp_db):
    """Test POST /papers/{paper_id}/summary?force=true bypasses cache."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "force.pdf", "hash_force")
    save_chunks(
        conn,
        paper_id,
        [
            {
                "text": "force test content",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
                "qdrant_id": "qid_force",
                "token_count": 10,
            }
        ],
    )
    conn.close()

    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "GeminiClient"
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(
        return_value={
            "objective": "obj",
            "method": "meth",
            "results": "res",
            "limitations": "lim",
            "keywords": ["k"],
        }
    )
    client.app.state.llm = mock_llm
    client.app.state.summarizer = mock_summarizer

    # First call — generates and caches
    r1 = client.post(f"/papers/{paper_id}/summary")
    assert r1.status_code == 200
    assert r1.json()["cached"] is False

    # Second call with force=true — must regenerate, not serve cache
    r2 = client.post(f"/papers/{paper_id}/summary?force=true")
    assert r2.status_code == 200
    assert r2.json()["cached"] is False
    assert mock_summarizer.summarize.call_count == 2


def test_summary_ollama_model_naming(client, temp_db):
    """Test POST /papers/{paper_id}/summary returns 'ollama/<model>' for OllamaClient."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "ollama.pdf", "hash_ollama")
    save_chunks(
        conn,
        paper_id,
        [
            {
                "text": "ollama test",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
                "qdrant_id": "qid_ollama",
                "token_count": 10,
            }
        ],
    )
    conn.close()

    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "OllamaClient"
    mock_llm.model = "qwen2.5:7b"
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(
        return_value={
            "objective": "obj",
            "method": "meth",
            "results": "res",
            "limitations": "lim",
            "keywords": [],
        }
    )
    client.app.state.llm = mock_llm
    client.app.state.summarizer = mock_summarizer

    response = client.post(f"/papers/{paper_id}/summary")
    assert response.status_code == 200
    assert response.json()["model"] == "ollama/qwen2.5:7b"


def test_summary_error_returns_400(client, temp_db):
    """Test POST /papers/{paper_id}/summary returns 400 when summarization raises."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "err.pdf", "hash_err")
    save_chunks(
        conn,
        paper_id,
        [
            {
                "text": "error test",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
                "qdrant_id": "qid_err",
                "token_count": 10,
            }
        ],
    )
    conn.close()

    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "GeminiClient"
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(side_effect=Exception("LLM timeout"))
    client.app.state.llm = mock_llm
    client.app.state.summarizer = mock_summarizer

    response = client.post(f"/papers/{paper_id}/summary")
    assert response.status_code == 400
    assert "Summarization error" in response.json()["detail"]


def test_summary_503_when_summarizer_none(client, temp_db):
    """Test POST /papers/{paper_id}/summary returns 503 when LLM is set but summarizer is None."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "test.pdf", "hash_no_sum")
    save_chunks(conn, paper_id, _make_chunks())
    conn.close()

    client.app.state.llm = MagicMock()
    client.app.state.summarizer = None

    response = client.post(f"/papers/{paper_id}/summary")
    assert response.status_code == 503
    assert "Summarizer not initialized" in response.json()["detail"]


def test_get_summary_cache_miss_404_no_generation(client, temp_db):
    """Regression (#140): GET must not generate — cache miss is 404, summarizer untouched."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "nogen.pdf", "hash_nogen")
    save_chunks(conn, paper_id, _make_chunks())
    conn.close()

    mock_summarizer = AsyncMock()
    client.app.state.llm = MagicMock()
    client.app.state.summarizer = mock_summarizer

    response = client.get(f"/papers/{paper_id}/summary")
    assert response.status_code == 404
    assert "not generated yet" in response.json()["detail"].lower()
    mock_summarizer.summarize.assert_not_called()
