"""Tests for GET /papers/{id}/summary endpoint."""

import tempfile
from io import BytesIO
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from academic_paper.db import init_db, get_connection, save_paper, save_chunks
from academic_paper.server import app
from academic_paper.config import settings


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
        with patch("academic_paper.server.EmbedderClient", return_value=mock_embedder), \
             patch("academic_paper.server.QdrantStore", return_value=mock_qdrant):
            client = TestClient(app)
            # Manually set the mocked services since lifespan is patched
            client.app.state.embedder = mock_embedder
            client.app.state.vector_store = mock_qdrant
            yield client


def test_summary_404_for_missing_paper(client):
    """Test GET /papers/{paper_id}/summary returns 404 for missing paper."""
    response = client.get("/papers/999/summary")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_summary_503_when_no_llm(client, temp_db):
    """Test GET /papers/{paper_id}/summary returns 503 when LLM not configured."""
    # Create a paper in the database
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "test.pdf", "hash123")
    
    # Create minimal chunks
    chunks = [{"text": "test content", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": "qid1", "token_count": 10}]
    save_chunks(conn, paper_id, chunks)
    conn.close()

    # Ensure LLM is None (no configuration)
    client.app.state.llm = None
    client.app.state.summarizer = None

    response = client.get(f"/papers/{paper_id}/summary")
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"].lower()


def test_summary_returns_structured_response(client, temp_db):
    """Test GET /papers/{paper_id}/summary returns structured response."""
    # Create a paper in the database
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "test.pdf", "hash123")
    
    # Create minimal chunks
    chunks = [{"text": "test content", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": "qid1", "token_count": 10}]
    save_chunks(conn, paper_id, chunks)
    conn.close()

    # Mock the LLM and summarizer
    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "GeminiClient"
    
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(return_value={
        "objective": "Test objective",
        "method": "Test method",
        "results": "Test results",
        "limitations": "Test limitations",
        "keywords": ["test", "keyword"],
    })
    
    client.app.state.llm = mock_llm
    client.app.state.summarizer = mock_summarizer

    response = client.get(f"/papers/{paper_id}/summary")
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
    """Test GET /papers/{paper_id}/summary returns cached=True on second call."""
    # Create a paper in the database
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "test.pdf", "hash123")
    
    # Create minimal chunks
    chunks = [{"text": "test content", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": "qid1", "token_count": 10}]
    save_chunks(conn, paper_id, chunks)
    conn.close()

    # Mock the LLM and summarizer
    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "GeminiClient"
    
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(return_value={
        "objective": "Test objective",
        "method": "Test method",
        "results": "Test results",
        "limitations": "Test limitations",
        "keywords": ["test", "keyword"],
    })
    
    client.app.state.llm = mock_llm
    client.app.state.summarizer = mock_summarizer

    # First call - should generate and cache
    response1 = client.get(f"/papers/{paper_id}/summary")
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["cached"] is False

    # Second call - should return cached
    response2 = client.get(f"/papers/{paper_id}/summary")
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cached"] is True
    assert data2["objective"] == data1["objective"]
    assert data2["method"] == data1["method"]

    # Summarizer should only be called once
    assert mock_summarizer.summarize.call_count == 1
