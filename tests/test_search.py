"""Tests for GET /search endpoint."""

import tempfile
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from academic_paper.db import init_db, get_connection
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
        mock_embedder.embed_single = AsyncMock(return_value=[0.2] * 768)  # Query embedding
        
        mock_qdrant = MagicMock()
        
        # Patch and create client
        with patch("academic_paper.server.EmbedderClient", return_value=mock_embedder), \
             patch("academic_paper.server.QdrantStore", return_value=mock_qdrant):
            client = TestClient(app)
            # Manually set the mocked services since lifespan is patched
            client.app.state.embedder = mock_embedder
            client.app.state.vector_store = mock_qdrant
            yield client


def test_search_returns_results(client):
    """Test GET /search returns results list."""
    # Mock search results from Qdrant
    mock_search_results = [
        {
            "id": "qdrant-id-1",
            "score": 0.95,
            "payload": {
                "paper_id": 1,
                "chunk_index": 0,
                "text": "This is a test document about machine learning and AI systems.",
            }
        },
        {
            "id": "qdrant-id-2",
            "score": 0.85,
            "payload": {
                "paper_id": 1,
                "chunk_index": 1,
                "text": "Deep learning models require significant computational resources.",
            }
        }
    ]
    client.app.state.vector_store.search = MagicMock(return_value=mock_search_results)
    
    # Mock database query for page_start
    with patch("academic_paper.server.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Mock cursor.fetchone() to return page_start
        mock_cursor.fetchone.side_effect = [
            {"page_start": 1},
            {"page_start": 2},
        ]
        
        response = client.get("/search?q=machine learning")
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "hybrid"  # default mode
        assert data["query"] == "machine learning"
        assert len(data["results"]) == 2
        assert data["results"][0]["rank"] == 1
        assert data["results"][0]["score"] == 0.95
        assert data["results"][0]["paper_id"] == 1
        assert data["results"][0]["chunk_index"] == 0
        assert data["results"][0]["page_start"] == 1
        assert "machine learning and AI" in data["results"][0]["snippet"]


def test_search_vector_mode(client):
    """Test GET /search with vector mode calls embed_single."""
    # Mock search results from Qdrant
    mock_search_results = [
        {
            "id": "qdrant-id-1",
            "score": 0.95,
            "payload": {
                "paper_id": 1,
                "chunk_index": 0,
                "text": "This is a test document.",
            }
        }
    ]
    client.app.state.vector_store.search = MagicMock(return_value=mock_search_results)
    
    with patch("academic_paper.server.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchone.return_value = {"page_start": 1}
        
        response = client.get("/search?q=test&mode=vector")
        
        assert response.status_code == 200
        # Verify embed_single was called with search mode
        client.app.state.embedder.embed_single.assert_called_once_with("test", mode="search")
        data = response.json()
        assert data["mode"] == "vector"


def test_search_with_paper_id_filter(client):
    """Test GET /search with paper_id filter."""
    # Mock search results from Qdrant
    mock_search_results = [
        {
            "id": "qdrant-id-1",
            "score": 0.95,
            "payload": {
                "paper_id": 1,
                "chunk_index": 0,
                "text": "This is a test document.",
            }
        }
    ]
    client.app.state.vector_store.search = MagicMock(return_value=mock_search_results)
    
    with patch("academic_paper.server.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchone.return_value = {"page_start": 1}
        
        response = client.get("/search?q=test&paper_id=1&limit=5")
        
        assert response.status_code == 200
        # Verify search was called with paper_id_filter
        client.app.state.vector_store.search.assert_called_once_with(
            query_vector=[0.2] * 768,
            limit=5,
            paper_id_filter=1,
        )
