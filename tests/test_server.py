"""Tests for FastAPI server endpoints."""

import tempfile
from io import BytesIO
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from academic_paper.db import init_db, get_connection, get_chunks
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


def create_minimal_pdf() -> bytes:
    """Create minimal PDF bytes for testing."""
    # Minimal PDF structure
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>\n"
        b"endobj\n"
        b"4 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
        b"5 0 obj\n"
        b"<< >>\n"
        b"stream\n"
        b"BT /F1 12 Tf 100 700 Td (Test Document) Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f\n"
        b"0000000009 00000 n\n"
        b"0000000058 00000 n\n"
        b"0000000115 00000 n\n"
        b"0000000260 00000 n\n"
        b"0000000341 00000 n\n"
        b"trailer\n"
        b"<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        b"472\n"
        b"%%EOF\n"
    )
    return pdf


def test_list_papers_empty(client):
    """Test GET /papers returns empty list initially."""
    response = client.get("/papers")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["papers"] == []


def test_ingest_valid_pdf(client):
    """Test POST /papers/ingest with valid PDF."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [
            {"page": 1, "text": "Test Document content"},
        ]

        response = client.post(
            "/papers/ingest",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "paper_id" in data
        assert data["file_name"] == "test.pdf"
        assert data["status"] == "indexed"
        assert data["chunks"] > 0


def test_ingest_duplicate_pdf(client):
    """Test POST /papers/ingest with duplicate PDF returns 409."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [
            {"page": 1, "text": "Test Document content"},
        ]

        # Ingest first time
        response1 = client.post(
            "/papers/ingest",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response1.status_code == 200

        # Ingest same PDF again
        response2 = client.post(
            "/papers/ingest",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response2.status_code == 409
        assert "already ingested" in response2.json()["detail"].lower()


def test_list_papers_with_data(client):
    """Test GET /papers returns papers after ingestion."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [
            {"page": 1, "text": "Test Document content"},
        ]

        # Ingest a paper
        response_ingest = client.post(
            "/papers/ingest",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response_ingest.status_code == 200

        # List papers
        response_list = client.get("/papers")
        assert response_list.status_code == 200
        data = response_list.json()
        assert data["total"] == 1
        assert len(data["papers"]) == 1
        assert data["papers"][0]["file_name"] == "test.pdf"


def test_ingest_calls_embedder(client):
    """Test POST /papers/ingest calls EmbedderClient and QdrantStore."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [
            {"page": 1, "text": "Test Document content paragraph one"},
        ]

        response = client.post(
            "/papers/ingest",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )

        assert response.status_code == 200
        # Verify embedder was called
        client.app.state.embedder.embed.assert_called_once()
        # Verify Qdrant ensure_collection was called
        client.app.state.vector_store.ensure_collection.assert_called_once()
        # Verify Qdrant upsert was called
        client.app.state.vector_store.upsert.assert_called_once()


def test_ingest_stores_qdrant_id(client):
    """Test POST /papers/ingest stores qdrant_id in database."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [
            {"page": 1, "text": "Test Document content paragraph one"},
        ]

        response = client.post(
            "/papers/ingest",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        paper_id = data["paper_id"]

        # Verify qdrant_id was stored in database
        conn = get_connection(settings.academic_db)
        chunks = get_chunks(conn, paper_id)
        conn.close()

        assert len(chunks) > 0
        for chunk in chunks:
            assert "qdrant_id" in chunk
            assert chunk["qdrant_id"] is not None
            assert len(chunk["qdrant_id"]) > 0


def test_health_returns_ok(client):
    """Test GET /health returns ok when all services are healthy."""
    # Mock vector_store to have a working client
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    client.app.state.vector_store.client = mock_client
    
    # Mock httpx to return 200 status
    with patch("academic_paper.server.httpx.AsyncClient") as mock_httpx:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["qdrant"] == "ok"
        assert data["embedding_svc"] == "ok"


def test_health_returns_degraded_on_qdrant_error(client):
    """Test GET /health returns degraded when Qdrant is unavailable."""
    # Mock vector_store to raise exception
    mock_client = MagicMock()
    mock_client.get_collections.side_effect = Exception("Connection failed")
    client.app.state.vector_store.client = mock_client
    
    # Mock httpx to return 200 status
    with patch("academic_paper.server.httpx.AsyncClient") as mock_httpx:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["qdrant"] == "error"
        assert data["embedding_svc"] == "ok"


def test_stats_returns_counts(client):
    """Test GET /stats returns papers, chunks, and qdrant_points counts."""
    # First ingest a paper to get some data
    pdf_content = create_minimal_pdf()
    
    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [
            {"page": 1, "text": "Test Document content"},
        ]
        
        response_ingest = client.post(
            "/papers/ingest",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response_ingest.status_code == 200
    
    # Mock vector_store client for stats call
    mock_client = MagicMock()
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 1
    mock_client.get_collection.return_value = mock_collection_info
    client.app.state.vector_store.client = mock_client
    
    # Get stats
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "papers" in data
    assert "chunks" in data
    assert "qdrant_points" in data
    assert "db" in data
    assert data["papers"] >= 1
    assert data["chunks"] >= 1
    assert data["qdrant_points"] >= 1
