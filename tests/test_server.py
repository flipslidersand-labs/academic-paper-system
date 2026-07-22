"""Tests for FastAPI server endpoints."""

import tempfile
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from academic_paper.db import init_db
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
    """Create a test client with patched settings."""
    with patch.object(settings, "academic_db", temp_db):
        yield TestClient(app)


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
