"""Tests for author/category filter, metadata ingest, and GET /summaries."""

import tempfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.db import (
    get_connection,
    init_db,
    list_papers_filtered,
    list_summaries,
    save_chunks,
    save_paper,
    save_summary,
)
from academic_paper.server import app


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    init_db(db_path)
    yield db_path


@pytest.fixture
def client(temp_db):
    with patch.object(settings, "academic_db", temp_db):
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
        mock_qdrant = MagicMock()
        with patch("academic_paper.server.EmbedderClient", return_value=mock_embedder), \
             patch("academic_paper.server.QdrantStore", return_value=mock_qdrant):
            c = TestClient(app)
            c.app.state.embedder = mock_embedder
            c.app.state.vector_store = mock_qdrant
            yield c


# --- DB layer ---

def test_save_paper_with_metadata(temp_db):
    """Test save_paper stores authors, categories, published_date, source."""
    conn = get_connection(temp_db)
    paper_id = save_paper(
        conn, "paper.pdf", "hash_meta",
        title="Test Paper",
        authors=["Alice", "Bob"],
        categories=["cs.AI", "cs.LG"],
        published_date="2026-01-01",
        source="arxiv",
    )
    conn.close()

    conn = get_connection(temp_db)
    total, papers = list_papers_filtered(conn)
    conn.close()

    assert total == 1
    p = papers[0]
    assert p["id"] == paper_id
    assert p["title"] == "Test Paper"
    assert p["authors"] == ["Alice", "Bob"]
    assert p["categories"] == ["cs.AI", "cs.LG"]
    assert p["published_date"] == "2026-01-01"
    assert p["source"] == "arxiv"


def test_list_papers_filtered_by_author(temp_db):
    """Test list_papers_filtered filters by author substring."""
    conn = get_connection(temp_db)
    save_paper(conn, "a.pdf", "h1", authors=["Alice Smith", "Bob"])
    save_paper(conn, "b.pdf", "h2", authors=["Carol Jones"])
    conn.close()

    conn = get_connection(temp_db)
    total, papers = list_papers_filtered(conn, author="Alice")
    conn.close()

    assert total == 1
    assert papers[0]["file_name"] == "a.pdf"


def test_list_papers_filtered_by_category(temp_db):
    """Test list_papers_filtered filters by category substring."""
    conn = get_connection(temp_db)
    save_paper(conn, "a.pdf", "h1", categories=["cs.AI", "cs.LG"])
    save_paper(conn, "b.pdf", "h2", categories=["stat.ML"])
    conn.close()

    conn = get_connection(temp_db)
    total, papers = list_papers_filtered(conn, category="cs.AI")
    conn.close()

    assert total == 1
    assert papers[0]["file_name"] == "a.pdf"


def test_list_summaries_returns_paper_info(temp_db):
    """Test list_summaries joins paper metadata."""
    conn = get_connection(temp_db)
    pid = save_paper(conn, "s.pdf", "hs1", title="Summary Paper", authors=["Dave"])
    save_chunks(conn, pid, [{"text": "t", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": "q1", "token_count": 1}])
    save_summary(conn, pid, "gemini-2.0-flash", {
        "objective": "obj", "method": "meth", "results": "res",
        "limitations": "lim", "keywords": ["kw"],
    })
    conn.close()

    conn = get_connection(temp_db)
    total, summaries = list_summaries(conn)
    conn.close()

    assert total == 1
    s = summaries[0]
    assert s["paper_id"] == pid
    assert s["title"] == "Summary Paper"
    assert s["authors"] == ["Dave"]
    assert s["keywords"] == ["kw"]
    assert s["model"] == "gemini-2.0-flash"


# --- API layer ---

def _make_minimal_pdf() -> bytes:
    return (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000058 00000 n\n0000000115 00000 n\n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
    )


def test_ingest_with_metadata(client):
    """Test POST /papers/ingest stores authors/categories/source."""
    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [{"page": 1, "text": "Test content for metadata"}]
        resp = client.post(
            "/papers/ingest",
            files={"file": ("meta.pdf", BytesIO(_make_minimal_pdf()), "application/pdf")},
            data={
                "title": "My Paper",
                "authors": '["Alice", "Bob"]',
                "categories": '["cs.AI"]',
                "published_date": "2026-01-15",
                "source": "arxiv",
            },
        )
        assert resp.status_code == 200
        paper_id = resp.json()["paper_id"]

    detail = client.get(f"/papers/{paper_id}")
    assert detail.status_code == 200
    d = detail.json()
    assert d["title"] == "My Paper"
    assert d["authors"] == ["Alice", "Bob"]
    assert d["categories"] == ["cs.AI"]
    assert d["published_date"] == "2026-01-15"
    assert d["source"] == "arxiv"


def test_list_papers_author_filter(client, temp_db):
    """Test GET /papers?author= returns filtered results."""
    conn = get_connection(temp_db)
    save_paper(conn, "x.pdf", "hx1", authors=["Yuki Hana"])
    save_paper(conn, "y.pdf", "hx2", authors=["Bob Smith"])
    conn.close()

    resp = client.get("/papers?author=Yuki")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["papers"][0]["file_name"] == "x.pdf"


def test_list_papers_category_filter(client, temp_db):
    """Test GET /papers?category= returns filtered results."""
    conn = get_connection(temp_db)
    save_paper(conn, "a.pdf", "hc1", categories=["cs.AI"])
    save_paper(conn, "b.pdf", "hc2", categories=["stat.ML"])
    conn.close()

    resp = client.get("/papers?category=cs.AI")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["papers"][0]["file_name"] == "a.pdf"


def test_list_summaries_empty(client):
    """Test GET /summaries returns empty list when no summaries exist."""
    resp = client.get("/summaries")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["summaries"] == []


def test_list_summaries_with_data(client, temp_db):
    """Test GET /summaries returns summaries with paper metadata."""
    conn = get_connection(temp_db)
    pid = save_paper(conn, "z.pdf", "hz1", title="Z Paper", authors=["Eve"], categories=["cs.LG"])
    save_chunks(conn, pid, [{"text": "t", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": "qz1", "token_count": 1}])
    save_summary(conn, pid, "gemini-2.0-flash", {
        "objective": "o", "method": "m", "results": "r",
        "limitations": "l", "keywords": ["deep learning"],
    })
    conn.close()

    resp = client.get("/summaries")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    s = data["summaries"][0]
    assert s["paper_id"] == pid
    assert s["title"] == "Z Paper"
    assert s["authors"] == ["Eve"]
    assert s["categories"] == ["cs.LG"]
    assert s["keywords"] == ["deep learning"]


def test_list_summaries_pagination(client, temp_db):
    """Test GET /summaries?limit=1&offset=1 paginates correctly."""
    conn = get_connection(temp_db)
    for i in range(3):
        pid = save_paper(conn, f"p{i}.pdf", f"hp{i}")
        save_chunks(conn, pid, [{"text": f"text{i}", "page_start": 1, "page_end": 1, "chunk_index": 0, "qdrant_id": f"qp{i}", "token_count": 1}])
        save_summary(conn, pid, "model", {"objective": "o", "method": "m", "results": "r", "limitations": "l", "keywords": []})
    conn.close()

    resp = client.get("/summaries?limit=1&offset=0")
    assert resp.status_code == 200
    assert resp.json()["total"] == 3
    assert len(resp.json()["summaries"]) == 1

    resp2 = client.get("/summaries?limit=2&offset=1")
    assert resp2.status_code == 200
    assert len(resp2.json()["summaries"]) == 2
