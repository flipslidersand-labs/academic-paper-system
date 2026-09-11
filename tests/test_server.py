"""Tests for FastAPI server endpoints."""

import tempfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from academic_paper.config import settings
from academic_paper.db import get_chunks, get_connection, save_chunks, save_paper
from academic_paper.server import _cleanup_orphaned_ingests, app


@pytest.fixture
def client(temp_db):
    """Create a test client with patched settings and mocked services."""
    with patch.object(settings, "academic_db", temp_db):
        # Create mock instances for EmbedderClient and QdrantStore
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])  # 768-dim vector

        mock_qdrant = MagicMock()
        # Async methods on QdrantStore must be AsyncMock so they can be awaited (#149).
        mock_qdrant.aupsert = AsyncMock(return_value=None)
        mock_qdrant.asearch = AsyncMock(return_value=[])
        mock_qdrant.adelete_by_paper_id = AsyncMock(return_value=None)
        mock_qdrant.aensure_collection = AsyncMock(return_value=None)

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
            "/papers/ingest?wait=true",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "paper_id" in data
        assert data["file_name"] == "test.pdf"
        assert data["status"] == "indexed"
        assert data["chunks"] > 0


def test_ingest_cleans_up_tmpfile(client):
    """Temp file must be deleted after successful ingest."""
    pdf_content = create_minimal_pdf()
    captured = {}

    original_mkstemp_ctx = tempfile.NamedTemporaryFile

    def fake_ntf(**kwargs):
        ctx = original_mkstemp_ctx(**kwargs)
        captured["path"] = ctx.name
        return ctx

    with (
        patch("academic_paper.server.tempfile.NamedTemporaryFile", side_effect=fake_ntf),
        patch("academic_paper.server.extract_text") as mock_extract,
    ):
        mock_extract.return_value = [{"page": 1, "text": "Test Document content"}]
        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response.status_code == 200

    import os

    assert "path" in captured
    assert not os.path.exists(captured["path"]), "temp file should be deleted after ingest"


def test_ingest_cleans_up_tmpfile_on_error(client):
    """Temp file must be deleted even when extraction fails."""
    pdf_content = create_minimal_pdf()
    captured = {}

    original_mkstemp_ctx = tempfile.NamedTemporaryFile

    def fake_ntf(**kwargs):
        ctx = original_mkstemp_ctx(**kwargs)
        captured["path"] = ctx.name
        return ctx

    with (
        patch("academic_paper.server.tempfile.NamedTemporaryFile", side_effect=fake_ntf),
        patch("academic_paper.server.extract_text", side_effect=RuntimeError("boom")),
    ):
        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        # Unexpected errors are now 500, not 400 (#148)
        assert response.status_code in (400, 500)

    import os

    assert "path" in captured
    assert not os.path.exists(captured["path"]), "temp file should be deleted even on error"


def test_ingest_extract_timeout_fails_job_and_cleans_tmpfile(client):
    """extract_text() taking longer than pdf_extract_timeout must fail the job, not hang it (#238)."""
    import os
    import time

    pdf_content = create_minimal_pdf()
    captured = {}

    original_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(**kwargs):
        ctx = original_ntf(**kwargs)
        captured["path"] = ctx.name
        return ctx

    def slow_extract(_path):
        time.sleep(0.3)
        return [{"page": 1, "text": "should never get here"}]

    with (
        patch.object(settings, "pdf_extract_timeout", 0.05),
        patch("academic_paper.server.tempfile.NamedTemporaryFile", side_effect=fake_ntf),
        patch("academic_paper.server.extract_text", side_effect=slow_extract),
    ):
        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response.status_code in (400, 500)

    assert "path" in captured
    assert not os.path.exists(captured["path"]), "temp file should be deleted after a timed-out extraction"


def test_ingest_duplicate_pdf(client):
    """Test POST /papers/ingest with duplicate PDF returns 409."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [
            {"page": 1, "text": "Test Document content"},
        ]

        # Ingest first time
        response1 = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response1.status_code == 200

        # Ingest same PDF again
        response2 = client.post(
            "/papers/ingest?wait=true",
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
            "/papers/ingest?wait=true",
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
            "/papers/ingest?wait=true",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )

        assert response.status_code == 200
        # Verify embedder was called
        client.app.state.embedder.embed.assert_called_once()
        # Verify async Qdrant methods were called (#149)
        client.app.state.vector_store.aensure_collection.assert_called_once()
        client.app.state.vector_store.aupsert.assert_called_once()


def test_ingest_stores_qdrant_id(client):
    """Test POST /papers/ingest stores qdrant_id in database."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [
            {"page": 1, "text": "Test Document content paragraph one"},
        ]

        response = client.post(
            "/papers/ingest?wait=true",
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
    """Test GET /health returns 503 degraded when Qdrant is unavailable (#195)."""
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
        assert response.status_code == 503
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
            "/papers/ingest?wait=true",
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
    # db path is not exposed to callers (#148)
    assert "db" not in data
    assert data["papers"] >= 1
    assert data["chunks"] >= 1
    assert data["qdrant_points"] >= 1


# --- New tests ---


def test_get_paper_by_id_success(client):
    """Test GET /papers/{paper_id} returns paper details."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [{"page": 1, "text": "Test Document content"}]
        resp = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("detail.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert resp.status_code == 200
        paper_id = resp.json()["paper_id"]

    response = client.get(f"/papers/{paper_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == paper_id
    assert data["file_name"] == "detail.pdf"
    assert data["status"] == "indexed"


def test_get_paper_by_id_not_found(client):
    """Test GET /papers/{paper_id} returns 404 for missing paper."""
    response = client.get("/papers/99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def _mock_embedding_response(status_code: int) -> MagicMock:
    """Build a mock httpx response whose raise_for_status honors status_code."""
    import httpx as _httpx

    mock_response = MagicMock()
    mock_response.status_code = status_code
    if status_code >= 400:
        mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=mock_response
        )
    return mock_response


def test_health_embedding_svc_degraded(client):
    """Test GET /health returns 503 degraded when embedding-svc returns 5xx (#195)."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    client.app.state.vector_store.client = mock_client

    with patch("academic_paper.server.httpx.AsyncClient") as mock_httpx:
        mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(return_value=_mock_embedding_response(503))

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["qdrant"] == "ok"
        assert data["embedding_svc"] == "error"


def test_health_embedding_svc_auth_failure_degraded(client):
    """Regression (#142, #195): a 401 from embedding-svc must report degraded with HTTP 503."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    client.app.state.vector_store.client = mock_client

    with patch("academic_paper.server.httpx.AsyncClient") as mock_httpx:
        mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(return_value=_mock_embedding_response(401))

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["embedding_svc"] == "error"


def test_stats_qdrant_error(client):
    """Test GET /stats returns qdrant_points=-1 when Qdrant collection is unavailable."""
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Connection failed")
    client.app.state.vector_store.client = mock_client

    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["qdrant_points"] == -1


def test_ingest_empty_pages(client):
    """Test POST /papers/ingest returns 400 when no text extracted from PDF."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = []

        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("empty.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response.status_code == 400


def test_ingest_no_chunks(client):
    """Test POST /papers/ingest returns 400 when chunker produces no chunks."""
    pdf_content = create_minimal_pdf()

    with (
        patch("academic_paper.server.extract_text") as mock_extract,
        patch("academic_paper.server.chunk_pages") as mock_chunk,
    ):
        mock_extract.return_value = [{"page": 1, "text": "Some text"}]
        mock_chunk.return_value = []

        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("nochunk.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response.status_code == 400


def test_ingest_embedding_connect_error_returns_503(client):
    """Embedding service connect failure → 503 (upstream unavailable), not 400 (#148)."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [{"page": 1, "text": "Test Document content"}]
        client.app.state.embedder.embed = AsyncMock(side_effect=httpx.ConnectError("refused"))

        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("fail.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response.status_code == 503


def test_ingest_extract_value_error_returns_400(client):
    """ValueError from extraction (no text / no chunks) → 400, not 500 (#148)."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [{"page": 1, "text": "Test Document content"}]
        client.app.state.embedder.embed = AsyncMock(side_effect=ValueError("No chunks generated"))

        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("fail.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response.status_code == 400


def test_ingest_embedding_count_mismatch_returns_400(client):
    """embedder returns fewer vectors than chunks (#227) → 400, not silently truncated by zip()."""
    pdf_content = create_minimal_pdf()

    with (
        patch("academic_paper.server.extract_text") as mock_extract,
        patch("academic_paper.server.chunk_pages") as mock_chunk,
    ):
        mock_extract.return_value = [{"page": 1, "text": "Some text"}]
        mock_chunk.return_value = [
            {"text": "chunk one", "page_start": 1},
            {"text": "chunk two", "page_start": 1},
        ]
        # Simulate a partial/degraded embedding-svc batch response (#227).
        client.app.state.embedder.embed = AsyncMock(return_value=[[0.1] * 768])

        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("mismatch.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert response.status_code == 400


def test_ingest_async_returns_202_and_completes_job(client):
    """Default ingest is async: returns 202 + job_id, job completes to done."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [{"page": 1, "text": "Async ingest content"}]

        response = client.post(
            "/papers/ingest",
            files={"file": ("async.pdf", BytesIO(pdf_content), "application/pdf")},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert "job_id" in body
    paper_id = body["paper_id"]

    # TestClient runs BackgroundTasks synchronously, so the job is already done.
    job = client.get(f"/jobs/{body['job_id']}").json()
    assert job["status"] == "done"
    assert job["result"]["paper_id"] == paper_id
    assert job["result"]["chunks"] > 0

    paper = client.get(f"/papers/{paper_id}").json()
    assert paper["status"] == "indexed"


def test_ingest_async_duplicate_returns_409(client):
    """Async ingest of an already-ingested file returns 409 synchronously."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [{"page": 1, "text": "Dup content"}]
        first = client.post(
            "/papers/ingest",
            files={"file": ("dup.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert first.status_code == 202

        second = client.post(
            "/papers/ingest",
            files={"file": ("dup.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert second.status_code == 409
        assert "already ingested" in second.json()["detail"].lower()


def test_ingest_async_job_failed_on_no_text(client):
    """When extraction yields no text, the async job ends 'failed' and paper is 'failed'."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = []
        response = client.post(
            "/papers/ingest",
            files={"file": ("notext.pdf", BytesIO(pdf_content), "application/pdf")},
        )

    assert response.status_code == 202
    body = response.json()
    job = client.get(f"/jobs/{body['job_id']}").json()
    assert job["status"] == "failed"
    assert job["errors"]

    paper = client.get(f"/papers/{body['paper_id']}").json()
    assert paper["status"] == "failed"


def test_list_papers_pagination(client, temp_db):
    """Test GET /papers pagination with limit and offset."""
    conn = get_connection(temp_db)
    save_paper(conn, "paper1.pdf", "hash_pg_1")
    save_paper(conn, "paper2.pdf", "hash_pg_2")
    save_paper(conn, "paper3.pdf", "hash_pg_3")
    conn.close()

    response = client.get("/papers?limit=2&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["papers"]) == 2

    response2 = client.get("/papers?limit=2&offset=2")
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["total"] == 3
    assert len(data2["papers"]) == 1
    assert data2["papers"][0]["id"] != data["papers"][0]["id"]


def test_parse_list_field_comma_separated(client):
    """_parse_list_field handles comma-separated strings and JSON decode errors."""
    from academic_paper.server import _parse_list_field

    assert _parse_list_field(None) is None
    assert _parse_list_field('["a", "b"]') == ["a", "b"]
    assert _parse_list_field("a,b,c") == ["a", "b", "c"]
    assert _parse_list_field("not json {[") == ["not json {["]


def test_parse_list_field_rejects_non_array_json(client):
    """Valid JSON that isn't an array of strings must be rejected, not silently
    CSV-split into corrupted fragments (#231)."""
    from academic_paper.server import _parse_list_field

    for value in ('{"name": "Alice"}', "42", "true", '"just a string"', "[1, 2, 3]"):
        with pytest.raises(HTTPException) as exc_info:
            _parse_list_field(value, "authors")
        assert exc_info.value.status_code == 422


def test_validate_published_date_rejects_out_of_range():
    """_validate_published_date rejects future dates and unreasonably old ones,
    even when they are formally valid ISO dates (#271)."""
    from academic_paper.server import _validate_published_date

    _validate_published_date(None)
    _validate_published_date("2024-01-15")

    with pytest.raises(HTTPException) as exc_info:
        _validate_published_date("9999-12-31")
    assert exc_info.value.status_code == 422

    with pytest.raises(HTTPException) as exc_info:
        _validate_published_date("0001-01-01")
    assert exc_info.value.status_code == 422

    from datetime import date, timedelta

    future = (date.today() + timedelta(days=1)).isoformat()
    with pytest.raises(HTTPException) as exc_info:
        _validate_published_date(future)
    assert exc_info.value.status_code == 422


def test_sanitize_text_strips_control_chars():
    """_sanitize_text removes control chars (incl. newlines) from external metadata (#233)."""
    from academic_paper.server import _sanitize_text

    assert _sanitize_text(None) is None
    assert _sanitize_text("Normal Title") == "Normal Title"
    assert _sanitize_text("Evil\ntitle\r\nOK  1.0.0 [x] fake-log-line") == "Evil title  OK  1.0.0 [x] fake-log-line"
    assert _sanitize_text("<script>alert(1)</script>\x00\x1f") == "<script>alert(1)</script>"
    assert _sanitize_text("\n\r\t") is None


def test_parse_list_field_strips_control_chars():
    """_parse_list_field sanitizes each item of authors/categories (#233)."""
    from academic_paper.server import _parse_list_field

    assert _parse_list_field('["A\\nlice", "Bob"]') == ["A lice", "Bob"]
    assert _parse_list_field("A\nlice,Bob\r\n") == ["A lice", "Bob"]


def test_ingest_sanitizes_title_and_source(client):
    """title/source with control chars are sanitized before being stored (#233)."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [{"page": 1, "text": "Test Document content"}]

        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("sanitize.pdf", BytesIO(pdf_content), "application/pdf")},
            data={
                "title": "Evil\ntitle  OK  1.0.0 [x] fake-log-line",
                "authors": "A\nlice,Bob",
                "source": "arxiv\r\n",
            },
        )

        assert response.status_code == 200
        paper_id = response.json()["paper_id"]

    detail = client.get(f"/papers/{paper_id}").json()
    assert "\n" not in detail["title"]
    assert "\r" not in detail["title"]
    assert detail["title"] == "Evil title  OK  1.0.0 [x] fake-log-line"
    assert detail["authors"] == ["A lice", "Bob"]
    assert detail["source"] == "arxiv"


def test_score_all_endpoint(client, temp_db):
    """POST /papers/score-all computes scores for all papers."""
    conn = get_connection(temp_db)
    save_paper(conn, "paper1.pdf", "hash_score1")
    save_paper(conn, "paper2.pdf", "hash_score2")
    conn.close()

    response = client.post("/papers/score-all")
    assert response.status_code == 200
    data = response.json()
    assert "scored" in data
    assert data["scored"] == 2


def test_score_paper_success(client, temp_db):
    """POST /papers/{paper_id}/score returns score for a valid paper."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "paper.pdf", "hash_sc1")
    conn.close()

    response = client.post(f"/papers/{paper_id}/score")
    assert response.status_code == 200
    data = response.json()
    assert data["paper_id"] == paper_id
    assert "score" in data


def test_score_paper_not_found(client):
    """POST /papers/{paper_id}/score returns 404 for missing paper."""
    response = client.post("/papers/9999/score")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_ingest_unexpected_exception(client):
    """POST /papers/ingest: unexpected error returns 500 with opaque message (#148)."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.hash_file", side_effect=RuntimeError("disk error")):
        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )
    # RuntimeError is classified as internal error (500); detail must NOT contain the raw error.
    assert response.status_code == 500
    assert "disk error" not in response.json()["detail"]


def test_ingest_file_too_large_returns_413(client):
    """POST /papers/ingest returns 413 when file exceeds max_upload_mb limit."""
    with patch.object(settings, "max_upload_mb", 0):
        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("big.pdf", BytesIO(b"X"), "application/pdf")},
        )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_ingest_file_at_limit_is_accepted(client):
    """POST /papers/ingest accepts a file exactly at the size limit."""
    pdf_content = create_minimal_pdf()
    limit_mb = len(pdf_content) // (1024 * 1024) + 1

    with (
        patch.object(settings, "max_upload_mb", limit_mb),
        patch("academic_paper.server.extract_text") as mock_extract,
    ):
        mock_extract.return_value = [{"page": 1, "text": "Test content"}]
        response = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("ok.pdf", BytesIO(pdf_content), "application/pdf")},
        )
    assert response.status_code == 200


def test_search_keyword_returns_real_chunk_index(client, temp_db):
    """Regression (#131): keyword mode must return the real chunk_index, not always 0."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "kw.pdf", "hash-kw-idx", title="KW")
    save_chunks(
        conn,
        paper_id,
        [
            {
                "text": "intro alpha term",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
                "qdrant_id": "kw-0",
                "token_count": 3,
            },
            {
                "text": "later alpha section appears again",
                "page_start": 5,
                "page_end": 5,
                "chunk_index": 7,
                "qdrant_id": "kw-1",
                "token_count": 5,
            },
        ],
    )
    conn.close()

    resp = client.get("/search?q=alpha&mode=keyword&limit=10")
    assert resp.status_code == 200
    results = resp.json()["results"]
    by_qdrant = {r["chunk_index"]: r for r in results}
    # Both chunks match "alpha"; the second must report chunk_index 7 (not 0).
    assert {r["chunk_index"] for r in results} == {0, 7}
    assert by_qdrant[7]["page_start"] == 5


def test_ingest_non_pdf_returns_415(client):
    """Regression (#138): non-PDF bytes are rejected with 415 before extraction."""
    response = client.post(
        "/papers/ingest",
        files={"file": ("fake.pdf", BytesIO(b"<html>rate limited</html>"), "application/pdf")},
    )
    assert response.status_code == 415
    assert "Not a PDF" in response.json()["detail"]


def test_ingest_empty_body_returns_415(client):
    """An empty upload has no %PDF- header and is rejected with 415."""
    response = client.post(
        "/papers/ingest",
        files={"file": ("empty.pdf", BytesIO(b""), "application/pdf")},
    )
    assert response.status_code == 415


def test_ingest_failed_paper_can_be_reingest(client):
    """Regression (#145): a paper stuck in 'failed' status can be re-uploaded."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [{"page": 1, "text": "Retry content"}]

        # First upload fails at embedding (503 = upstream down)
        client.app.state.embedder.embed = AsyncMock(side_effect=httpx.ConnectError("embed down"))
        r1 = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("retry.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert r1.status_code == 503

        # Second upload of same PDF should succeed (not 409) once embedding is back
        client.app.state.embedder.embed = AsyncMock(return_value=[[0.1] * 768])
        r2 = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("retry.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert r2.status_code == 200, r2.json()


def test_ingest_indexed_paper_still_409(client):
    """Successfully indexed paper still returns 409 on duplicate upload (#145)."""
    pdf_content = create_minimal_pdf()

    with patch("academic_paper.server.extract_text") as mock_extract:
        mock_extract.return_value = [{"page": 1, "text": "Already indexed"}]

        r1 = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("dup.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert r1.status_code == 200

        r2 = client.post(
            "/papers/ingest?wait=true",
            files={"file": ("dup.pdf", BytesIO(pdf_content), "application/pdf")},
        )
        assert r2.status_code == 409


def test_write_endpoints_require_api_key_when_configured(client, temp_db):
    """Regression (#146): write endpoints return 401 when API_KEY is set and key missing."""
    with patch.object(settings, "api_key", "secret-key"):
        # No key → 401
        r = client.post(
            "/papers/ingest",
            files={"file": ("a.pdf", BytesIO(create_minimal_pdf()), "application/pdf")},
        )
        assert r.status_code == 401

        # Wrong key → 401
        r2 = client.post(
            "/papers/ingest",
            files={"file": ("a.pdf", BytesIO(create_minimal_pdf()), "application/pdf")},
            headers={"X-API-Key": "wrong"},
        )
        assert r2.status_code == 401

        # Correct key → proceeds past auth (may fail later, but not 401)
        with patch("academic_paper.server.extract_text") as mock_extract:
            mock_extract.return_value = [{"page": 1, "text": "Auth test"}]
            r3 = client.post(
                "/papers/ingest?wait=true",
                files={"file": ("b.pdf", BytesIO(create_minimal_pdf()), "application/pdf")},
                headers={"X-API-Key": "secret-key"},
            )
        assert r3.status_code != 401


def test_write_endpoints_pass_without_api_key_when_unconfigured(client):
    """When API_KEY is empty (default), write endpoints accept requests without key."""
    with patch.object(settings, "api_key", ""):
        with patch("academic_paper.server.extract_text") as mock_extract:
            mock_extract.return_value = [{"page": 1, "text": "No auth needed"}]
            r = client.post(
                "/papers/ingest?wait=true",
                files={"file": ("c.pdf", BytesIO(create_minimal_pdf()), "application/pdf")},
            )
        assert r.status_code != 401


def test_read_endpoints_require_api_key_when_configured(client, temp_db):
    """Regression (#241): read endpoints (list/detail/summary/search) also return 401
    when API_KEY is set and the key is missing or wrong — the read side must not be
    an unauthenticated hole while write/job endpoints are gated (#190)."""
    with patch.object(settings, "api_key", "secret-key"):
        assert client.get("/papers").status_code == 401
        assert client.get("/papers", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/papers", headers={"X-API-Key": "secret-key"}).status_code == 200

        assert client.get("/papers/1").status_code == 401
        assert client.get("/papers/1", headers={"X-API-Key": "secret-key"}).status_code in (200, 404)

        assert client.get("/papers/1/summary").status_code == 401
        assert client.get("/papers/1/summary", headers={"X-API-Key": "secret-key"}).status_code in (200, 404)

        assert client.get("/search?q=alpha").status_code == 401
        assert client.get("/search?q=alpha", headers={"X-API-Key": "secret-key"}).status_code != 401


def test_read_endpoints_pass_without_api_key_when_unconfigured(client):
    """When API_KEY is empty (default), read endpoints accept requests without key."""
    with patch.object(settings, "api_key", ""):
        assert client.get("/papers").status_code != 401
        assert client.get("/papers/1").status_code != 401
        assert client.get("/papers/1/summary").status_code != 401
        assert client.get("/search?q=alpha").status_code != 401


def _lifespan_client(temp_db):
    """Build a real TestClient (runs lifespan) with I/O services mocked out,
    mirroring the `client` fixture but as a plain context manager so callers
    can control caplog capture around the startup event."""
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])
    mock_qdrant = MagicMock()
    mock_qdrant.aupsert = AsyncMock(return_value=None)
    mock_qdrant.asearch = AsyncMock(return_value=[])
    mock_qdrant.adelete_by_paper_id = AsyncMock(return_value=None)
    mock_qdrant.aensure_collection = AsyncMock(return_value=None)
    mock_qdrant.aclose = AsyncMock(return_value=None)
    mock_qdrant.client.get_collections = MagicMock(return_value=None)
    return (
        patch.object(settings, "academic_db", temp_db),
        patch("academic_paper.server.EmbedderClient", return_value=mock_embedder),
        patch("academic_paper.server.QdrantStore", return_value=mock_qdrant),
    )


def test_startup_warns_when_api_key_unset(capsys, temp_db):
    """Regression (#241): lifespan logs an explicit warning when API_KEY is empty so
    an accidentally-unauthenticated deployment isn't silently invisible at startup.

    configure_logging() replaces root handlers with a stdout JSON handler (removing
    caplog's), so the startup log is asserted via captured stdout instead of caplog.
    """
    p1, p2, p3 = _lifespan_client(temp_db)
    with patch.object(settings, "api_key", ""), p1, p2, p3:
        with TestClient(app):
            pass

    assert "API_KEY is not set" in capsys.readouterr().out


def test_startup_no_warning_when_api_key_set(capsys, temp_db):
    """No spurious warning when API_KEY is configured."""
    p1, p2, p3 = _lifespan_client(temp_db)
    with patch.object(settings, "api_key", "secret-key"), p1, p2, p3:
        with TestClient(app):
            pass

    assert "API_KEY is not set" not in capsys.readouterr().out


# --- _cleanup_orphaned_ingests tests (#194) ---


@pytest.mark.anyio
async def test_cleanup_orphaned_ingests_no_stuck_papers(temp_db):
    """_cleanup_orphaned_ingests does nothing when no papers are stuck."""
    mock_app = MagicMock()
    mock_qdrant = MagicMock()
    mock_qdrant.adelete_by_paper_id = AsyncMock()
    mock_app.state.vector_store = mock_qdrant

    with patch.object(settings, "academic_db", temp_db):
        await _cleanup_orphaned_ingests(mock_app)

    mock_qdrant.adelete_by_paper_id.assert_not_called()


@pytest.mark.anyio
async def test_cleanup_orphaned_ingests_cleans_stuck_paper(temp_db):
    """_cleanup_orphaned_ingests deletes Qdrant vectors and marks papers failed."""
    conn = get_connection(temp_db)
    paper_id = save_paper(conn, "stuck.pdf", "hash_stuck")
    conn.execute("UPDATE papers SET status = 'pending' WHERE id = ?", (paper_id,))
    conn.commit()
    conn.close()

    mock_app = MagicMock()
    mock_qdrant = MagicMock()
    mock_qdrant.adelete_by_paper_id = AsyncMock()
    mock_app.state.vector_store = mock_qdrant

    with patch.object(settings, "academic_db", temp_db):
        await _cleanup_orphaned_ingests(mock_app)

    mock_qdrant.adelete_by_paper_id.assert_called_once_with(paper_id)

    updated = get_connection(temp_db).execute("SELECT status FROM papers WHERE id = ?", (paper_id,)).fetchone()
    assert updated["status"] == "failed"


@pytest.mark.anyio
async def test_cleanup_orphaned_ingests_handles_db_error_non_fatal():
    """_cleanup_orphaned_ingests swallows DB errors (non-fatal startup)."""
    mock_app = MagicMock()
    mock_app.state.vector_store = MagicMock()

    with patch("academic_paper.server.db_connection", side_effect=Exception("db down")):
        await _cleanup_orphaned_ingests(mock_app)  # must not raise
