"""Tests for scripts/_collect_common.py helpers."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# scripts/ is added to sys.path in tests/conftest.py (#272).
from _collect_common import _paper_label, assert_safe_url, download_pdf, ingest_pdf, run_collect  # noqa: E402

# ---------------------------------------------------------------------------
# download_pdf
# ---------------------------------------------------------------------------


def _mock_stream_response(content: bytes, content_type: str = "application/pdf"):
    """Return a mock that behaves like httpx streaming context manager."""

    class _FakeStreamResp:
        headers = {"content-type": content_type}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=65536):
            yield content

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            return _FakeStreamResp()

    return _FakeClient()


# ---------------------------------------------------------------------------
# assert_safe_url / SSRF guard (#230)
# ---------------------------------------------------------------------------


def test_assert_safe_url_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="disallowed scheme"):
        assert_safe_url("file:///etc/passwd")


def test_assert_safe_url_rejects_loopback_host():
    with pytest.raises(ValueError, match="non-public address"):
        assert_safe_url("http://127.0.0.1/paper.pdf")


def test_assert_safe_url_rejects_cloud_metadata_ip():
    with pytest.raises(ValueError, match="non-public address"):
        assert_safe_url("http://169.254.169.254/latest/meta-data/")


def test_assert_safe_url_rejects_private_ip():
    with pytest.raises(ValueError, match="non-public address"):
        assert_safe_url("http://10.0.0.5/paper.pdf")


def test_assert_safe_url_allows_public_host():
    assert_safe_url("https://arxiv.org/pdf/2001.00001.pdf")


def test_download_pdf_rejects_unsafe_url_before_streaming():
    """download_pdf must refuse an SSRF-unsafe URL without ever calling client.stream."""

    class _ExplodingClient:
        def stream(self, method, url, **kwargs):
            raise AssertionError("client.stream() must not be called for an unsafe URL")

    with pytest.raises(ValueError, match="non-public address"):
        with download_pdf(_ExplodingClient(), "http://169.254.169.254/latest/meta-data/") as _:
            pass


def test_download_pdf_writes_tempfile():
    client = _mock_stream_response(b"%PDF-test")
    with download_pdf(client, "http://example.com/paper.pdf") as path:
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"%PDF-test"
    assert not Path(path).exists()


def test_download_pdf_raises_on_non_pdf_content_type():
    client = _mock_stream_response(b"not a pdf", "text/html")
    with pytest.raises(ValueError, match="Not a PDF"):
        with download_pdf(client, "http://example.com/some-page") as _:
            pass


def test_download_pdf_allows_pdf_url_extension_without_content_type():
    """URL ending in .pdf is accepted even when content-type is octet-stream."""
    client = _mock_stream_response(b"%PDF-1.4", "application/octet-stream")
    with download_pdf(client, "http://example.com/paper.pdf") as path:
        assert Path(path).read_bytes() == b"%PDF-1.4"


def test_download_pdf_raises_when_size_exceeds_limit():
    """Streaming aborts as soon as cumulative bytes exceed max_mb, without buffering the rest."""

    class _FakeOversizedResp:
        headers = {"content-type": "application/pdf"}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=65536):
            # Each chunk is 1 MB; limit below is 1 MB so the 2nd chunk trips it.
            for _ in range(5):
                yield b"x" * (1024 * 1024)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            return _FakeOversizedResp()

    client = _FakeClient()
    with pytest.raises(ValueError, match="exceeds max size"):
        with download_pdf(client, "http://example.com/huge.pdf", max_mb=1) as _:
            pass


def test_download_pdf_cleanup_on_exception():
    """Temp file is deleted even when an exception occurs inside the with block."""

    class _FailStream:
        headers = {"content-type": "application/pdf"}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=65536):
            yield b"%PDF"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class _FailClient:
        def stream(self, method, url, **kwargs):
            return _FailStream()

    client = _FailClient()
    saved_path = None
    with pytest.raises(RuntimeError):
        with download_pdf(client, "http://example.com/p.pdf") as path:
            saved_path = path
            raise RuntimeError("deliberate")
    assert saved_path is not None
    assert not Path(saved_path).exists()


# ---------------------------------------------------------------------------
# ingest_pdf
# ---------------------------------------------------------------------------


def test_ingest_pdf_calls_submit_and_wait(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    mock_client = MagicMock()
    with patch("_collect_common.submit_and_wait", return_value={"status": "ingested"}) as mock_saw:
        result = ingest_pdf(mock_client, "http://api/", "test.pdf", str(pdf), {}, poll_timeout=60)
    mock_saw.assert_called_once()
    assert result["status"] == "ingested"


def test_ingest_pdf_surfaces_server_detail_on_http_error(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    mock_client = MagicMock()

    # Build a realistic HTTPStatusError
    req = httpx.Request("POST", "http://api/papers/ingest")
    resp = httpx.Response(400, json={"detail": "invalid authors"}, request=req)
    http_err = httpx.HTTPStatusError("400", request=req, response=resp)

    with patch("_collect_common.submit_and_wait", side_effect=http_err):
        with pytest.raises(RuntimeError, match="invalid authors"):
            ingest_pdf(mock_client, "http://api/", "test.pdf", str(pdf), {})


def test_ingest_pdf_falls_back_to_text_when_no_json(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    mock_client = MagicMock()

    req = httpx.Request("POST", "http://api/papers/ingest")
    resp = httpx.Response(500, text="Internal Server Error", request=req)
    http_err = httpx.HTTPStatusError("500", request=req, response=resp)

    with patch("_collect_common.submit_and_wait", side_effect=http_err):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            ingest_pdf(mock_client, "http://api/", "test.pdf", str(pdf), {})


# ---------------------------------------------------------------------------
# run_collect
# ---------------------------------------------------------------------------


def test_run_collect_counts_and_prints(capsys):
    papers = [{"title": "Paper A"}, {"title": "Paper B"}]

    def _ingest(client, paper):
        return {"status": "ingested", "label": paper["title"][:7], "paper_id": 1}

    with patch("_collect_common.httpx.Client") as mock_cls:
        mock_cls.return_value.__enter__ = lambda s: s
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        run_collect("Test", papers, _ingest, None)

    out = capsys.readouterr().out
    assert "Ingested : 2" in out
    assert "Failed   : 0" in out


def test_run_collect_exits_1_on_failure(tmp_path):
    papers = [{"title": "Bad"}]

    def _fail(client, paper):
        raise RuntimeError("download failed")

    with patch("_collect_common.httpx.Client") as mock_cls:
        mock_cls.return_value.__enter__ = lambda s: s
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        with pytest.raises(SystemExit) as exc_info:
            run_collect("Test", papers, _fail, None)
    assert exc_info.value.code == 1


def test_run_collect_writes_summary_file(tmp_path):
    summary = tmp_path / "summary.json"
    papers = [{"title": "P1", "arxiv_id": "2001.00001"}]

    def _ingest(client, paper):
        return {"status": "duplicate", "label": "2001.00001"}

    with patch("_collect_common.httpx.Client") as mock_cls:
        mock_cls.return_value.__enter__ = lambda s: s
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        run_collect("Test", papers, _ingest, str(summary))

    data = json.loads(summary.read_text())
    assert data["duplicate"] == 1
    assert data["fetched"] == 1
    assert data["detail"][0]["status"] == "duplicate"


def test_run_collect_includes_fetch_error_in_summary(tmp_path):
    summary = tmp_path / "summary.json"

    with patch("_collect_common.httpx.Client") as mock_cls:
        mock_cls.return_value.__enter__ = lambda s: s
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        run_collect("Test", [], lambda c, p: {}, str(summary), fetch_error="timeout")

    data = json.loads(summary.read_text())
    assert data["fetch_error"] == "timeout"


# ---------------------------------------------------------------------------
# _paper_label
# ---------------------------------------------------------------------------


def test_paper_label_prefers_arxiv_id():
    assert _paper_label({"arxiv_id": "2001.12345", "id": "other"}) == "2001.12345"


def test_paper_label_falls_back_to_id():
    assert _paper_label({"id": "W123"}) == "W123"


def test_paper_label_unknown():
    assert _paper_label({}) == "?"
