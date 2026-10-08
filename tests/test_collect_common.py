"""Tests for scripts/_collect_common.py helpers."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# scripts/ is added to sys.path in tests/conftest.py (#272).
from _collect_common import (  # noqa: E402
    _paper_label,
    add_common_args,
    assert_safe_url,
    download_pdf,
    format_date_range,
    ingest_downloaded,
    ingest_pdf,
    run_collect,
    write_summary,
)

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


def test_download_pdf_rejects_content_type_with_pdf_as_substring():
    """A spoofed content-type merely containing 'pdf' (e.g. text/pdfxml) must not
    pass on substring matching alone (#275)."""
    client = _mock_stream_response(b"<html>not a pdf</html>", "text/pdfxml")
    with pytest.raises(ValueError, match="Not a PDF"):
        with download_pdf(client, "http://example.com/some-page") as _:
            pass


def test_download_pdf_rejects_body_without_pdf_magic_number():
    """Even when Content-Type is spoofed as application/pdf and the URL ends in
    .pdf, a body that doesn't start with the %PDF- magic number is rejected (#275)."""
    client = _mock_stream_response(b"<html>fake pdf</html>", "application/pdf")
    with pytest.raises(ValueError, match="magic number"):
        with download_pdf(client, "http://example.com/paper.pdf") as _:
            pass


def test_download_pdf_raises_when_size_exceeds_limit():
    """Streaming aborts as soon as cumulative bytes exceed max_mb, without buffering the rest."""

    class _FakeOversizedResp:
        headers = {"content-type": "application/pdf"}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=65536):
            # Each chunk is 1 MB; limit below is 1 MB so the 2nd chunk trips it.
            # First chunk starts with the PDF magic number so it passes that check.
            yield b"%PDF-" + b"x" * (1024 * 1024 - 5)
            for _ in range(4):
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
            yield b"%PDF-"

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
# download_pdf redirect handling (#278: httpx follow_redirects=True never
# re-validated the Location header against the SSRF guard)
# ---------------------------------------------------------------------------


def _fake_redirect_resp(location: str):
    class _FakeRedirectResp:
        is_redirect = True
        headers = {"location": location}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=65536):
            return iter(())

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    return _FakeRedirectResp()


def _fake_final_resp(content: bytes, content_type: str = "application/pdf"):
    class _FakeFinalResp:
        is_redirect = False
        headers = {"content-type": content_type}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=65536):
            yield content

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    return _FakeFinalResp()


def test_download_pdf_rejects_redirect_to_unsafe_host():
    """A redirect Location pointing at cloud metadata must be rejected before it is fetched (#278)."""

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            assert url == "http://example.com/redirect"
            return _fake_redirect_resp("http://169.254.169.254/latest/meta-data/")

    with pytest.raises(ValueError, match="non-public address"):
        with download_pdf(_FakeClient(), "http://example.com/redirect") as _:
            pass


def test_download_pdf_follows_safe_redirect_to_final_pdf():
    """A redirect to another public host is followed and re-validated per hop (#278)."""

    calls = []

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            calls.append(url)
            if url == "http://example.com/redirect":
                return _fake_redirect_resp("http://example.org/final.pdf")
            return _fake_final_resp(b"%PDF-final")

    with download_pdf(_FakeClient(), "http://example.com/redirect") as path:
        assert Path(path).read_bytes() == b"%PDF-final"
    assert calls == ["http://example.com/redirect", "http://example.org/final.pdf"]


def test_download_pdf_raises_on_too_many_redirects():
    class _FakeClient:
        def stream(self, method, url, **kwargs):
            return _fake_redirect_resp("http://example.com/redirect")

    with pytest.raises(ValueError, match="Too many redirects"):
        with download_pdf(_FakeClient(), "http://example.com/redirect") as _:
            pass


# ---------------------------------------------------------------------------
# DNS rebinding: the resolution used for the SSRF check must be the exact
# resolution used for the connection (#278)
# ---------------------------------------------------------------------------


def test_pin_resolution_forces_getaddrinfo_to_validated_result():
    import socket

    from _collect_common import _pin_resolution

    real_getaddrinfo = socket.getaddrinfo
    fake_addrinfos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 443))]
    with _pin_resolution("example.com", fake_addrinfos):
        # Even though example.com really resolves elsewhere, the pin must win.
        assert socket.getaddrinfo("example.com", 443) == fake_addrinfos
        # A different host must still resolve normally (pin is host-scoped).
        assert socket.getaddrinfo is not real_getaddrinfo
    # Pin must not leak past the context manager.
    assert socket.getaddrinfo is real_getaddrinfo


def test_pin_resolution_rejects_nested_use_and_releases_afterwards():
    """#314: a second pin while one is active raises, and the state is restored."""
    import socket

    from _collect_common import _pin_resolution

    real_getaddrinfo = socket.getaddrinfo
    ai = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 443))]
    with _pin_resolution("a.example", ai):
        patched = socket.getaddrinfo
        with pytest.raises(RuntimeError, match="already active"):
            with _pin_resolution("b.example", ai):
                pass
        # The failed inner entry must not have disturbed the outer pin.
        assert socket.getaddrinfo is patched
    assert socket.getaddrinfo is real_getaddrinfo
    # Lock released: a fresh pin works again.
    with _pin_resolution("a.example", ai):
        pass


def test_pin_resolution_rejects_mismatched_port():
    """#314: the pinned addrinfos are only valid for the validated port."""
    import socket

    from _collect_common import _pin_resolution

    ai = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 443))]
    with _pin_resolution("example.com", ai):
        assert socket.getaddrinfo("example.com", 443) == ai
        assert socket.getaddrinfo("example.com", "443") == ai
        assert socket.getaddrinfo("example.com", port=443) == ai
        with pytest.raises(RuntimeError, match="pinned"):
            socket.getaddrinfo("example.com", 8080)
        with pytest.raises(RuntimeError, match="pinned"):
            socket.getaddrinfo("example.com", None)


def test_download_pdf_pins_resolution_during_fetch():
    """download_pdf must pin socket.getaddrinfo so the client's own connect
    can't re-resolve a rebinding domain to a different (private) address."""
    import socket

    seen_during_stream = {}

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            # Simulate the HTTP client's own DNS resolution happening inside stream().
            seen_during_stream["addrinfo"] = socket.getaddrinfo("example.com", 80, type=socket.SOCK_STREAM)
            return _fake_final_resp(b"%PDF-ok")

    with download_pdf(_FakeClient(), "http://example.com/paper.pdf") as _:
        pass

    for _family, _type, _proto, _canon, sockaddr in seen_during_stream["addrinfo"]:
        assert ipaddress_is_global(sockaddr[0])


def ipaddress_is_global(ip: str) -> bool:
    import ipaddress

    return ipaddress.ip_address(ip).is_global


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


# ---------------------------------------------------------------------------
# add_common_args / format_date_range / write_summary / ingest_downloaded (#525)
# ---------------------------------------------------------------------------


def test_add_common_args_defaults():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([])
    assert args.api_url == "http://localhost:8020"
    assert args.poll_timeout == 300
    assert args.summary_file is None
    assert args.from_date == ""
    assert args.until_date == ""


def test_add_common_args_parses_values_and_validates_dates():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        ["--api-url", "http://x:1", "--poll-timeout", "5", "--summary-file", "s.json",
         "--from-date", "2025-01-02", "--until-date", "2025-02-03"]
    )  # fmt: skip
    assert (args.api_url, args.poll_timeout, args.summary_file) == ("http://x:1", 5, "s.json")
    assert (args.from_date, args.until_date) == ("2025-01-02", "2025-02-03")
    with pytest.raises(SystemExit):
        parser.parse_args(["--from-date", "not-a-date"])


def test_format_date_range():
    assert format_date_range("", "") == ""
    assert format_date_range("2025-01-01", "") == " [2025-01-01 → *]"
    assert format_date_range("", "2025-02-01") == " [* → 2025-02-01]"


def test_write_summary_path_none_is_noop(tmp_path):
    write_summary(None, fetched=3)
    assert list(tmp_path.iterdir()) == []


def test_write_summary_defaults_when_counts_omitted(tmp_path):
    out = tmp_path / "s.json"
    write_summary(str(out), fetched=0)
    assert json.loads(out.read_text()) == {"ingested": 0, "duplicate": 0, "failed": 0, "fetched": 0, "detail": []}


def test_write_summary_with_fetch_error(tmp_path):
    out = tmp_path / "s.json"
    write_summary(str(out), fetched=0, fetch_error="boom")
    data = json.loads(out.read_text())
    assert data["fetch_error"] == "boom"
    assert data["fetched"] == 0


def test_write_summary_with_counts_and_detail_no_fetch_error_key(tmp_path):
    out = tmp_path / "s.json"
    counts = {"ingested": 1, "duplicate": 0, "failed": 1}
    detail = [{"label": "a", "status": "ingested"}]
    write_summary(str(out), fetched=2, counts=counts, detail=detail)
    data = json.loads(out.read_text())
    assert data == {**counts, "fetched": 2, "detail": detail}


def test_ingest_downloaded_builds_metadata_and_calls_pre_ingest():
    seen = {}

    def _pre(tmp_path):
        seen["pre"] = tmp_path

    with patch("_collect_common.ingest_pdf", return_value={"status": "ingested"}) as mock_ingest:
        result = ingest_downloaded(
            _mock_stream_response(b"%PDF-1.4 x"),
            "http://api",
            pdf_url="https://example.com/a.pdf",
            file_name="a.pdf",
            title="T",
            authors=["A", "B"],
            categories=["c"],
            published_date=None,
            source="arxiv",
            poll_timeout=7,
            pre_ingest=_pre,
        )
    assert result == {"status": "ingested"}
    assert seen["pre"].endswith(".pdf")
    args = mock_ingest.call_args.args
    assert args[1:3] == ("http://api", "a.pdf")
    assert args[4] == {
        "title": "T",
        "authors": json.dumps(["A", "B"]),
        "categories": json.dumps(["c"]),
        "published_date": "",
        "source": "arxiv",
    }
    assert args[5] == 7
