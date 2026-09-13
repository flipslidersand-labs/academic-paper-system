"""Shared helpers for paper collector scripts.

Provides:
  - download_pdf   : stream a PDF URL into a temp file (avoid full-content memory)
  - ingest_pdf     : call submit_and_wait and surface server detail on HTTP error
  - run_collect    : canonical ingest loop — counts, summary print, JSON write, exit
"""

import contextlib
import ipaddress
import json
import re
import socket
import sys
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from ingest_client import submit_and_wait

from academic_paper.config import settings

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
_PDF_MAGIC = b"%PDF-"
_MAX_REDIRECTS = 5


def _resolve_safe(host: str, port: int) -> list:
    """Resolve *host* and return only the globally-routable addrinfo results.

    Raise ValueError when the host doesn't resolve or resolves to nothing
    but private/loopback/link-local addresses (e.g. the 169.254.169.254
    cloud metadata endpoint). *port* must match the port that will actually
    be connected to, so the returned addrinfo tuples are directly reusable
    by ``_pin_resolution`` for the real connection (#278) instead of
    triggering a second, independent DNS lookup that a DNS-rebinding attack
    could answer differently.
    """
    try:
        addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host {host!r}: {exc}") from exc
    safe = [info for info in addrinfos if ipaddress.ip_address(info[4][0]).is_global]
    if not safe:
        raise ValueError(f"Refusing to fetch URL resolving to non-public address(es): {host}")
    return safe


def assert_safe_url(url: str) -> None:
    """Raise ValueError if *url* is not a safe http(s) URL to fetch (SSRF guard).

    Third-party indexes (e.g. OpenAlex) return arbitrary externally-supplied
    URLs. Reject anything but http(s), and reject hosts that resolve to a
    private, loopback, link-local, or otherwise non-public address (e.g. the
    169.254.169.254 cloud metadata endpoint) so a malicious record can't make
    the collector reach into internal infrastructure.

    This only performs the one-off check used by tests and callers that
    don't need the pinned-connection guarantee; ``download_pdf`` re-validates
    every hop itself (including redirects) via ``_resolve_safe``.
    """
    host, _port = _split_safe_url(url)
    _resolve_safe(host, _port)


def _split_safe_url(url: str) -> tuple[str, int]:
    """Validate scheme/host and return (host, port) for *url*, or raise ValueError."""
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Refusing to fetch URL with disallowed scheme {parts.scheme!r}: {url}")
    host = parts.hostname
    if not host:
        raise ValueError(f"Refusing to fetch URL with no host: {url}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return host, port


@contextlib.contextmanager
def _pin_resolution(host: str, addrinfos: list):
    """Force ``socket.getaddrinfo(host, ...)`` to return the already-validated
    *addrinfos* for the duration of the block.

    Without this, the SSRF check in ``_resolve_safe`` and the actual TCP
    connect made by the HTTP client perform two independent DNS lookups; a
    malicious domain with a short TTL can answer the first with a public IP
    and the second with a private one (DNS rebinding, #278). Pinning makes
    both steps consume the exact same resolution.

    Collector scripts process papers strictly sequentially in a single
    thread (see ``run_collect``), so process-wide monkeypatching here is
    safe; this must not be used from concurrent/async code paths.
    """
    real_getaddrinfo = socket.getaddrinfo

    def _patched(node, *args, **kwargs):
        if node == host:
            return addrinfos
        return real_getaddrinfo(node, *args, **kwargs)

    socket.getaddrinfo = _patched
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo


@contextlib.contextmanager
def download_pdf(client: httpx.Client, url: str, timeout: int = 60, max_mb: int | None = None):
    """Stream a PDF from *url* using *client* into a named temp file.

    Yields the temp-file path.  The file is deleted on exit.
    Raises ValueError when the URL (or any redirect hop) fails the SSRF
    safety check, when more than ``_MAX_REDIRECTS`` redirects occur, when
    the response content-type is not exactly "application/pdf" or
    "application/x-pdf" (falling back to the URL ending in ".pdf" only when
    the content-type doesn't match), when the downloaded body doesn't start
    with the "%PDF-" magic number (guards against a server that spoofs the
    content-type header), or when the cumulative downloaded size exceeds
    *max_mb* (defaults to ``settings.max_upload_mb``, the same limit the
    server enforces on ``/papers/ingest``), aborting the stream immediately
    to avoid exhausting disk on an oversized or unbounded response.

    Redirects are followed manually (``follow_redirects=False``) so each
    hop is re-validated against the SSRF guard before being fetched —
    httpx's built-in redirect following never re-checks a Location header,
    which previously let a third-party API's URL redirect straight past the
    guard into internal infrastructure (#278).
    """
    max_bytes = (max_mb if max_mb is not None else settings.max_upload_mb) * 1024 * 1024
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        current_url = url
        for _hop in range(_MAX_REDIRECTS + 1):
            host, port = _split_safe_url(current_url)
            addrinfos = _resolve_safe(host, port)
            with _pin_resolution(host, addrinfos):
                with client.stream("GET", current_url, timeout=timeout, follow_redirects=False) as resp:
                    if getattr(resp, "is_redirect", False):
                        location = resp.headers.get("location")
                        if not location:
                            raise ValueError(f"Redirect response missing Location header: {current_url}")
                        current_url = urljoin(current_url, location)
                        continue
                    resp.raise_for_status()
                    ct = resp.headers.get("content-type", "").lower()
                    ct_base = ct.split(";")[0].strip()
                    if ct_base not in _ALLOWED_PDF_CONTENT_TYPES and not current_url.lower().endswith(".pdf"):
                        raise ValueError(f"Not a PDF (content-type: {ct})")
                    written = 0
                    checked_magic = False
                    with open(tmp_path, "wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            if not checked_magic:
                                checked_magic = True
                                if not chunk.startswith(_PDF_MAGIC):
                                    raise ValueError(f"Not a PDF (missing %PDF- magic number): {current_url}")
                            written += len(chunk)
                            if written > max_bytes:
                                raise ValueError(
                                    f"PDF exceeds max size ({max_bytes // (1024 * 1024)} MB): {current_url}"
                                )
                            fh.write(chunk)
                    yield tmp_path
                    return
        raise ValueError(f"Too many redirects (>{_MAX_REDIRECTS}) fetching {url}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def ingest_pdf(
    client: httpx.Client,
    api_url: str,
    file_name: str,
    tmp_path: str,
    metadata: dict,
    poll_timeout: int = 300,
) -> dict:
    """Read *tmp_path* and submit it via submit_and_wait.

    On HTTP error, extracts ``{"detail": ...}`` from the response body so the
    error message shown in cron logs is the server reason, not just the status.
    """
    try:
        with open(tmp_path, "rb") as fh:
            return submit_and_wait(client, api_url, file_name, fh, metadata, poll_timeout=poll_timeout)
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text[:200])
        except Exception:
            detail = exc.response.text[:200]
        raise RuntimeError(f"HTTP {exc.response.status_code}: {detail}") from exc


def run_collect(
    source_label: str,
    papers: list[dict],
    ingest_fn,
    summary_file: str | None,
    *,
    fetch_error: str | None = None,
) -> None:
    """Run the ingest loop for *papers* and handle summary / exit-code.

    ``ingest_fn(client, paper) -> dict`` must return a dict containing at
    minimum ``"status"`` (``"ingested"`` or ``"duplicate"``) and ``"label"``
    (string used for per-paper log output).  Any additional keys are stored
    verbatim in the detail list inside the summary JSON.

    Exits with code 1 when any paper failed.
    """
    counts: dict[str, int] = {"ingested": 0, "duplicate": 0, "failed": 0}
    detail: list[dict] = []

    with httpx.Client() as client:
        for paper in papers:
            try:
                result = ingest_fn(client, paper)
                label = result.pop("label", "?")
                status = result["status"]
                counts[status] = counts.get(status, 0) + 1
                tag = "OK  " if status == "ingested" else "SKIP"
                title = (paper.get("title") or "")[:50] if isinstance(paper, dict) else ""
                # Strip control chars (incl. newlines) before printing — externally
                # sourced titles can otherwise forge/inject cron log lines (#233).
                title = _CONTROL_CHARS_RE.sub(" ", title)
                print(f"  {tag} [{label}] {title}")
                detail.append({"label": label, "status": status, **result})
            except Exception as exc:
                label = _paper_label(paper)
                counts["failed"] += 1
                print(f"  ERR  [{label}] {exc}", file=sys.stderr)
                detail.append({"label": label, "status": "failed", "error": str(exc)})

    tag_lower = source_label.lower().replace(" ", "-")
    print(f"\n## {source_label} Summary")
    print(f"- Found    : {len(papers)}")
    print(f"- Ingested : {counts['ingested']}")
    print(f"- Duplicate: {counts['duplicate']}")
    print(f"- Failed   : {counts['failed']}")

    if summary_file:
        payload: dict = {**counts, "fetched": len(papers), "detail": detail}
        if fetch_error is not None:
            payload["fetch_error"] = fetch_error
        with open(summary_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[{tag_lower}] summary written to {summary_file}")

    if counts["failed"] > 0:
        sys.exit(1)


def _paper_label(paper: dict) -> str:
    """Best-effort one-line identifier for a paper dict, used in error logs."""
    for key in ("arxiv_id", "paperId", "pmc_id", "id"):
        val = paper.get(key)
        if val:
            return str(val)[:16]
    return "?"
