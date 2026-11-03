"""Polling-branch tests for scripts/ingest_client.submit_and_wait (#348).

Complements tests/test_ingest_client.py (auth header, 409, simple failed): covers the
pending->done state transition, result expansion, failed-without-errors, timeout and
poll-side HTTP errors. No network: respx mocks httpx; time.sleep/monotonic are patched.
"""

import io
import sys
from pathlib import Path

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import ingest_client  # noqa: E402

API = "http://api.test"


@pytest.fixture(autouse=True)
def _no_api_key_no_sleep(monkeypatch):
    monkeypatch.delenv("PAPER_API_KEY", raising=False)
    monkeypatch.setattr(ingest_client.time, "sleep", lambda _s: None)


def _submit(client: httpx.Client, **kw) -> dict:
    kw.setdefault("poll_timeout", 5)
    kw.setdefault("poll_interval", 0)
    return ingest_client.submit_and_wait(client, API, "x.pdf", io.BytesIO(b"%PDF-"), {"source": "arxiv"}, **kw)


def _mock_submit(paper_id=7):
    respx.post(f"{API}/papers/ingest").mock(
        return_value=httpx.Response(202, json={"job_id": "job-1", "paper_id": paper_id})
    )


@respx.mock
def test_polls_through_pending_and_running_until_done(monkeypatch):
    _mock_submit()
    sleeps: list[float] = []
    monkeypatch.setattr(ingest_client.time, "sleep", sleeps.append)
    job_route = respx.get(f"{API}/jobs/job-1").mock(
        side_effect=[
            httpx.Response(200, json={"id": "job-1", "status": "pending"}),
            httpx.Response(200, json={"id": "job-1", "status": "running"}),
            httpx.Response(
                200,
                json={"id": "job-1", "status": "done", "result": {"paper_id": 7, "chunks": 3, "status": "indexed"}},
            ),
        ]
    )

    with httpx.Client() as client:
        result = _submit(client, poll_interval=0.5)

    assert result == {"paper_id": 7, "chunks": 3, "status": "ingested"}
    assert job_route.call_count == 3
    assert sleeps == [0.5, 0.5]  # slept between the two non-terminal polls only


@respx.mock
def test_done_without_result_falls_back_to_submit_paper_id():
    _mock_submit(paper_id=42)
    respx.get(f"{API}/jobs/job-1").mock(return_value=httpx.Response(200, json={"id": "job-1", "status": "done"}))

    with httpx.Client() as client:
        result = _submit(client)

    assert result == {"paper_id": 42, "status": "ingested"}


@respx.mock
def test_done_result_status_is_overridden_with_ingested():
    _mock_submit()
    respx.get(f"{API}/jobs/job-1").mock(
        return_value=httpx.Response(200, json={"status": "done", "result": {"paper_id": 9, "status": "indexed"}})
    )

    with httpx.Client() as client:
        result = _submit(client)

    assert result["status"] == "ingested"
    assert result["paper_id"] == 9  # result's paper_id wins over the submit response's


@respx.mock
def test_failed_without_errors_uses_default_message():
    _mock_submit()
    respx.get(f"{API}/jobs/job-1").mock(return_value=httpx.Response(200, json={"status": "failed", "errors": []}))

    with httpx.Client() as client, pytest.raises(RuntimeError, match="ingest job failed"):
        _submit(client)


@respx.mock
def test_failed_raises_first_error_message_after_pending_poll():
    _mock_submit()
    respx.get(f"{API}/jobs/job-1").mock(
        side_effect=[
            httpx.Response(200, json={"status": "running"}),
            httpx.Response(200, json={"status": "failed", "errors": ["first", "second"]}),
        ]
    )

    with httpx.Client() as client, pytest.raises(RuntimeError) as exc_info:
        _submit(client)

    assert str(exc_info.value) == "first"


@respx.mock
def test_timeout_raises_timeout_error(monkeypatch):
    _mock_submit()
    clock = iter([0.0, 1.0, 2.0, 11.0, 12.0, 13.0])  # deadline = 0 + 10; third check exceeds it
    monkeypatch.setattr(ingest_client.time, "monotonic", lambda: next(clock))
    job_route = respx.get(f"{API}/jobs/job-1").mock(return_value=httpx.Response(200, json={"status": "running"}))

    with httpx.Client() as client, pytest.raises(TimeoutError, match="job-1.*10s"):
        _submit(client, poll_timeout=10)

    assert job_route.call_count == 3


@respx.mock
def test_done_on_last_poll_before_deadline_does_not_time_out(monkeypatch):
    """A done job is returned even if the clock is already past the deadline (status checked first)."""
    _mock_submit()
    clock = iter([0.0, 100.0])
    monkeypatch.setattr(ingest_client.time, "monotonic", lambda: next(clock))
    respx.get(f"{API}/jobs/job-1").mock(
        return_value=httpx.Response(200, json={"status": "done", "result": {"paper_id": 7}})
    )

    with httpx.Client() as client:
        assert _submit(client, poll_timeout=10)["status"] == "ingested"


@respx.mock
def test_poll_http_error_propagates():
    _mock_submit()
    respx.get(f"{API}/jobs/job-1").mock(return_value=httpx.Response(404))

    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        _submit(client)


@respx.mock
def test_submit_non_2xx_non_409_propagates_without_polling():
    respx.post(f"{API}/papers/ingest").mock(return_value=httpx.Response(500))
    job_route = respx.get(f"{API}/jobs/job-1").mock(return_value=httpx.Response(200, json={"status": "done"}))

    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        _submit(client)

    assert not job_route.called


@respx.mock
def test_poll_requests_carry_api_key(monkeypatch):
    monkeypatch.setenv("PAPER_API_KEY", "sekrit")
    _mock_submit()
    job_route = respx.get(f"{API}/jobs/job-1").mock(return_value=httpx.Response(200, json={"status": "done"}))

    with httpx.Client() as client:
        _submit(client)

    assert job_route.calls[0].request.headers["X-API-Key"] == "sekrit"


# --- transient poll errors (#825) -------------------------------------------------


def _done(paper_id=7):
    return httpx.Response(200, json={"status": "done", "result": {"paper_id": paper_id, "chunks": 3}})


@respx.mock
def test_poll_read_timeout_then_done_is_ingested():
    """A ReadTimeout on one poll must not fail an ingest that later completes (#825)."""
    _mock_submit()
    job_route = respx.get(f"{API}/jobs/job-1").mock(side_effect=[httpx.ReadTimeout("slow"), _done()])

    with httpx.Client() as client:
        result = _submit(client)

    assert result["status"] == "ingested"
    assert job_route.call_count == 2


@respx.mock
@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_poll_5xx_then_done_is_ingested(code):
    _mock_submit()
    respx.get(f"{API}/jobs/job-1").mock(side_effect=[httpx.Response(code), _done()])

    with httpx.Client() as client:
        assert _submit(client)["status"] == "ingested"


@respx.mock
def test_poll_connect_error_then_done_is_ingested():
    _mock_submit()
    respx.get(f"{API}/jobs/job-1").mock(side_effect=[httpx.ConnectError("restarting"), _done()])

    with httpx.Client() as client:
        assert _submit(client)["status"] == "ingested"


@respx.mock
def test_poll_error_counter_resets_after_successful_poll():
    """Errors are *consecutive*: interleaved successes keep the budget from running out."""
    _mock_submit()
    running = httpx.Response(200, json={"status": "running"})
    err = httpx.ReadTimeout("slow")
    respx.get(f"{API}/jobs/job-1").mock(side_effect=[err, err, running, err, err, running, err, err, _done()])

    with httpx.Client() as client:
        assert _submit(client, max_poll_errors=3)["status"] == "ingested"


@respx.mock
def test_poll_persistent_transient_errors_raise_after_limit():
    _mock_submit()
    job_route = respx.get(f"{API}/jobs/job-1").mock(side_effect=httpx.ReadTimeout("down"))

    with httpx.Client() as client, pytest.raises(httpx.ReadTimeout):
        _submit(client, max_poll_errors=3)

    assert job_route.call_count == 3


@respx.mock
@pytest.mark.parametrize("code", [401, 403, 404])
def test_poll_permanent_4xx_is_not_retried(code):
    _mock_submit()
    job_route = respx.get(f"{API}/jobs/job-1").mock(return_value=httpx.Response(code))

    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        _submit(client)

    assert job_route.call_count == 1


@respx.mock
def test_poll_transient_errors_respect_deadline(monkeypatch):
    _mock_submit()
    clock = iter([0.0, 1.0, 11.0, 12.0])
    monkeypatch.setattr(ingest_client.time, "monotonic", lambda: next(clock))
    respx.get(f"{API}/jobs/job-1").mock(side_effect=httpx.ReadTimeout("slow"))

    with httpx.Client() as client, pytest.raises(TimeoutError):
        _submit(client, poll_timeout=10, max_poll_errors=50)
