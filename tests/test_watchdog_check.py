"""Unit tests for scripts/watchdog_check.py (#499)."""

import io
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import watchdog_check as wc  # noqa: E402

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _run(hours_ago, status="completed", conclusion="success", name="wf"):
    created = (NOW - timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")
    return {"status": status, "conclusion": conclusion, "createdAt": created, "workflowName": name}


def test_queued_over_threshold():
    v = wc.check_runs([_run(3.5, "queued", "")], NOW)
    assert len(v) == 1 and "queued" in v[0]


def test_queued_exactly_threshold_ok():
    assert wc.check_runs([_run(3, "queued", "")], NOW) == []


def test_latest_cancelled_or_failure():
    for c in ("cancelled", "failure"):
        v = wc.check_runs([_run(1, conclusion=c), _run(25)], NOW)
        assert len(v) == 1 and c in v[0]


def test_older_failure_not_flagged_when_latest_success():
    assert wc.check_runs([_run(30, conclusion="failure"), _run(1)], NOW) == []


def test_success_age_over_threshold():
    v = wc.check_runs([_run(27)], NOW, max_success_age_hours=26)
    assert len(v) == 1 and "last success" in v[0]


def test_success_age_exactly_threshold_ok():
    assert wc.check_runs([_run(26)], NOW, max_success_age_hours=26) == []


def test_no_runs():
    assert wc.check_runs([], NOW) == []
    assert len(wc.check_runs([], NOW, max_success_age_hours=26)) == 1


def test_never_succeeded():
    v = wc.check_runs([_run(1, conclusion="failure")], NOW, max_success_age_hours=26)
    assert len(v) == 2


def test_in_progress_not_flagged():
    runs = [_run(10, "in_progress", ""), _run(1)]
    assert wc.check_runs(runs, NOW, max_success_age_hours=26) == []


def test_workflow_filter():
    runs = [_run(1, conclusion="failure", name="other"), _run(1, name="wf")]
    assert wc.check_runs(runs, NOW, workflow="wf") == []
    assert len(wc.check_runs(runs, NOW, workflow="other")) == 1


def test_main_exit_codes(monkeypatch, capsys):
    now = NOW.isoformat()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps([_run(1, conclusion="cancelled")])))
    assert wc.main(["--now", now]) == 1
    assert "cancelled" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps([_run(1)])))
    assert wc.main(["--now", now]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert wc.main(["--now", now]) == 0
