"""Direct tests for Job.to_dict (#666)."""

import pytest

from academic_paper.jobs import Job


@pytest.mark.parametrize("n,expected", [(0, 0), (1, 1), (10, 10), (11, 10), (25, 10)])
def test_to_dict_truncates_errors_to_first_10(n, expected):
    errors = [f"e{i}" for i in range(n)]
    job = Job(id="j", status="running", errors=errors)

    d = job.to_dict()

    assert len(d["errors"]) == expected
    assert d["errors"] == errors[:10]
    # Non-destructive: original list untouched, result is a copy.
    assert len(job.errors) == n
    assert d["errors"] is not job.errors


def test_to_dict_includes_finished_at_and_result():
    job = Job(id="j", status="done", kind="k", total=3, processed=2, failed=1, finished_at=123.5, result={"ok": 2})
    d = job.to_dict()
    assert d["finished_at"] == 123.5
    assert d["result"] == {"ok": 2}
    assert (d["id"], d["status"], d["kind"], d["total"], d["processed"], d["failed"]) == ("j", "done", "k", 3, 2, 1)


def test_to_dict_defaults_none():
    d = Job(id="j", status="pending").to_dict()
    assert d["finished_at"] is None
    assert d["result"] is None
    assert d["errors"] == []
