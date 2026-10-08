"""Structure guard for .github/workflows/scheduled-watchdog.yml (#499).

The watchdog exists because self-hosted-runner jobs can sit queued forever
without any step running. If it were moved back to self-hosted, lost a
monitored workflow, or gained write permissions, it would silently stop
protecting anything.
"""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "scheduled-watchdog.yml"


def _load() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _jobs() -> list[dict]:
    return list(_load()["jobs"].values())


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def test_not_self_hosted():
    for job in _jobs():
        assert "self-hosted" not in _as_list(job["runs-on"])


def test_permissions_actions_read_only():
    data = _load()
    assert data["permissions"] == {"actions": "read"}
    for job in _jobs():
        assert job.get("permissions", {"actions": "read"}) == {"actions": "read"}


def test_monitors_both_workflows():
    text = WORKFLOW.read_text()
    assert "arxiv-daily.yml" in text
    assert "portfolio-publish.yml" in text


def test_timeout_set():
    for job in _jobs():
        assert isinstance(job.get("timeout-minutes"), int)
