#!/usr/bin/env python3
"""Detect stalled / failing scheduled workflows from `gh run list` JSON (#499).

Reads the JSON produced by

    gh run list --workflow <file> --json status,conclusion,createdAt,workflowName

on stdin (or --input FILE) and reports violations of three conditions:

  (a) a run has been ``queued`` for more than --max-queued-hours
  (b) the most recent *completed* run ended ``cancelled`` or ``failure``
  (c) the last ``success`` is older than --max-success-age-hours
      (or there has never been one)

All comparisons are strict ("exactly at the threshold" is not a violation).
``in_progress`` runs are never flagged. Exit 1 when any violation is found.
Standard library only.
"""

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta

BAD_CONCLUSIONS = {"cancelled", "failure"}


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def check_runs(
    runs: list[dict],
    now: datetime,
    max_queued_hours: float = 3,
    max_success_age_hours: float | None = None,
    workflow: str | None = None,
) -> list[str]:
    """Return human-readable violation messages (empty list = healthy)."""
    if workflow:
        runs = [r for r in runs if r.get("workflowName") == workflow]
    label = workflow or "workflow"
    violations: list[str] = []

    queue_limit = timedelta(hours=max_queued_hours)
    for run in runs:
        if run.get("status") != "queued":
            continue
        age = now - _parse_ts(run["createdAt"])
        if age > queue_limit:
            hours = age.total_seconds() / 3600
            violations.append(f"{label}: run queued since {run['createdAt']} ({hours:.1f}h > {max_queued_hours}h)")

    completed = sorted(
        (r for r in runs if r.get("status") == "completed"),
        key=lambda r: _parse_ts(r["createdAt"]),
        reverse=True,
    )
    if completed and completed[0].get("conclusion") in BAD_CONCLUSIONS:
        violations.append(
            f"{label}: latest completed run ({completed[0]['createdAt']}) is {completed[0]['conclusion']}"
        )

    if max_success_age_hours is not None:
        successes = [r for r in completed if r.get("conclusion") == "success"]
        if not successes:
            violations.append(f"{label}: no successful run found")
        else:
            age = now - _parse_ts(successes[0]["createdAt"])
            if age > timedelta(hours=max_success_age_hours):
                hours = age.total_seconds() / 3600
                violations.append(
                    f"{label}: last success {successes[0]['createdAt']} ({hours:.1f}h > {max_success_age_hours}h ago)"
                )

    return violations


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workflow", help="only consider runs whose workflowName equals this")
    p.add_argument("--max-queued-hours", type=float, default=3)
    p.add_argument("--max-success-age-hours", type=float, default=None, help="omit to skip condition (c)")
    p.add_argument("--now", help="ISO-8601 timestamp to use as 'now' (for tests)")
    p.add_argument("--input", help="read JSON from this file instead of stdin")
    args = p.parse_args(argv)

    raw = open(args.input).read() if args.input else sys.stdin.read()
    runs = json.loads(raw) if raw.strip() else []
    now = _parse_ts(args.now) if args.now else datetime.now(UTC)

    violations = check_runs(runs, now, args.max_queued_hours, args.max_success_age_hours, args.workflow)
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
