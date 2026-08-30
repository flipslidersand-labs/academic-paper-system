"""Shared argparse validators for the collector scripts (#144).

Previously --from-date/--until-date accepted raw strings, so inputs like
"2025/08/01" produced broken arXiv queries that silently returned 0 results,
and --max accepted 0 or negative values.
"""

import argparse
from datetime import date

MAX_RESULTS_CAP = 500


def iso_date(value: str) -> str:
    """argparse type: require YYYY-MM-DD (empty string allowed = no filter)."""
    if not value:
        return ""
    try:
        date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {value!r} (expected YYYY-MM-DD)")
    return value


def positive_int(value: str) -> int:
    """argparse type: integer in [1, MAX_RESULTS_CAP]."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer {value!r}")
    if not 1 <= n <= MAX_RESULTS_CAP:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_RESULTS_CAP}, got {n}")
    return n


def check_date_order(parser: argparse.ArgumentParser, from_date: str, until_date: str) -> None:
    """parser.error() when both dates are given and from > until."""
    if from_date and until_date and from_date > until_date:
        parser.error(f"--from-date ({from_date}) must be on or before --until-date ({until_date})")
