"""Request-input validators/sanitizers (moved from server.py, #614)."""

import json
import re
from datetime import date

from fastapi import HTTPException

__all__ = ["_sanitize_text", "_parse_list_field", "_validate_published_date"]

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _sanitize_text(value: str | None) -> str | None:
    """Strip control characters (incl. newlines) from externally-sourced metadata.

    title/authors/categories/source come from arXiv/OpenAlex/PubMed/Semantic
    Scholar, sources anyone can post free-form text to. Left unsanitized they
    are stored verbatim and later exposed via /papers, /summaries, and cron
    logs — enabling log injection and stored XSS (#233), the same class of
    issue file_name was fixed for in #189.
    """
    if value is None:
        return None
    cleaned = _CONTROL_CHARS_RE.sub(" ", value).strip()
    return cleaned or None


def _parse_list_field(value: str | None, field: str = "field") -> list[str] | None:
    """Parse a JSON array string or comma-separated string into a list.

    JSON arrays must contain only strings — nested objects/arrays were
    previously coerced via str(x) and stored as Python reprs (#144).
    Each resulting item is sanitized of control characters (#233).

    Raises:
        HTTPException 422: value is valid JSON but not an array of strings
        (e.g. a JSON object or number), or is a JSON array with non-string
        elements (#231).
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [s for s in (_sanitize_text(x) for x in value.split(",")) if s]
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise HTTPException(
            status_code=422, detail=f"{field} must be a JSON array of strings or a comma-separated string"
        )
    return [s for s in (_sanitize_text(x) for x in parsed) if s]


_MIN_PUBLISHED_YEAR = 1900


def _validate_published_date(value: str | None, field: str = "published_date") -> None:
    """Reject non-ISO dates, and dates outside a sane range, before they reach scoring.

    date.fromisoformat() alone accepts formally valid but meaningless dates
    (e.g. 0001-01-01, 9999-12-31) or future dates; score_all_papers/score_paper
    compute freshness from days-since-published, so such values silently
    produce nonsense scores instead of erroring (#271).
    """
    if not value:
        return
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}")
    if parsed > date.today():
        raise HTTPException(status_code=422, detail=f"{field} must not be in the future, got {value!r}")
    if parsed.year < _MIN_PUBLISHED_YEAR:
        raise HTTPException(
            status_code=422, detail=f"{field} must be on or after {_MIN_PUBLISHED_YEAR}-01-01, got {value!r}"
        )
