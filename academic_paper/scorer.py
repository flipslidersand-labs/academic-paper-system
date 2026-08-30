"""Paper scoring based on freshness and category relevance."""

import logging
from datetime import date

logger = logging.getLogger(__name__)


def compute_score(paper: dict, preferred_categories: list[str]) -> float:
    """Compute a relevance score (0.0–1.0) for a paper.

    score = freshness (0–0.5) + category_match (0–0.5)

    Freshness decays exponentially with a 30-day half-life.
    Category match is the fraction of preferred categories present.
    """
    freshness = _freshness(paper.get("published_date"))
    category = _category_match(paper.get("categories") or [], preferred_categories)
    return round(min(freshness + category, 1.0), 4)


def _freshness(published_date: str | None) -> float:
    """0.0–0.5; exponential decay, half-life = 30 days."""
    if not published_date:
        return 0.1
    try:
        days_old = (date.today() - date.fromisoformat(str(published_date)[:10])).days
        return round(0.5 * (0.5 ** (days_old / 30)), 4)
    except (ValueError, OverflowError):
        # Ingest now rejects non-ISO dates (#144); legacy rows may still hit this.
        logger.warning("Non-ISO published_date %r — freshness degraded to 0.1", published_date)
        return 0.1


def _category_match(paper_categories: list[str], preferred: list[str]) -> float:
    """0.0–0.5; proportion of preferred categories matched."""
    if not preferred:
        return 0.0
    matched = len(set(paper_categories) & set(preferred))
    return round(0.5 * min(matched / len(preferred), 1.0), 4)
