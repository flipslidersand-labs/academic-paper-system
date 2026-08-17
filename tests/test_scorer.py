"""Unit tests for academic_paper.scorer."""

from datetime import date, timedelta

import pytest

from academic_paper.scorer import _category_match, _freshness, compute_score


class TestFreshness:
    def test_today_is_near_max(self):
        score = _freshness(date.today().isoformat())
        assert 0.49 < score <= 0.5

    def test_30_days_ago_is_half(self):
        d = (date.today() - timedelta(days=30)).isoformat()
        score = _freshness(d)
        assert 0.24 < score < 0.26

    def test_very_old_paper_approaches_zero(self):
        d = (date.today() - timedelta(days=365)).isoformat()
        score = _freshness(d)
        assert score < 0.01

    def test_none_returns_fallback(self):
        assert _freshness(None) == 0.1

    def test_invalid_string_returns_fallback(self):
        assert _freshness("not-a-date") == 0.1

    def test_accepts_datetime_prefix(self):
        """published_date may be a full datetime string."""
        d = date.today().isoformat() + "T12:00:00Z"
        score = _freshness(d)
        assert 0.49 < score <= 0.5


class TestCategoryMatch:
    def test_full_match(self):
        assert _category_match(["cs.AI", "cs.LG"], ["cs.AI", "cs.LG"]) == 0.5

    def test_partial_match(self):
        score = _category_match(["cs.AI"], ["cs.AI", "cs.LG"])
        assert score == pytest.approx(0.25, abs=0.01)

    def test_no_match(self):
        assert _category_match(["cs.CV"], ["cs.AI", "cs.LG"]) == 0.0

    def test_empty_preferred_returns_zero(self):
        assert _category_match(["cs.AI"], []) == 0.0

    def test_empty_paper_categories(self):
        assert _category_match([], ["cs.AI"]) == 0.0

    def test_superset_capped_at_half(self):
        """Extra matching categories don't exceed 0.5."""
        score = _category_match(["cs.AI", "cs.LG", "cs.CL"], ["cs.AI"])
        assert score == 0.5


class TestComputeScore:
    def test_bounds(self):
        paper = {"published_date": date.today().isoformat(), "categories": ["cs.AI"]}
        s = compute_score(paper, ["cs.AI"])
        assert 0.0 <= s <= 1.0

    def test_high_score_for_fresh_matching_paper(self):
        paper = {"published_date": date.today().isoformat(), "categories": ["cs.AI", "cs.LG"]}
        s = compute_score(paper, ["cs.AI", "cs.LG"])
        assert s > 0.9

    def test_low_score_for_old_nonmatching_paper(self):
        paper = {"published_date": "2020-01-01", "categories": ["math.CO"]}
        s = compute_score(paper, ["cs.AI", "cs.LG"])
        assert s < 0.2

    def test_no_categories_key(self):
        """Paper without categories key should not raise."""
        paper = {"published_date": date.today().isoformat()}
        s = compute_score(paper, ["cs.AI"])
        assert 0.0 <= s <= 1.0

    def test_score_capped_at_one(self):
        """Score must never exceed 1.0."""
        paper = {"published_date": date.today().isoformat(), "categories": ["cs.AI"] * 100}
        s = compute_score(paper, ["cs.AI"])
        assert s <= 1.0
