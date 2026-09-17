"""Tests for scripts/cli_utils.py shared argparse validators (#345)."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cli_utils import MAX_RESULTS_CAP, check_date_order, iso_date, positive_int  # noqa: E402


class TestIsoDate:
    def test_empty_string_allowed_as_no_filter(self):
        assert iso_date("") == ""

    def test_valid_iso_date_returned_unchanged(self):
        assert iso_date("2025-08-01") == "2025-08-01"

    def test_invalid_format_raises_argument_type_error(self):
        with pytest.raises(argparse.ArgumentTypeError):
            iso_date("2025/08/01")

    def test_non_date_string_raises_argument_type_error(self):
        with pytest.raises(argparse.ArgumentTypeError):
            iso_date("not-a-date")


class TestPositiveInt:
    def test_valid_int_in_range_returned(self):
        assert positive_int("10") == 10

    def test_non_integer_raises_argument_type_error(self):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("abc")

    def test_zero_raises_argument_type_error(self):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("0")

    def test_negative_raises_argument_type_error(self):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("-5")

    def test_exceeding_max_results_cap_raises_argument_type_error(self):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(str(MAX_RESULTS_CAP + 1))

    def test_max_results_cap_itself_is_accepted(self):
        assert positive_int(str(MAX_RESULTS_CAP)) == MAX_RESULTS_CAP


class TestCheckDateOrder:
    def _parser(self):
        return argparse.ArgumentParser()

    def test_from_before_until_does_not_error(self, capsys):
        parser = self._parser()
        check_date_order(parser, "2025-01-01", "2025-12-31")
        # no SystemExit means success; nothing printed to stderr
        assert capsys.readouterr().err == ""

    def test_from_after_until_calls_parser_error(self):
        parser = self._parser()
        with pytest.raises(SystemExit):
            check_date_order(parser, "2025-12-31", "2025-01-01")

    def test_missing_from_date_skips_check(self):
        parser = self._parser()
        check_date_order(parser, "", "2025-01-01")

    def test_missing_until_date_skips_check(self):
        parser = self._parser()
        check_date_order(parser, "2025-01-01", "")
