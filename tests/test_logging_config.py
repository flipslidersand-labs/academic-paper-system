"""Tests for structured JSON logging configuration."""

import json
import logging

import pytest

from academic_paper.logging_config import _JsonFormatter, configure_logging


def _make_record(msg: str = "hello", level: int = logging.INFO, exc_info=None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    return record


def test_json_formatter_produces_valid_json():
    """_JsonFormatter.format() returns valid JSON."""
    fmt = _JsonFormatter()
    record = _make_record("test message")
    output = fmt.format(record)
    data = json.loads(output)
    assert data["message"] == "test message"
    assert data["level"] == "INFO"
    assert data["logger"] == "test.logger"
    assert "timestamp" in data


def test_json_formatter_includes_exc_info():
    """Exception info is included in JSON output."""
    fmt = _JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _make_record("error", exc_info=sys.exc_info())
    output = fmt.format(record)
    data = json.loads(output)
    assert "exc_info" in data
    assert "ValueError" in data["exc_info"]


def test_json_formatter_no_exc_info_key_when_clean():
    """exc_info key absent when record has no exception."""
    fmt = _JsonFormatter()
    record = _make_record("clean")
    data = json.loads(fmt.format(record))
    assert "exc_info" not in data


def test_configure_logging_json_sets_json_handler():
    """configure_logging(fmt='json') installs a _JsonFormatter handler."""
    configure_logging(level="DEBUG", fmt="json")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, _JsonFormatter)
    assert root.level == logging.DEBUG


def test_configure_logging_text_sets_plain_handler():
    """configure_logging(fmt='text') installs a plain Formatter."""
    configure_logging(level="WARNING", fmt="text")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert not isinstance(root.handlers[0].formatter, _JsonFormatter)
    assert root.level == logging.WARNING


def test_configure_logging_idempotent():
    """Calling configure_logging twice does not double-add handlers."""
    configure_logging(fmt="json")
    configure_logging(fmt="json")
    assert len(logging.getLogger().handlers) == 1


@pytest.fixture(autouse=True)
def reset_root_logger():
    """Restore root logger state after each test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.level = original_level
