"""Structured logging configuration for academic paper system."""

import json
import logging
import sys


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            data["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(data, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure root logger with JSON or plain-text output.

    Args:
        level: Root log level string (e.g. "INFO", "DEBUG").
        fmt: "json" for structured output (production default) or "text" for human-readable.
    """
    root = logging.getLogger()
    # Avoid adding duplicate handlers if called more than once
    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        )

    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
