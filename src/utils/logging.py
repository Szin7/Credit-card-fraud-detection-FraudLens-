"""Structured JSON logging shared by every pipeline stage.

Plain `print()` calls don't compose with log aggregation, and a bespoke
logging dependency (structlog etc.) isn't justified for a project this size —
the stdlib `logging` module already supports custom formatters, so a thin
JSON formatter is all that's needed.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        context = getattr(record, "context", None)
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with the JSON formatter.

    Safe to call repeatedly (e.g. once per module) — handlers are only
    attached the first time a given logger name is requested.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, message: str | None = None, **context: Any) -> None:
    """Log a named pipeline event with structured context.

    Keep `context` to identifiers and counts (row counts, durations, file
    names) — never pass credentials, tokens, or full row contents.
    """
    logger.info(message or event, extra={"event": event, "context": context})
