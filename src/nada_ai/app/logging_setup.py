"""Structured logging for the FastAPI process (stdlib only, no new dependency).

Two formats:

- ``text`` (default) — human-readable, request-ID-prefixed. Good for local dev.
- ``json`` — one JSON object per line (``ts``, ``level``, ``logger``,
  ``message``, ``request_id``, plus ``exc_info`` when present). Set
  ``NADA_LOG_FORMAT=json`` in any environment with a log aggregator that
  parses JSON (CloudWatch, Loki, Datadog, etc.).

Both formats inject the current request ID (``app/request_context.py``) into
every log record so a single HTTP request's log lines can be grepped/filtered
together, including lines emitted from ``asyncio.to_thread`` callees.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from nada_ai.app.request_context import get_request_id


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()  # type: ignore[attr-defined]
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(fmt: str = "text", level: str = "INFO") -> None:
    """Install one stdout handler on the root logger. Idempotent."""
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    handler.addFilter(_RequestIdFilter())

    root.handlers.clear()
    root.addHandler(handler)
