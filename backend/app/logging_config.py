"""Small structured-logging setup shared by the API and ML pipeline."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit machine-readable logs while retaining normal ``logging`` APIs."""

    _standard_fields = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelname", "levelno", "lineno", "message",
            "module", "msecs", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "taskName", "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in self._standard_fields and not key.startswith("_"): 
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure one stdout handler exactly once for application logs."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for handler in root.handlers:
        if getattr(handler, "_face_auth_structured", False):
            return

    handler = logging.StreamHandler(sys.stdout)
    handler._face_auth_structured = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
