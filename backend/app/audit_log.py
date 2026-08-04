"""Append compact, spreadsheet-friendly authentication audit rows."""
import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings

_lock = threading.Lock()
logger = logging.getLogger(__name__)
_fields = ["timestamp", "authenticated", "student_id", "reason", "similarity_score", "liveness_score", "detection_ms", "alignment_ms", "liveness_ms", "recognition_ms", "total_ms"]


def write_auth_audit(*, authenticated: bool, student_id: str | None, reason: str, similarity_score: float | None, liveness_score: float | None, timings: dict[str, float], total_ms: float) -> None:
    path = Path(settings.AUTH_AUDIT_LOG_PATH)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "authenticated": authenticated,
        "student_id": student_id or "", "reason": reason,
        "similarity_score": "" if similarity_score is None else round(similarity_score, 4),
        "liveness_score": "" if liveness_score is None else round(liveness_score, 4),
        "detection_ms": timings.get("detection", 0), "alignment_ms": timings.get("alignment", 0),
        "liveness_ms": timings.get("liveness", 0), "recognition_ms": timings.get("recognition", 0),
        "total_ms": round(total_ms, 2),
    }
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_fields)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
    except OSError:
        # Audit storage should be observable but must not take authentication down.
        logger.exception("authentication_audit_write_failed", extra={"event": "authentication_audit_write_failed", "path": str(path)})
