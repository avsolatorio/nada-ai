"""Append-only JSONL audit trail for admin/webhook mutating actions.

Each entry captures who did what, to which target, and the outcome. This is
deliberately a flat file with atomic appends guarded by an ``asyncio.Lock``
rather than a database — consistent with the rest of the app's file-based
config stores (``search/dynamic_filters.py``, ``app/keys_store.py``).

Writing an audit entry never raises: a logging failure should not fail the
admin action it is describing. Failures are logged instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nada_ai.settings import Settings

if TYPE_CHECKING:
    from nada_ai.app.auth import Principal
    from nada_ai.app.state import AppState

logger = logging.getLogger(__name__)

_DEFAULT_AUDIT_LOG_PATH = Path(__file__).resolve().parents[3] / "config" / "audit.log"


def _resolve_audit_path(settings: Settings | None) -> Path:
    if settings is not None and settings.audit_log_path:
        return Path(settings.audit_log_path)
    return _DEFAULT_AUDIT_LOG_PATH


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


async def audit_log(
    s: "AppState",
    principal: "Principal",
    *,
    action: str,
    target: str | None = None,
    status: str = "ok",
    detail: str | None = None,
) -> None:
    """Append one audit entry. Best-effort — failures are logged, never raised."""
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "principal_id": principal.id,
        "principal_name": principal.name,
        "role": principal.role.value,
        "action": action,
        "target": target,
        "status": status,
        "detail": detail,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    path = _resolve_audit_path(s.settings)
    try:
        async with s.audit_lock:
            await asyncio.to_thread(_append_line, path, line)
    except Exception:
        logger.exception("failed to write audit log entry: %s", entry)


def read_audit_log(
    settings: Settings | None,
    *,
    limit: int = 100,
    action: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent audit entries, newest first."""
    path = _resolve_audit_path(settings)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries: list[dict[str, Any]] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if action is not None and entry.get("action") != action:
            continue
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries
