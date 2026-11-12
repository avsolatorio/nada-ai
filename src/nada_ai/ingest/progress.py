"""Per-``catalog_type`` ingest checkpoints and live job-progress reporting.

A full-catalog ``index_from_catalog`` run is a single long synchronous loop
(see ``ingest/pipeline.py::iter_langdoc_records``) executed on a worker thread
via ``asyncio.to_thread``. Two things were previously invisible for the
duration of that loop:

1. **Progress** — how far along the run is, right now.
2. **Resumability** — if the run is stopped (or crashes) partway through,
   which studies are already done, so a follow-up run doesn't start from zero.

``IngestProgressTracker`` addresses both: it is stepped once per study (by its
NADA internal id, ``sid``) as the pipeline processes it, forwards a live
snapshot to the job (via ``on_update``, normally ``JobRegistry.set_progress``
bound to the job's id), and periodically persists a checkpoint file so a
resumed run can skip studies already completed.

Checkpoints are plain JSON, one file per ``catalog_type``, written atomically
(temp file + ``os.replace``) — no database or new service dependency. This is
deliberately not a database table: writes come from exactly one job at a time
(``JobRegistry`` single-flights ingest by ``catalog_type``), so there is never
a concurrent-writer problem to solve, and an operator can just ``cat`` the
file to see what a stuck run had done.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = "config/ingest_checkpoints"


def _checkpoint_dir(settings: Settings) -> Path:
    return Path(settings.ingest_checkpoint_dir or DEFAULT_CHECKPOINT_DIR)


def checkpoint_path(settings: Settings, catalog_type: str) -> Path:
    return _checkpoint_dir(settings) / f"{catalog_type}.json"


@dataclass
class IngestCheckpoint:
    catalog_type: str
    total: int = 0
    completed_sids: set[int] = field(default_factory=set)
    failed: dict[int, str] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_type": self.catalog_type,
            "total": self.total,
            "completed_sids": sorted(self.completed_sids),
            "failed": {str(sid): error for sid, error in self.failed.items()},
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestCheckpoint:
        return cls(
            catalog_type=str(data.get("catalog_type") or ""),
            total=int(data.get("total") or 0),
            completed_sids={int(sid) for sid in data.get("completed_sids") or []},
            failed={int(sid): str(error) for sid, error in (data.get("failed") or {}).items()},
            updated_at=float(data.get("updated_at") or 0.0),
        )


def load_checkpoint(settings: Settings, catalog_type: str) -> IngestCheckpoint | None:
    path = checkpoint_path(settings, catalog_type)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return IngestCheckpoint.from_dict(data)
    except Exception as e:  # noqa: BLE001 - a corrupt/partial checkpoint must never block a run
        logger.warning("Ignoring unreadable checkpoint %s: %s", path, e)
        return None


def save_checkpoint(settings: Settings, checkpoint: IngestCheckpoint) -> None:
    checkpoint.updated_at = time.time()
    path = checkpoint_path(settings, checkpoint.catalog_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-to-temp-then-rename: a crash mid-write never leaves a half-written,
    # unparseable checkpoint behind (os.replace is atomic on POSIX and Windows).
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(checkpoint.to_dict(), f)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def clear_checkpoint(settings: Settings, catalog_type: str) -> None:
    path = checkpoint_path(settings, catalog_type)
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Failed to remove checkpoint %s: %s", path, e)


class IngestProgressTracker:
    """Steps through one full-catalog ingest run; reports progress, persists checkpoints.

    ``on_update``, if given, is called synchronously (from whichever thread
    calls :meth:`mark`) with a small JSON-safe snapshot dict — normally bound
    to ``JobRegistry.set_progress(job_id, ...)`` so ``GET /jobs/{id}`` reflects
    it immediately, no polling delay beyond the client's own refresh interval.
    """

    def __init__(
        self,
        settings: Settings,
        catalog_type: str,
        total: int,
        *,
        checkpoint: IngestCheckpoint | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        save_every: int = 20,
    ) -> None:
        self._settings = settings
        self.catalog_type = catalog_type
        self.total = total
        self.checkpoint = checkpoint or IngestCheckpoint(catalog_type=catalog_type, total=total)
        self.checkpoint.total = total
        self._on_update = on_update
        self._save_every = max(1, save_every)
        self._since_save = 0
        # Pre-seed counts from a resumed checkpoint so progress reflects the
        # whole run, not just the segment still to do.
        self.processed = len(self.checkpoint.completed_sids) + len(self.checkpoint.failed)
        self.failed_count = len(self.checkpoint.failed)
        # Studies whose chunks are handed to the writer but not all confirmed written yet, by sid (every chunk
        # carries ``metadata.sid``, set from the study itself): [chunks left, first write error]. A study is
        # marked (and so checkpointed) only once all of its chunks are confirmed.
        self._unconfirmed: dict[int, list[Any]] = {}

    def expect(self, sid: int, chunks: int) -> None:
        """Study ``sid``'s ``chunks`` documents are about to be handed to the writer; it is marked once the writer
        has :meth:`confirm`-ed every one. Until then it is not in the checkpoint, so a resume after a cancel or crash
        retries it instead of skipping chunks that were never written."""
        entry = self._unconfirmed.setdefault(sid, [0, None])
        entry[0] += chunks

    def confirm(self, sid: int, ok: bool, error: str | None = None) -> None:
        """The writer's outcome for one of study ``sid``'s chunks: the backend accepted it (``ok``) or rejected it."""
        entry = self._unconfirmed.get(sid)
        if entry is None:
            return
        if not ok and entry[1] is None:
            entry[1] = f"write failed: {error or 'unknown error'}"
        entry[0] -= 1
        if entry[0] == 0:
            del self._unconfirmed[sid]
            self.mark(sid, ok=entry[1] is None, error=entry[1])

    def mark(self, sid: int, ok: bool, error: str | None = None) -> None:
        self.processed += 1
        if ok:
            self.checkpoint.completed_sids.add(sid)
            self.checkpoint.failed.pop(sid, None)
        else:
            self.failed_count += 1
            self.checkpoint.failed[sid] = (error or "unknown error")[:500]

        if self._on_update is not None:
            percent = round(100 * self.processed / self.total, 1) if self.total else None
            self._on_update(
                {
                    "processed": self.processed,
                    "total": self.total,
                    "failed": self.failed_count,
                    "current_sid": sid,
                    "percent": percent,
                }
            )

        self._since_save += 1
        if self._since_save >= self._save_every or self.processed >= self.total:
            self._since_save = 0
            save_checkpoint(self._settings, self.checkpoint)

    def finalize(self, *, completed: bool) -> None:
        """Call once the run loop exits. ``completed=True`` clears the checkpoint
        (nothing left to resume); ``False`` (cancelled/crashed) leaves the latest
        state on disk for a future ``resume=True`` run."""
        if completed:
            clear_checkpoint(self._settings, self.catalog_type)
        else:
            save_checkpoint(self._settings, self.checkpoint)


class CancelToken:
    """Thin wrapper around ``threading.Event`` checked cooperatively inside the
    synchronous ingest loop (see ``pipeline.iter_langdoc_records``).

    ``asyncio.Task.cancel()`` alone does not stop work already dispatched to a
    worker thread via ``asyncio.to_thread`` — the thread runs to completion
    regardless, so "Stop" previously kept burning CPU/RAM until the whole
    catalog finished. Checking this token once per study makes Stop actually
    stop within one document's processing time.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()
