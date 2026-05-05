"""In-memory job registry for long-running admin/ingest tasks.

Design:

* ``submit(kind, key, factory, params)`` schedules an ``asyncio.Task`` only when
  no other job with the same ``key`` is currently ``pending``/``running``.
  Otherwise, the *existing* job is returned (single-flight). Callers can use
  :attr:`Job.was_already_running` to decide whether to respond with 202 or 409.
* The registry is in-memory; jobs survive only as long as the FastAPI process.
  History is bounded by ``max_history`` (oldest finished jobs are evicted).
* Cancellation calls :meth:`asyncio.Task.cancel`. Inside ``factory`` coroutines
  that wrap synchronous work via ``asyncio.to_thread``, cancellation is
  best-effort: the running thread completes, then the task transitions to
  ``cancelled``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

CoroFactory = Callable[[], Awaitable[dict[str, Any] | None]]


class JobStatus(StrEnum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


_TERMINAL = {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled}


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Job:
    id: str
    kind: str
    key: str
    params: dict[str, Any]
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)

    was_already_running: bool = False
    """Set on the snapshot returned by :meth:`JobRegistry.submit` when the
    submitter hit an existing in-flight job (single-flight rejection)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "key": self.key,
            "params": self.params,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
        }


class JobRegistry:
    """Tracks ``Job`` snapshots and their backing ``asyncio.Task`` objects."""

    def __init__(self, max_history: int = 200) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._key_to_id: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._max_history = max_history

    async def submit(
        self,
        kind: str,
        key: str,
        factory: CoroFactory,
        params: dict[str, Any] | None = None,
    ) -> Job:
        """Schedule ``factory`` as a background task, single-flighted by ``key``.

        If a job with the same ``key`` is already active, that existing job is
        returned with :attr:`Job.was_already_running` set to ``True``.
        """
        async with self._lock:
            existing_id = self._key_to_id.get(key)
            if existing_id is not None:
                existing = self._jobs.get(existing_id)
                if existing is not None and existing.status not in _TERMINAL:
                    snap = self._snapshot(existing)
                    snap.was_already_running = True
                    return snap

            job = Job(
                id=uuid.uuid4().hex,
                kind=kind,
                key=key,
                params=dict(params or {}),
                status=JobStatus.pending,
                created_at=_now(),
            )
            self._jobs[job.id] = job
            self._key_to_id[key] = job.id

            task = asyncio.create_task(self._run(job, factory), name=f"job:{kind}:{job.id}")
            self._tasks[job.id] = task
            self._evict_finished()
            return self._snapshot(job)

    async def _run(self, job: Job, factory: CoroFactory) -> None:
        job.status = JobStatus.running
        job.started_at = _now()
        try:
            result = await factory()
            job.result = result if isinstance(result, dict) else ({"value": result} if result is not None else None)
            job.status = JobStatus.succeeded
        except asyncio.CancelledError:
            job.status = JobStatus.cancelled
            job.error = "cancelled"
            logger.info("job %s (%s) cancelled", job.id, job.kind)
            raise
        except Exception as exc:
            job.status = JobStatus.failed
            job.error = f"{type(exc).__name__}: {exc}"
            logger.exception("job %s (%s) failed: %s", job.id, job.kind, exc)
        finally:
            job.finished_at = _now()
            current = self._key_to_id.get(job.key)
            if current == job.id:
                self._key_to_id.pop(job.key, None)
            self._tasks.pop(job.id, None)

    def _snapshot(self, job: Job) -> Job:
        snap = Job(
            id=job.id,
            kind=job.kind,
            key=job.key,
            params=dict(job.params),
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            result=dict(job.result) if isinstance(job.result, dict) else job.result,
            error=job.error,
            progress=dict(job.progress),
        )
        return snap

    def get(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        return self._snapshot(job) if job is not None else None

    def list(self, status: JobStatus | None = None, limit: int = 50) -> list[Job]:
        items = list(self._jobs.values())
        items.sort(key=lambda j: j.created_at, reverse=True)
        if status is not None:
            items = [j for j in items if j.status == status]
        return [self._snapshot(j) for j in items[:limit]]

    async def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in _TERMINAL:
            return self._snapshot(job)
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        return self._snapshot(job)

    async def shutdown(self) -> None:
        """Cancel all active tasks and wait for them to finish."""
        tasks = [t for t in self._tasks.values() if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _evict_finished(self) -> None:
        if len(self._jobs) <= self._max_history:
            return
        for jid in list(self._jobs.keys()):
            if len(self._jobs) <= self._max_history:
                break
            j = self._jobs[jid]
            if j.status in _TERMINAL:
                self._jobs.pop(jid, None)
