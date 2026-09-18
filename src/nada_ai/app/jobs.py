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

    def to_dict(self, *, include_sensitive: bool = False) -> dict[str, Any]:
        """Serialise to a dict safe for external API consumers.

        ``error`` is redacted by default to avoid leaking backend details
        (connection strings, hostnames, credentials) in exception messages.
        Pass ``include_sensitive=True`` only for internal server-side logging.
        """
        error: str | None
        if self.error is None:
            error = None
        elif include_sensitive:
            error = self.error
        else:
            # Expose the exception type but not the potentially sensitive message.
            error = self.error.split(":")[0] if ":" in self.error else self.error
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
            "error": error,
            "progress": self.progress,
        }


class JobRegistry:
    """Tracks ``Job`` snapshots and their backing ``asyncio.Task`` objects."""

    def __init__(self, max_history: int = 200) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._key_to_id: dict[str, str] = {}
        self._cancel_tokens: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._max_history = max_history

    async def submit(
        self,
        kind: str,
        key: str,
        factory: CoroFactory,
        params: dict[str, Any] | None = None,
        job_id: str | None = None,
        cancel_token: Any | None = None,
    ) -> Job:
        """Schedule ``factory`` as a background task, single-flighted by ``key``.

        If a job with the same ``key`` is already active, that existing job is
        returned with :attr:`Job.was_already_running` set to ``True`` — in that
        case any ``job_id``/``cancel_token`` the caller pre-created are simply
        never registered (nothing to clean up; they're just discarded).

        ``job_id``, if given, is used instead of a freshly generated id — lets a
        caller create a :class:`~nada_ai.ingest.progress.CancelToken` and start
        reporting progress via :meth:`set_progress` for this job's id *before*
        the job is known to actually run (see ``app/admin.py``'s full-catalog
        ingest routes). ``cancel_token`` is stashed so :meth:`cancel` can signal
        it in addition to cancelling the ``asyncio.Task`` (see its docstring for
        why both are needed).
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
                id=job_id or uuid.uuid4().hex,
                kind=kind,
                key=key,
                params=dict(params or {}),
                status=JobStatus.pending,
                created_at=_now(),
            )
            self._jobs[job.id] = job
            self._key_to_id[key] = job.id
            if cancel_token is not None:
                self._cancel_tokens[job.id] = cancel_token

            task = asyncio.create_task(self._run(job, factory), name=f"job:{kind}:{job.id}")
            self._tasks[job.id] = task
            self._evict_finished()
            return self._snapshot(job)

    def set_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        """Best-effort progress update, safe to call from a worker thread.

        No lock: this is a plain attribute assignment on a dict CPython already
        treats atomically, and progress is purely informational (unlike
        status/result transitions in :meth:`_run`, nothing downstream depends
        on ordering between two progress updates).
        """
        job = self._jobs.get(job_id)
        if job is not None:
            job.progress = dict(progress)

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
            self._cancel_tokens.pop(job.id, None)

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
        """Cancel a job.

        Two independent signals, because one alone isn't enough for a job
        whose actual work runs in a worker thread via ``asyncio.to_thread``:
        ``task.cancel()`` only stops *this* coroutine from awaiting the
        thread's result — the thread itself keeps running the synchronous
        ingest loop to completion regardless (Python cannot forcibly kill a
        thread). Setting the job's ``cancel_token`` (a plain
        ``threading.Event`` under the hood — see ``ingest/progress.CancelToken``)
        is what the loop itself checks once per document, so cancelling
        actually stops CPU/RAM usage promptly instead of silently finishing
        the whole run in the background.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in _TERMINAL:
            return self._snapshot(job)
        token = self._cancel_tokens.get(job_id)
        if token is not None:
            token.set()
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
