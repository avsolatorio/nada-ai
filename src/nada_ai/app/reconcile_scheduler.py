"""The scheduler: works through NADA's search-index change queue, one item at a time.

NADA decides what the index needs (every catalog change, and what a sync queues) as rows of its change queue; this
loop is the only worker, so it never competes with itself:

1. fetch a page of pending rows (catalog changes first), process them one by one, ack them in small batches;
2. a full page means there is more: the next page is fetched straight away (the request gate still spaces requests);
3. a short page means the queue is caught up: wait ``reconcile_search_index_interval_seconds`` (or until Process
   pending wakes it), then look again.

Every request to NADA goes through the request gate (``nada.gate``): one at a time, a minimum gap, retries with
backoff. What the gate could not fix decides what happens next:

- an item failed while NADA answers its status: the item is the problem, it is acked ``failed`` and skipped;
- NADA does not answer: nothing more is started, the rows stay pending, and the scheduler pauses (a minute, doubling
  up to ``unavailable_backoff_max_seconds``; longer when NADA's ``Retry-After`` asks for longer);
- a firewall in front of NADA blocks the requests: pause for ``firewall_pause_seconds``, without retrying;
- NADA does not track changes for this deployment (or tracks another nada-ai's index): pause, look again every
  ``reconcile_search_index_not_tracking_interval_seconds`` (a first "another index" refusal: after a minute, since
  both sides cache the generation that long after the study index is recreated);
- ``ack_failure_limit`` acks in a row fail while NADA answers: results are not being recorded, so nothing more is
  started (the work would be redone forever); the unsent acks are kept and sent first when it resumes.

It can be paused (``POST /admin/scheduler/pause``): it finishes the item in hand, then starts nothing until resumed.
The pause is a control file next to the lock, so it survives a restart and any process can set it.

Only one process runs it: the first to lock ``reconcile_search_index_lock_path``. It writes its state next to the lock
on every heartbeat, so any process can report it (``GET /admin/scheduler/status``). A crash restarts the loop after a
growing wait (the watchdog); a heartbeat that stops moving shows as ``stuck``.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from ai4data.discovery.config import metadata_catalog

from nada_ai.app.state import AppState, ensure_embedding_initialized
from nada_ai.ingest.search_index_sync import (
    AckRefused,
    SearchIndexQueueItem,
    ack_batch,
    fetch_studies_by_sids,
    get_status,
    list_queue,
    process_item,
)
from nada_ai.nada import gate as nada_gate
from nada_ai.search.backend.opensearch.generation import forget_index_generation
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

#: Full study records (metadata and resources, often large) fetched per request.
FULL_RECORDS_PER_REQUEST = 20
#: Acks are sent once this many are waiting (and at the end of every page).
ACK_FLUSH_SIZE = 10
#: A heartbeat older than this while processing one item counts as stuck.
STUCK_PROCESSING_SECONDS = 3600
#: Slack on top of a planned wait before the scheduler counts as stuck.
STUCK_GRACE_SECONDS = 300
#: How often a paused scheduler looks at the control file again (a resume from another process).
PAUSED_CHECK_SECONDS = 10
#: After a first "another nada-ai's index" refusal, look again this soon: right after the study index is recreated,
#: nada-ai and NADA each go on using the generation they cached (for up to a minute) before they agree again.
GENERATION_RECHECK_SECONDS = 60
#: Pause when NADA refuses NADA-AI's key: someone has to fix the key (Process pending tries again sooner).
ACCESS_DENIED_PAUSE_SECONDS = 1800
#: Pause when ``ack_failure_limit`` acks fail in a row; doubled each time it trips again, up to the maximum.
ACK_FAILURE_PAUSE_SECONDS = 1800
ACK_FAILURE_PAUSE_MAX_SECONDS = 4 * 3600
#: The longest Retry-After the scheduler waits out (a far-off date in it must not park the scheduler for good).
RETRY_AFTER_PAUSE_MAX_SECONDS = 24 * 3600
_REFUSAL_REASONS = {
    "INDEX_ENGINE_NOT_NADA_AI": "not_tracking_for_nada_ai",
    "INDEX_GENERATION_MISMATCH": "other_nada_ai",
}


def _now() -> float:
    return time.time()


def _int(value: float | None) -> int | None:
    return int(value) if value is not None else None


@dataclass
class SchedulerState:
    """What the scheduler is doing, for ``GET /admin/scheduler/status`` and ``/health``."""

    enabled: bool = False
    running_here: bool = False
    #: starting | idle | processing | paused | not_tracking | disabled | not_configured | crashed | running_elsewhere
    state: str = "starting"
    pause: dict[str, Any] | None = None
    current: dict[str, Any] | None = None
    pass_: dict[str, Any] = field(default_factory=dict)
    last_pass: dict[str, Any] | None = None
    unsent_acks: list[dict[str, Any]] = field(default_factory=list)
    consecutive_ack_failures: int = 0
    unavailable_streak: int = 0
    ack_pause_streak: int = 0
    last_poll_at: float | None = None
    next_poll_at: float | None = None
    heartbeat_at: float | None = None
    last_error: dict[str, Any] | None = None
    crash: dict[str, Any] | None = None
    restarts: int = 0
    interval_seconds: int = 0
    owner: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state": self.state,
            "pause": self.pause,
            "current": self.current,
            "pass": self.pass_ or None,
            "last_pass": self.last_pass,
            "acks": {"unsent": len(self.unsent_acks), "consecutive_failures": self.consecutive_ack_failures},
            "last_poll_at": _int(self.last_poll_at),
            "next_poll_at": _int(self.next_poll_at),
            "heartbeat_at": _int(self.heartbeat_at),
            "last_error": self.last_error,
            "crash": self.crash,
            "restarts": self.restarts,
            "interval_seconds": self.interval_seconds,
            "owner": self.owner,
        }


def with_health(snapshot: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    """``snapshot`` with ``state: stuck`` when its heartbeat stopped moving."""
    now = _now() if now is None else now
    heartbeat = snapshot.get("heartbeat_at")
    state = snapshot.get("state")
    if heartbeat is None or state not in ("idle", "processing", "paused", "not_tracking"):
        return snapshot
    if state == "processing":
        late = now - heartbeat > STUCK_PROCESSING_SECONDS
    else:
        wake = snapshot.get("next_poll_at") or (snapshot.get("pause") or {}).get("resume_at") or heartbeat
        late = now > wake + STUCK_GRACE_SECONDS
    return {**snapshot, "state": "stuck"} if late else snapshot


class Scheduler:
    def __init__(self, s: AppState) -> None:
        self.s = s
        self.settings: Settings = s.settings
        self.st = SchedulerState(
            enabled=self.settings.reconcile_search_index_enabled,
            interval_seconds=self.settings.reconcile_search_index_interval_seconds,
        )
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._one_pass: asyncio.Task | None = None
        self._lock_file: Any = None
        self._lock_path = Path(self.settings.reconcile_search_index_lock_path or "config/search_index_scheduler.lock")
        self._last_logged_pause: str | None = None
        self._generation_rechecked = False

    # -- start / stop -----------------------------------------------------------------------------------------------

    def nada_url_configured(self) -> bool:
        return bool(self.settings.search_index_base_url or metadata_catalog.url)

    def start(self) -> None:
        """Start the loop in this process when it is enabled, NADA's URL is set and the lock is free."""
        if not self.st.enabled:
            self.st.state = "disabled"
            return
        if not self.nada_url_configured():
            self.st.state = "not_configured"
            logger.warning(
                "search-index scheduler not started: no NADA URL (NADA_SEARCH_INDEX_BASE_URL / AI4DATA_METADATA_CATALOG_URL)"
            )
            return
        if not self._take_lock():
            self.st.state = "running_elsewhere"
            logger.info("search-index scheduler runs in another process (lock %s held)", self._lock_path)
            return
        self.st.running_here = True
        self.st.owner = {"pid": os.getpid(), "host": socket.gethostname(), "started_at": int(_now())}
        self._task = asyncio.create_task(self._watchdog(), name="search-index-scheduler")

    async def stop(self) -> None:
        for task in (self._task, self._one_pass):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._release_lock()

    def _take_lock(self) -> bool:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._lock_path, "a+")  # noqa: SIM115 - held open for the life of the process
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._lock_file = handle
        return True

    def _release_lock(self) -> None:
        if self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_file.close()
                self._lock_file = None

    @property
    def _state_path(self) -> Path:
        return self._lock_path.with_suffix(".state.json")

    @property
    def _control_path(self) -> Path:
        return self._lock_path.with_suffix(".control.json")

    # -- pause / resume ---------------------------------------------------------------------------------------------

    def paused_since(self) -> int | None:
        """When the scheduler was paused, or None when it is not."""
        try:
            control = json.loads(self._control_path.read_text())
        except (OSError, ValueError):
            return None
        return int(control.get("since") or 0) if control.get("paused") else None

    def pause(self) -> None:
        if self.paused_since() is None:
            self._control_path.parent.mkdir(parents=True, exist_ok=True)
            self._control_path.write_text(json.dumps({"paused": True, "since": int(_now())}))

    def resume(self) -> None:
        self._control_path.unlink(missing_ok=True)
        self._wake.set()

    # -- status -----------------------------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """This process's scheduler, or (when another process runs it) what that one last wrote, its request gate's
        figures included: the gate is per process, and this one's sees none of the scheduler's requests."""
        if self.st.state == "running_elsewhere":
            try:
                data = json.loads(self._state_path.read_text())
            except (OSError, ValueError):
                data = self.st.snapshot()
            data.setdefault("requests", None)
        else:
            data = {**self.st.snapshot(), "requests": self._requests()}
        since = self.paused_since()
        if since is not None:
            # paused by hand: shown from the moment it is asked, while the item in hand is being finished too (a
            # scheduler that is off or not configured keeps that state; the pause is reported with it)
            data["pause"] = {"reason": "manual", "since": since, "resume_at": None, "message": "Indexing is paused"}
            if data.get("state") not in ("disabled", "not_configured"):
                data["state"] = "paused"
        data = with_health(data)
        data["running_here"] = self.st.running_here
        return data

    def _requests(self) -> dict[str, Any]:
        """This process's request gate: its gap between requests, its requests of the last hour, its last failure."""
        gate = nada_gate.get_gate()
        interval = gate.interval_seconds() if gate else self.settings.request_min_interval_ms / 1000
        return {
            "min_interval_ms": int(interval * 1000),
            "last_hour": gate.requests_last_hour() if gate else 0,
            "last_trouble": gate.last_trouble.as_dict() if gate and gate.last_trouble else None,
        }

    def _heartbeat(self) -> None:
        self.st.heartbeat_at = _now()
        if self.st.running_here:
            try:
                self._state_path.write_text(json.dumps({**self.st.snapshot(), "requests": self._requests()}))
            except OSError as e:
                logger.debug("could not write the scheduler state file: %s", e)

    # -- waking -----------------------------------------------------------------------------------------------------

    def wake(self) -> dict[str, Any]:
        """Process pending: start on the queue now. When the loop does not run here, one pass is run instead. Nothing
        while paused (resume is the way back)."""
        if self.paused_since() is not None:
            return {"started": False, "state": "paused"}
        if self._task is not None and not self._task.done():
            self._wake.set()
            return {"started": True, "state": self.st.state}
        if self.st.state == "running_elsewhere":
            return {"started": False, "state": "running_elsewhere"}
        if self._one_pass is not None and not self._one_pass.done():
            return {"started": False, "state": "processing"}
        self._one_pass = asyncio.create_task(self._drain(), name="search-index-one-pass")
        return {"started": True, "state": "processing"}

    async def _wait(self, seconds: float) -> None:
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=max(0.0, seconds))
        except TimeoutError:
            pass
        self._wake.clear()

    # -- pausing ----------------------------------------------------------------------------------------------------

    def _pause(self, reason: str, message: str, seconds: float, *, state: str = "paused") -> None:
        now = _now()
        self.st.state = state
        self.st.current = None
        self.st.pause = {"reason": reason, "since": int(now), "resume_at": int(now + seconds), "message": message}
        key = f"{reason}:{message}"
        if key != self._last_logged_pause:
            logger.error("search-index scheduler paused (%s): %s — next try in %ss", reason, message, int(seconds))
            self._last_logged_pause = key

    def _pause_unavailable(self, message: str) -> None:
        if self.st.pause is not None and self.st.pause["reason"] == "nada_unavailable":
            return  # already paused for it (the acks of the same page failing too): one pause, not a longer one
        delay = min(60 * (2**self.st.unavailable_streak), self.settings.unavailable_backoff_max_seconds)
        self.st.unavailable_streak += 1
        gate = nada_gate.get_gate()
        asked = min(gate.not_before - _now(), RETRY_AFTER_PAUSE_MAX_SECONDS) if gate is not None else 0
        if asked > delay:
            # NADA (or what is in front of it) said when to come back: not before then
            delay = asked
            message = f"{message} (NADA asked to wait {int(asked)}s)"
        self._pause("nada_unavailable", message, delay)

    def _pause_for(self, error: Exception) -> None:
        if isinstance(error, nada_gate.NadaFirewallBlocked):
            ray = f" (Ray ID {error.cf_ray})" if error.cf_ray else ""
            self._pause("firewall", f"{error}{ray}", self.settings.firewall_pause_seconds)
        else:
            self._pause_unavailable(str(error))

    async def _stop_for(self, error: Exception, started: float) -> None:
        """Pause after a request the scheduler cannot go on without (a queue page, a batch of records) failed:
        for NADA's trouble (a firewall, or NADA not answering), for NADA refusing the key, or for an error of the
        request itself (an answer it cannot read, a bug), which waiting out an outage would not fix."""
        gate = nada_gate.get_gate()
        trouble = isinstance(error, nada_gate.NadaTrouble) or (
            gate is not None and gate.trouble_since(started) is not None
        )
        if isinstance(error, nada_gate.NadaFirewallBlocked) or (trouble and not await self._nada_answers()):
            self._pause_for(error)
        elif isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 403:
            self._pause("access_denied", "NADA refuses NADA-AI's key (ACCESS_DENIED)", ACCESS_DENIED_PAUSE_SECONDS)
        else:
            logger.error("search-index scheduler request failed: %s", error, exc_info=error)
            message = f"{type(error).__name__}: {error}"[:500]
            self.st.last_error = {"at": int(_now()), "kind": "error", "message": message}
            self._pause("error", message, self.settings.reconcile_search_index_interval_seconds)

    def _not_tracking(self, reason: str | None) -> None:
        seconds = self.settings.reconcile_search_index_not_tracking_interval_seconds
        if reason == "other_nada_ai":
            # the generation sent may be a cached one from before the study index was recreated: read it again, and
            # look again soon once (NADA may still be on its cached one too); refused again, it is another index
            forget_index_generation()
            if not self._generation_rechecked:
                self._generation_rechecked = True
                seconds = GENERATION_RECHECK_SECONDS
        texts = {
            "not_tracking_for_nada_ai": "NADA does not track changes for NADA-AI (its index kept in sync is not NADA-AI)",
            "other_nada_ai": "NADA tracks another NADA-AI index than this one",
        }
        self._pause(
            "other_nada_ai" if reason == "other_nada_ai" else "not_tracking",
            texts.get(reason or "", "NADA does not track changes for this deployment"),
            seconds,
            state="not_tracking",
        )

    async def _nada_answers(self) -> bool:
        try:
            await asyncio.to_thread(get_status, self.settings)
            return True
        except Exception as e:  # noqa: BLE001 - the probe only asks whether NADA answers
            logger.warning("NADA does not answer its status: %s", e)
            return False

    # -- the loop ---------------------------------------------------------------------------------------------------

    async def _watchdog(self) -> None:
        while True:
            try:
                await self._loop()
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - restarted below, and shown in the status
                logger.exception("search-index scheduler crashed: %s", e)
                self.st.restarts += 1
                self.st.state = "crashed"
                self.st.crash = {"at": int(_now()), "message": f"{type(e).__name__}: {e}"[:500]}
                self._heartbeat()
                await asyncio.sleep(min(60 * (2 ** (self.st.restarts - 1)), 1800))

    async def _loop(self) -> None:
        logger.info("search-index scheduler started (idle check every %ss)", self.st.interval_seconds)
        while True:
            self._heartbeat()
            if self.paused_since() is not None:
                self.st.state = "paused"
                self.st.current = None
                await self._wait(PAUSED_CHECK_SECONDS)
                continue
            if self.st.pause is not None:
                await self._wait(max(0.0, self.st.pause["resume_at"] - _now()))
                self.st.pause = None
                self.st.state = "idle"
            if self.st.unsent_acks and not await self._send_acks(list(self.st.unsent_acks), set(), resend=True):
                continue
            await self._drain()
            if self.st.pause is None:
                self.st.state = "idle"
                self.st.current = None
                self.st.next_poll_at = _now() + self.st.interval_seconds
                self._heartbeat()
                await self._wait(self.st.interval_seconds)
                self.st.next_poll_at = None

    async def _drain(self) -> None:
        """Work through the queue until a page comes back short (or something pauses it)."""
        limit = self.settings.reconcile_search_index_batch_limit
        exclude: set[int] = set()
        # acked: what NADA has taken off its queue (processed items wait for their ack in batches)
        self.st.pass_ = {"started_at": int(_now()), "processed": 0, "acked": 0, "indexed": 0, "deleted": 0, "failed": 0}
        while True:
            self._heartbeat()
            if self.paused_since() is not None:
                return
            started = _now()
            try:
                page = await asyncio.to_thread(
                    list_queue, self.settings, status="pending", limit=limit, exclude=sorted(exclude)
                )
            except Exception as e:  # noqa: BLE001 - sorted out by _stop_for
                await self._stop_for(e, started)
                return
            self.st.last_poll_at = _now()
            if not page.tracking_enabled:
                self._not_tracking(page.reason)
                return
            self._generation_rechecked = False
            if not page.items:
                break
            self.st.state = "processing"
            if not await self._process_page(page.items, exclude):
                return
            self.st.unavailable_streak = 0
            if len(page.items) < limit:
                break
        self.st.unavailable_streak = 0  # caught up, the queue empty included: the next outage starts at a minute again
        self.st.last_pass = {**self.st.pass_, "finished_at": int(_now())}

    async def _process_page(self, items: list[SearchIndexQueueItem], exclude: set[int]) -> bool:
        """Process one page; False when it had to stop (paused)."""
        settings = self.settings
        extracts: dict[int, dict[str, Any]] = {}
        prefailed: dict[int, str] = {}
        # the records a page needs, fetched by study id in a few requests: options only for the partial updates, the
        # full record (metadata, resources) for the full indexes
        partial = [
            i
            for i in items
            if i.object_type == "survey"
            and i.change_class == "upsert_partial"
            and settings.search_backend == "opensearch"
        ]
        full = [i for i in items if i.object_type == "survey" and i.change_class == "upsert_full"]
        batches = [([i.object_id for i in partial], False)] if partial else []
        batches += [
            ([i.object_id for i in full[start : start + FULL_RECORDS_PER_REQUEST]], True)
            for start in range(0, len(full), FULL_RECORDS_PER_REQUEST)
        ]
        for sids, with_metadata in batches:
            started = _now()
            try:
                batch = await asyncio.to_thread(
                    fetch_studies_by_sids,
                    settings,
                    sids,
                    include_metadata=with_metadata,
                    include_resources=with_metadata,
                )
            except Exception as e:  # noqa: BLE001 - sorted out by _stop_for
                await self._stop_for(e, started)
                return False
            for doc in batch["studies"]:
                core = doc.get("core_fields") or {}
                if core.get("catalog_id") is not None:
                    extracts[int(core["catalog_id"])] = doc
            for sid in batch["missing"]:
                prefailed[sid] = "the study is not in NADA's catalog any more"
            for err in batch["errors"]:
                prefailed[int(err.get("catalog_id") or 0)] = (
                    f"NADA could not build the study's extract: {err.get('message')}"
                )

        needs_embedding = any(
            i.object_type == "survey" and not i.is_delete and i.change_class in ("upsert_full", "upsert_partial")
            for i in items
        )
        if needs_embedding and settings.embedding_backend == "local":
            await ensure_embedding_initialized(self.s)
        if needs_embedding:
            async with self.s.ingest_semaphore:
                return await self._process_items(items, exclude, extracts, prefailed)
        return await self._process_items(items, exclude, extracts, prefailed)

    async def _process_items(
        self,
        items: list[SearchIndexQueueItem],
        exclude: set[int],
        extracts: dict[int, dict[str, Any]],
        prefailed: dict[int, str],
    ) -> bool:
        acks: list[dict[str, Any]] = []
        gate = nada_gate.get_gate()
        stopped = False
        for item in items:
            if self.paused_since() is not None:
                stopped = True  # paused: the rest stays pending in NADA
                break
            self.st.current = {
                "queue_id": item.id,
                "object_type": item.object_type,
                "object_key": item.object_key,
                "change_class": item.change_class,
                "started_at": int(_now()),
            }
            self._heartbeat()
            if item.object_type == "survey" and item.object_id in prefailed:
                acks.append(self._failed(item, prefailed[item.object_id]))
            else:
                started = _now()
                try:
                    action = await asyncio.to_thread(
                        process_item,
                        self.settings,
                        item,
                        extract=extracts.get(item.object_id),
                        embedding=self.s.embedding,
                    )
                except nada_gate.NadaFirewallBlocked as e:
                    self._pause_for(e)
                    stopped = True
                    break
                except Exception as e:  # noqa: BLE001 - classified below: NADA's trouble, or the item's
                    trouble = isinstance(e, nada_gate.NadaTrouble) or (
                        gate is not None and gate.trouble_since(started) is not None
                    )
                    if trouble and not await self._nada_answers():
                        self._pause_unavailable(str(e))
                        stopped = True
                        break
                    logger.warning("search-index item failed (%s %s): %s", item.change_class, item.object_key, e)
                    acks.append(self._failed(item, str(e)))
                else:
                    # a delete too: NADA takes indexed or failed only, and handles a delete row by its change class
                    acks.append({"id": item.id, "result": "indexed", "changed": item.changed})
                    self.st.pass_[action] = self.st.pass_.get(action, 0) + 1
            self.st.pass_["processed"] = self.st.pass_.get("processed", 0) + 1
            if len(acks) >= ACK_FLUSH_SIZE:
                if not await self._send_acks(acks, exclude):
                    return False
                acks = []
        # the acks of what was done are sent even when it stopped early
        if acks and not await self._send_acks(acks, exclude):
            return False
        self.st.current = None
        return not stopped

    def _failed(self, item: SearchIndexQueueItem, error: str) -> dict[str, Any]:
        self.st.pass_["failed"] = self.st.pass_.get("failed", 0) + 1
        self.st.last_error = {"at": int(_now()), "kind": "item", "object_key": item.object_key, "message": error[:500]}
        return {"id": item.id, "result": "failed", "changed": item.changed, "error": error}

    async def _send_acks(self, acks: list[dict[str, Any]], exclude: set[int], *, resend: bool = False) -> bool:
        """Send acks (100 at most per request); False when the scheduler had to pause. Unsent acks are kept."""
        for start in range(0, len(acks), 100):
            chunk = acks[start : start + 100]
            try:
                results = await asyncio.to_thread(ack_batch, self.settings, chunk)
            except AckRefused as e:
                if e.code in _REFUSAL_REASONS:
                    self.st.unsent_acks = acks[start:]
                    self._not_tracking(_REFUSAL_REASONS[e.code])
                    return False
                if e.status == 403:
                    self.st.unsent_acks = acks[start:]
                    self._pause(
                        "access_denied", "NADA refuses NADA-AI's key (ACCESS_DENIED)", ACCESS_DENIED_PAUSE_SECONDS
                    )
                    return False
                # rate limited (429) or failing (5xx): NADA's trouble, not a refusal of these acks
                if (e.status == 429 or e.status >= 500) and not await self._nada_answers():
                    self.st.unsent_acks = acks[start:]
                    self._pause_unavailable(str(e))
                    return False
                if not self._ack_failed(chunk, exclude, str(e)):
                    self.st.unsent_acks = acks[start:]
                    return False
                continue
            except Exception as e:  # noqa: BLE001 - the gate's errors: NADA (or a firewall) could not take the acks
                if isinstance(e, nada_gate.NadaFirewallBlocked) or not await self._nada_answers():
                    self.st.unsent_acks = acks[start:]
                    self._pause_for(e)
                    return False
                if not self._ack_failed(chunk, exclude, str(e)):
                    self.st.unsent_acks = acks[start:]
                    return False
                continue
            self.st.consecutive_ack_failures = 0
            self.st.ack_pause_streak = 0
            # rows NADA no longer lists as pending: recorded, or already gone (a changed row stays pending)
            self.st.pass_["acked"] = self.st.pass_.get("acked", 0) + sum(
                1 for r in results if r.get("applied") or r.get("error") == "QUEUE_NOT_FOUND"
            )
            for result in results:
                if not result.get("applied") and result.get("error") not in ("QUEUE_CHANGED", "QUEUE_NOT_FOUND"):
                    logger.error("NADA did not record ack %s: %s", result.get("id"), result.get("error"))
                    exclude.add(int(result.get("id") or 0))
        if resend:
            self.st.unsent_acks = []
        return True

    def _ack_failed(self, chunk: list[dict[str, Any]], exclude: set[int], message: str) -> bool:
        """Acks NADA could not record while it answers: skip their rows in this pass; trip the breaker at the limit."""
        self.st.consecutive_ack_failures += len(chunk)
        exclude.update(int(a["id"]) for a in chunk)
        self.st.last_error = {"at": int(_now()), "kind": "ack", "message": message[:500]}
        logger.error(
            "NADA did not record %d acks (%d in a row): %s", len(chunk), self.st.consecutive_ack_failures, message
        )
        if self.st.consecutive_ack_failures >= self.settings.ack_failure_limit:
            delay = min(ACK_FAILURE_PAUSE_SECONDS * (2**self.st.ack_pause_streak), ACK_FAILURE_PAUSE_MAX_SECONDS)
            self.st.ack_pause_streak += 1
            self._pause(
                "acks_failing",
                f"NADA refused {self.st.consecutive_ack_failures} acks in a row: {message}",
                delay,
            )
            return False
        return True
