"""The search-index scheduler (app/reconcile_scheduler.py): works through NADA's change queue one item at a time,
pausing on NADA's trouble and skipping an item that is the problem itself."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from nada_ai.app import reconcile_scheduler
from nada_ai.app.reconcile_scheduler import Scheduler, with_health
from nada_ai.app.state import AppState
from nada_ai.ingest.search_index_sync import AckRefused, QueuePage, SearchIndexQueueItem, SearchIndexStatus
from nada_ai.nada.gate import NadaFirewallBlocked, NadaUnavailable
from nada_ai.settings import Settings


def _item(item_id: int, *, change_class: str = "upsert_full", object_type: str = "survey") -> SearchIndexQueueItem:
    return SearchIndexQueueItem(
        id=item_id,
        object_type=object_type,
        object_id=item_id,
        object_key=f"S{item_id}",
        change_class=change_class,
        status="pending",
        changed=1700000000 + item_id,
        data_type="timeseries",
        fetch_document=change_class != "delete",
    )


def _state(**settings) -> AppState:
    s = AppState()
    s.settings = Settings(
        search_backend="opensearch",
        embedding_backend="opensearch_ml",
        opensearch_ml_model_id="dummy-model",
        opensearch_ml_embedding_dimension=8,
        reconcile_search_index_batch_limit=settings.pop("limit", 2),
        reconcile_search_index_interval_seconds=30,
        **settings,
    )
    s.embedding = None
    s.embedding_init_lock = asyncio.Lock()
    s.embedding_init_error = None
    s.ingest_semaphore = asyncio.Semaphore(1)
    return s


class FakeNada:
    """The queue NADA serves, and what it records."""

    def __init__(self, items: list[SearchIndexQueueItem], *, tracking: bool = True, reason: str | None = None):
        self.items = list(items)
        self.tracking = tracking
        self.reason = reason
        self.acked: list[dict] = []
        self.status_ok = True
        self.ack_error: Exception | None = None
        self.fetched: list = []

    def list_queue(self, settings, *, status, limit, exclude):
        if not self.tracking:
            return QueuePage(items=[], tracking_enabled=False, reason=self.reason)
        left = [i for i in self.items if i.id not in exclude]
        return QueuePage(items=left[:limit], total=len(left), tracking_enabled=True)

    def ack_batch(self, settings, acks):
        if not self.status_ok:
            raise NadaUnavailable("NADA is down")
        if self.ack_error is not None:
            raise self.ack_error
        self.acked.extend(acks)
        done = {a["id"] for a in acks}
        self.items = [i for i in self.items if i.id not in done]
        return [{"id": a["id"], "applied": True, "result": a["result"]} for a in acks]

    def get_status(self, settings):
        if not self.status_ok:
            raise NadaUnavailable("NADA is down")
        return SearchIndexStatus(status="success", tracking_enabled=True)

    def fetch_studies_by_sids(self, settings, sids, include_metadata=False, include_resources=False):
        self.fetched.append((list(sids), include_metadata))
        return {
            "studies": [{"core_fields": {"catalog_id": s}, "filters": {}} for s in sids],
            "missing": [],
            "errors": [],
        }


def _run(nada: FakeNada, process_item, **settings) -> Scheduler:
    scheduler = Scheduler(_state(**settings))
    with (
        patch.object(reconcile_scheduler, "list_queue", nada.list_queue),
        patch.object(reconcile_scheduler, "ack_batch", nada.ack_batch),
        patch.object(reconcile_scheduler, "get_status", nada.get_status),
        patch.object(
            reconcile_scheduler, "fetch_studies_by_sids", lambda *a, **kw: nada.fetch_studies_by_sids(*a, **kw)
        ),
        patch.object(reconcile_scheduler, "process_item", process_item),
    ):
        asyncio.run(scheduler._drain())
    return scheduler


def test_works_through_every_page_until_the_queue_is_empty():
    nada = FakeNada([_item(i) for i in range(1, 6)])
    scheduler = _run(nada, lambda settings, item, **kw: "indexed")

    assert [a["id"] for a in nada.acked] == [1, 2, 3, 4, 5]
    assert nada.items == []
    assert scheduler.st.pause is None
    assert scheduler.st.last_pass["indexed"] == 5
    assert scheduler.st.last_pass["acked"] == 5


def test_an_item_that_fails_while_nada_answers_is_acked_failed_and_skipped():
    nada = FakeNada(
        [_item(1), _item(2), _item(3)],
    )

    def process(settings, item, **kw):
        if item.id == 2:
            raise NadaUnavailable("HTTP 500 for this study")
        return "indexed"

    scheduler = _run(nada, process, limit=5)
    assert [(a["id"], a["result"]) for a in nada.acked] == [(1, "indexed"), (2, "failed"), (3, "indexed")]
    assert scheduler.st.pause is None


def test_nada_going_down_pauses_and_keeps_the_acks_and_the_rest_pending():
    nada = FakeNada([_item(1), _item(2), _item(3)])

    def process(settings, item, **kw):
        if item.id == 2:
            nada.status_ok = False
            raise NadaUnavailable("timeout")
        return "indexed"

    scheduler = _run(nada, process, limit=5)
    assert nada.acked == []  # item 1's ack could not be sent: NADA is down
    assert scheduler.st.unsent_acks == [{"id": 1, "result": "indexed", "changed": 1700000001}]
    assert scheduler.st.pause["reason"] == "nada_unavailable"
    assert scheduler.st.pause["resume_at"] - scheduler.st.pause["since"] == 60
    assert [i.id for i in nada.items] == [1, 2, 3]


def test_a_firewall_block_pauses_without_failing_the_item():
    nada = FakeNada([_item(1)])

    def process(settings, item, **kw):
        raise NadaFirewallBlocked("GET /studies blocked (HTTP 403)", cf_ray="ray1", status=403)

    scheduler = _run(nada, process)
    assert nada.acked == []
    assert scheduler.st.pause["reason"] == "firewall"
    assert "ray1" in scheduler.st.pause["message"]


def test_not_tracking_pauses_with_the_reason():
    nada = FakeNada([_item(1)], tracking=False, reason="other_nada_ai")
    scheduler = _run(nada, lambda settings, item, **kw: "indexed")
    assert scheduler.st.state == "not_tracking"
    assert scheduler.st.pause["reason"] == "other_nada_ai"


def test_acks_that_keep_failing_trip_the_breaker():
    nada = FakeNada([_item(i) for i in range(1, 30)])
    nada.ack_error = AckRefused("DB is read-only", 400)
    scheduler = _run(nada, lambda settings, item, **kw: "indexed", nada_ack_failure_limit=10, limit=50)

    assert scheduler.st.pause["reason"] == "acks_failing"
    assert scheduler.st.consecutive_ack_failures == 10
    assert len(scheduler.st.unsent_acks) == 10  # nothing more was started


def test_a_generation_mismatch_on_ack_pauses_as_another_nada_ai():
    nada = FakeNada([_item(1)])
    nada.ack_error = AckRefused("INDEX_GENERATION_MISMATCH", 400)
    scheduler = _run(nada, lambda settings, item, **kw: "indexed")
    assert scheduler.st.pause["reason"] == "other_nada_ai"
    assert len(scheduler.st.unsent_acks) == 1


def test_partial_updates_of_a_page_are_fetched_in_one_request():
    nada = FakeNada([_item(1, change_class="upsert_partial"), _item(2, change_class="upsert_partial")])
    fetched: list = []
    original = nada.fetch_studies_by_sids
    nada.fetch_studies_by_sids = lambda settings, sids, **kw: fetched.append(list(sids)) or original(settings, sids)
    seen: list = []
    _run(nada, lambda settings, item, extract=None, **kw: seen.append(extract) or "indexed")

    assert fetched == [[1, 2]]
    assert [e["core_fields"]["catalog_id"] for e in seen] == [1, 2]


def test_full_indexes_of_a_page_get_their_full_records_by_id_in_one_request():
    nada = FakeNada([_item(1), _item(2), _item(3, change_class="upsert_partial")])
    _run(nada, lambda settings, item, **kw: "indexed", limit=5)
    assert sorted(nada.fetched) == [([1, 2], True), ([3], False)]


def test_a_study_gone_from_the_catalog_fails_its_partial_update():
    nada = FakeNada([_item(1, change_class="upsert_partial")])
    nada.fetch_studies_by_sids = lambda settings, sids, **kw: {"studies": [], "missing": [1], "errors": []}
    _run(nada, lambda settings, item, **kw: "indexed")
    assert nada.acked[0]["result"] == "failed"
    assert "not in NADA" in nada.acked[0]["error"]


def test_a_heartbeat_that_stops_moving_reads_as_stuck():
    snapshot = {"state": "idle", "heartbeat_at": 1000, "next_poll_at": 1300}
    assert with_health(snapshot, now=1500)["state"] == "idle"
    assert with_health(snapshot, now=1300 + reconcile_scheduler.STUCK_GRACE_SECONDS + 1)["state"] == "stuck"
    processing = {"state": "processing", "heartbeat_at": 1000}
    assert with_health(processing, now=1000 + reconcile_scheduler.STUCK_PROCESSING_SECONDS + 1)["state"] == "stuck"


@pytest.mark.asyncio
async def test_only_one_process_runs_the_scheduler():
    first = Scheduler(_state(reconcile_search_index_enabled=True, search_index_base_url="https://nada.example/api"))
    second = Scheduler(_state(reconcile_search_index_enabled=True, search_index_base_url="https://nada.example/api"))

    async def idle(self):
        await asyncio.sleep(3600)

    with patch.object(Scheduler, "_watchdog", idle):
        first.start()
        second.start()
        try:
            assert first.st.running_here is True
            assert second.st.state == "running_elsewhere"
            assert second.wake() == {"started": False, "state": "running_elsewhere"}
        finally:
            await first.stop()
            await second.stop()


def test_it_does_not_start_without_nadas_url():
    scheduler = Scheduler(_state(reconcile_search_index_enabled=True))
    with patch.object(reconcile_scheduler.metadata_catalog, "url", None):
        scheduler.start()
    assert scheduler.st.state == "not_configured"


def test_a_pause_stops_between_items_and_leaves_the_rest_pending():
    nada = FakeNada([_item(1), _item(2), _item(3)])
    scheduler_box: list[Scheduler] = []

    def process(settings, item, **kw):
        if item.id == 1:
            scheduler_box[0].pause()  # paused while item 1 is in hand: it is finished, nothing more is started
        return "indexed"

    scheduler = Scheduler(_state(limit=5))
    scheduler_box.append(scheduler)
    with (
        patch.object(reconcile_scheduler, "list_queue", nada.list_queue),
        patch.object(reconcile_scheduler, "ack_batch", nada.ack_batch),
        patch.object(reconcile_scheduler, "get_status", nada.get_status),
        patch.object(reconcile_scheduler, "fetch_studies_by_sids", nada.fetch_studies_by_sids),
        patch.object(reconcile_scheduler, "process_item", process),
    ):
        asyncio.run(scheduler._drain())

    assert [a["id"] for a in nada.acked] == [1]
    assert [i.id for i in nada.items] == [2, 3]
    status = scheduler.status()
    assert status["state"] == "paused" and status["pause"]["reason"] == "manual"


def test_process_pending_does_nothing_while_paused_and_resume_clears_it():
    scheduler = Scheduler(_state())
    scheduler.pause()
    assert scheduler.wake() == {"started": False, "state": "paused"}
    assert scheduler.paused_since() is not None

    scheduler.resume()
    assert scheduler.paused_since() is None
    assert scheduler.status()["state"] != "paused"


def test_a_pause_survives_a_restart():
    Scheduler(_state()).pause()
    assert Scheduler(_state()).paused_since() is not None  # a new process reads the same control file
