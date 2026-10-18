"""The search-index scheduler (app/reconcile_scheduler.py): works through NADA's change queue one item at a time,
pausing on NADA's trouble and skipping an item that is the problem itself."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
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
        self.list_error: Exception | None = None
        self.fetch_error: Exception | None = None

    def list_queue(self, settings, *, status, limit, exclude):
        if self.list_error is not None:
            raise self.list_error
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
        if self.fetch_error is not None:
            raise self.fetch_error
        self.fetched.append((list(sids), include_metadata))
        return {
            "studies": [{"core_fields": {"catalog_id": s}, "filters": {}} for s in sids],
            "missing": [],
            "errors": [],
        }


def _run(nada: FakeNada, process_item, scheduler: Scheduler | None = None, **settings) -> Scheduler:
    scheduler = scheduler or Scheduler(_state(**settings))
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


def test_an_outage_pause_waits_at_least_as_long_as_nada_asked():
    import time

    from nada_ai.nada.gate import NadaGate

    gate = NadaGate(min_interval_seconds=0, max_retries=0, retry_base_seconds=0, retry_after_max_seconds=60)
    nada = FakeNada([])
    nada.list_error = NadaUnavailable("GET /queue: HTTP 503")
    nada.status_ok = False
    index = lambda settings, item, **kw: "indexed"  # noqa: E731
    with patch.object(reconcile_scheduler.nada_gate, "get_gate", lambda: gate):
        gate.not_before = time.time() + 3600  # NADA's Retry-After, longer than the backoff's first minute
        scheduler = _run(nada, index)
        assert 3590 <= _pause_seconds(scheduler) <= 3600
        assert "NADA asked to wait" in scheduler.st.pause["message"]

        gate.not_before = time.time() + 10 * 24 * 3600  # a far-off date: capped
        scheduler = _run(nada, index)
        assert _pause_seconds(scheduler) == reconcile_scheduler.RETRY_AFTER_PAUSE_MAX_SECONDS

        gate.not_before = time.time() - 1  # already past: the backoff alone
        scheduler = _run(nada, index)
        assert _pause_seconds(scheduler) == 60


class RateLimitedNada(FakeNada):
    """NADA behind a real request gate, answering 429 with ``Retry-After: 900`` to what ``limited`` names."""

    def __init__(self, items, limited: set[str]):
        from nada_ai.nada.gate import NadaGate

        super().__init__(items)
        self.gate = NadaGate(min_interval_seconds=0, max_retries=0, retry_base_seconds=0, retry_after_max_seconds=60)
        self.limited = limited
        self.status_probes = 0

    def _request(self, what: str) -> None:
        def answer(request):
            if what == "status":
                self.status_probes += 1
            if what in self.limited:
                return httpx.Response(429, json={}, headers={"Retry-After": "900"}, request=request)
            return httpx.Response(200, json={}, request=request)

        self.gate.send(answer, httpx.Request("GET", f"https://nada.example/{what}")).raise_for_status()

    def list_queue(self, settings, **kw):
        self._request("queue")
        return super().list_queue(settings, **kw)

    def get_status(self, settings):
        self._request("status")
        return super().get_status(settings)

    def ack_batch(self, settings, acks):
        try:
            self._request("ack")
        except httpx.HTTPStatusError as e:
            raise AckRefused("TOO_MANY_REQUESTS", e.response.status_code) from e
        return super().ack_batch(settings, acks)


def _run_gated(nada: RateLimitedNada, process_item) -> Scheduler:
    with patch.object(reconcile_scheduler.nada_gate, "get_gate", lambda: nada.gate):
        return _run(nada, process_item)


def test_a_rate_limited_queue_page_pauses_as_long_as_nada_asked_and_sends_no_probe():
    nada = RateLimitedNada([_item(1)], limited={"queue"})  # the status would answer: it must not be asked
    scheduler = _run_gated(nada, lambda settings, item, **kw: "indexed")
    assert nada.status_probes == 0
    assert scheduler.st.pause["reason"] == "nada_unavailable"
    assert 890 <= _pause_seconds(scheduler) <= 900


def test_a_rate_limited_item_is_not_failed_it_waits():
    nada = RateLimitedNada([_item(1)], limited={"item"})

    def process(settings, item, **kw):
        nada._request("item")
        return "indexed"

    scheduler = _run_gated(nada, process)
    assert nada.acked == [] and [i.id for i in nada.items] == [1]  # still pending, not acked failed
    assert nada.status_probes == 0
    assert 890 <= _pause_seconds(scheduler) <= 900


def test_a_rate_limited_ack_keeps_the_acks_and_waits_without_counting_an_ack_failure():
    nada = RateLimitedNada([_item(1)], limited={"ack"})
    scheduler = _run_gated(nada, lambda settings, item, **kw: "indexed")
    assert nada.acked == [] and nada.status_probes == 0
    assert scheduler.st.pause["reason"] == "nada_unavailable"
    assert len(scheduler.st.unsent_acks) == 1 and scheduler.st.consecutive_ack_failures == 0
    assert 890 <= _pause_seconds(scheduler) <= 900


def test_outages_back_off_and_catching_up_on_an_empty_queue_starts_the_wait_over():
    nada = FakeNada([])
    nada.list_error = NadaUnavailable("GET /queue: timed out")
    nada.status_ok = False
    index = lambda settings, item, **kw: "indexed"  # noqa: E731
    scheduler = _run(nada, index)
    for _ in range(2):
        scheduler.st.pause = None  # the loop clears it once the wait is over
        _run(nada, index, scheduler)
    assert _pause_seconds(scheduler) == 240  # 60, 120, 240

    # NADA is back and nothing is waiting: the next outage starts at a minute again
    nada.list_error, nada.status_ok = None, True
    scheduler.st.pause = None
    _run(nada, index, scheduler)
    assert scheduler.st.pause is None and scheduler.st.unavailable_streak == 0
    nada.list_error, nada.status_ok = NadaUnavailable("GET /queue: timed out"), False
    _run(nada, index, scheduler)
    assert _pause_seconds(scheduler) == 60


def test_a_firewall_block_pauses_without_failing_the_item():
    nada = FakeNada([_item(1)])

    def process(settings, item, **kw):
        raise NadaFirewallBlocked("GET /studies blocked (HTTP 403)", cf_ray="ray1", status=403)

    scheduler = _run(nada, process)
    assert nada.acked == []
    assert scheduler.st.pause["reason"] == "firewall"
    assert "ray1" in scheduler.st.pause["message"]


def test_a_queue_page_nada_cannot_serve_pauses_as_unavailable():
    nada = FakeNada([_item(1)])
    nada.list_error = NadaUnavailable("GET /queue: timed out")
    nada.status_ok = False
    scheduler = _run(nada, lambda settings, item, **kw: "indexed")
    assert scheduler.st.pause["reason"] == "nada_unavailable"


def test_a_queue_page_that_fails_while_nada_answers_pauses_as_an_error_not_an_outage():
    """A 5xx of the queue endpoint alone, an answer that cannot be read, a bug: waiting out an outage fixes none."""
    for error in (NadaUnavailable("GET /queue: HTTP 500"), ValueError("unexpected answer")):
        nada = FakeNada([_item(1)])
        nada.list_error = error
        scheduler = _run(nada, lambda settings, item, **kw: "indexed")
        assert scheduler.st.pause["reason"] == "error"
        assert type(error).__name__ in scheduler.st.pause["message"]
        assert _pause_seconds(scheduler) == scheduler.settings.reconcile_search_index_interval_seconds
        assert scheduler.st.last_error["kind"] == "error"
        assert scheduler.st.unavailable_streak == 0


def test_a_queue_page_refused_with_403_pauses_as_access_denied():
    request = httpx.Request("GET", "https://nada.example/api/admin/search-index/queue")
    nada = FakeNada([_item(1)])
    nada.list_error = httpx.HTTPStatusError("403", request=request, response=httpx.Response(403, request=request))
    scheduler = _run(nada, lambda settings, item, **kw: "indexed")
    assert scheduler.st.pause["reason"] == "access_denied"


def test_a_batch_of_records_that_fails_while_nada_answers_pauses_as_an_error_and_fails_no_item():
    nada = FakeNada([_item(1), _item(2)])
    nada.fetch_error = ValueError("unexpected answer")
    scheduler = _run(nada, lambda settings, item, **kw: "indexed")
    assert scheduler.st.pause["reason"] == "error"
    assert nada.acked == []
    assert [i.id for i in nada.items] == [1, 2]


def test_not_tracking_pauses_with_the_reason():
    nada = FakeNada([_item(1)], tracking=False, reason="other_nada_ai")
    scheduler = _run(nada, lambda settings, item, **kw: "indexed")
    assert scheduler.st.state == "not_tracking"
    assert scheduler.st.pause["reason"] == "other_nada_ai"


def test_another_nada_ai_refusal_reads_the_index_generation_again():
    """The refused generation may be a cached one from before the study index was recreated."""
    nada = FakeNada([_item(1)], tracking=False, reason="other_nada_ai")
    with patch.object(reconcile_scheduler, "forget_index_generation") as forget:
        _run(nada, lambda settings, item, **kw: "indexed")
    forget.assert_called()

    nada = FakeNada([_item(1)], tracking=False, reason="not_tracking_for_nada_ai")
    with patch.object(reconcile_scheduler, "forget_index_generation") as forget:
        _run(nada, lambda settings, item, **kw: "indexed")
    forget.assert_not_called()


def _pause_seconds(scheduler: Scheduler) -> int:
    return scheduler.st.pause["resume_at"] - scheduler.st.pause["since"]


def test_a_first_another_nada_ai_refusal_looks_again_in_a_minute_a_second_one_waits_the_full_interval():
    """Right after a recreate both sides may still use a cached generation; refused twice in a row, it is another
    nada-ai's index."""
    index = lambda settings, item, **kw: "indexed"  # noqa: E731
    nada = FakeNada([_item(1)], tracking=False, reason="other_nada_ai")
    scheduler = _run(nada, index)
    assert _pause_seconds(scheduler) == reconcile_scheduler.GENERATION_RECHECK_SECONDS
    _run(nada, index, scheduler)
    assert _pause_seconds(scheduler) == scheduler.settings.reconcile_search_index_not_tracking_interval_seconds

    # once a page comes back tracked, the next refusal is a first one again
    nada.tracking = True
    _run(nada, index, scheduler)
    nada.tracking = False
    _run(nada, index, scheduler)
    assert _pause_seconds(scheduler) == reconcile_scheduler.GENERATION_RECHECK_SECONDS


def test_not_tracking_for_nada_ai_waits_the_full_interval_at_once():
    nada = FakeNada([_item(1)], tracking=False, reason="not_tracking_for_nada_ai")
    scheduler = _run(nada, lambda settings, item, **kw: "indexed")
    assert _pause_seconds(scheduler) == scheduler.settings.reconcile_search_index_not_tracking_interval_seconds


def test_acks_that_keep_failing_trip_the_breaker():
    nada = FakeNada([_item(i) for i in range(1, 30)])
    nada.ack_error = AckRefused("DB is read-only", 400)
    scheduler = _run(nada, lambda settings, item, **kw: "indexed", ack_failure_limit=10, limit=50)

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


@pytest.mark.asyncio
async def test_another_process_reports_the_running_schedulers_request_figures_not_its_own():
    import time

    from nada_ai.nada.gate import NadaGate

    def gate(requests: int) -> NadaGate:
        g = NadaGate(min_interval_seconds=0.5, max_retries=0, retry_base_seconds=0, retry_after_max_seconds=1)
        g._sent.extend([time.time()] * requests)
        return g

    first = Scheduler(_state(reconcile_search_index_enabled=True, search_index_base_url="https://nada.example/api"))
    second = Scheduler(_state(reconcile_search_index_enabled=True, search_index_base_url="https://nada.example/api"))

    async def idle(self):
        await asyncio.sleep(3600)

    with patch.object(Scheduler, "_watchdog", idle):
        first.start()
        second.start()
        try:
            with patch.object(reconcile_scheduler.nada_gate, "get_gate", lambda: gate(7)):
                first._heartbeat()  # the running process: its gate sent 7 requests
            with patch.object(reconcile_scheduler.nada_gate, "get_gate", lambda: gate(0)):
                assert second.status()["requests"]["last_hour"] == 7
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
