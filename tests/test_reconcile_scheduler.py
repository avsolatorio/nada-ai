"""Tests for the in-process search-index reconciliation scheduler
(app/reconcile_scheduler.py) — the piece that lets a queue-driven reindex
single-flight against a webhook/admin-triggered one for the same idno via
the shared JobRegistry, instead of racing with no coordination at all."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest

from nada_ai.app import reconcile_scheduler
from nada_ai.app._ingest import content_sync_job_key
from nada_ai.app.jobs import JobRegistry
from nada_ai.app.state import AppState
from nada_ai.ingest.search_index_sync import SearchIndexQueueItem, SearchIndexStatus
from nada_ai.settings import Settings


def _queue_item(idno: str, *, delete: bool = False, item_id: int = 1) -> SearchIndexQueueItem:
    return SearchIndexQueueItem(
        id=item_id, object_type="survey", object_id=item_id, object_key=idno,
        change_class="delete" if delete else "upsert_full",
        status="pending", changed=1700000000, fetch_document=not delete,
    )


def _state() -> AppState:
    s = AppState()
    s.settings = Settings(
        embedding_backend="opensearch_ml",
        opensearch_ml_model_id="dummy-model",
        opensearch_ml_embedding_dimension=8,
        reconcile_search_index_batch_limit=50,
        reconcile_search_index_interval_seconds=30,
    )
    s.jobs = JobRegistry()
    s.embedding = None
    s.embedding_init_lock = asyncio.Lock()
    s.embedding_init_error = None
    s.ingest_semaphore = asyncio.Semaphore(s.settings.max_concurrent_ingest_jobs)
    return s


@pytest.mark.asyncio
async def test_submit_one_delete_uses_delete_key():
    s = _state()
    item = _queue_item("DOC-1", delete=True)
    with patch("nada_ai.app.reconcile_scheduler.apply_and_ack_queue_item", return_value={"action": "deleted"}):
        await reconcile_scheduler._submit_one(s, item)

    jobs = s.jobs.list()
    assert len(jobs) == 1
    assert jobs[0].key == "delete:DOC-1"
    assert jobs[0].kind == "search_index_reconcile_delete"


@pytest.mark.asyncio
async def test_submit_one_upsert_uses_content_sync_job_key_matching_other_paths():
    s = _state()
    item = _queue_item("DOC-1")
    with patch("nada_ai.app.reconcile_scheduler.lookup_metadata_type", return_value="indicator"), \
         patch("nada_ai.app.reconcile_scheduler.apply_and_ack_queue_item", return_value={"action": "indexed"}):
        await reconcile_scheduler._submit_one(s, item)

    jobs = s.jobs.list()
    assert len(jobs) == 1
    # Must match exactly what webhooks.py / catalog_admin.py build for the same (type, idno).
    assert jobs[0].key == content_sync_job_key("indicator", "DOC-1") == "content:indicator:DOC-1"


@pytest.mark.asyncio
async def test_submit_one_upsert_unknown_type_falls_back_to_idno_only_key():
    s = _state()
    item = _queue_item("DOC-1")
    with patch("nada_ai.app.reconcile_scheduler.lookup_metadata_type", return_value=None), \
         patch("nada_ai.app.reconcile_scheduler.apply_and_ack_queue_item", return_value={"action": "failed"}):
        await reconcile_scheduler._submit_one(s, item)

    jobs = s.jobs.list()
    assert jobs[0].key == "content:unknown:DOC-1"


@pytest.mark.asyncio
async def test_submit_one_dedupes_against_itself_for_same_idno():
    """Core regression: two reconciliation submissions for the same idno while
    the first is still running must single-flight, not run twice."""
    s = _state()
    gate = threading.Event()

    def slow_apply(settings, item, metadata_type=None, embedding=None):
        gate.wait(timeout=5)
        return {"action": "indexed"}

    with patch("nada_ai.app.reconcile_scheduler.lookup_metadata_type", return_value="indicator"), \
         patch("nada_ai.app.reconcile_scheduler.apply_and_ack_queue_item", side_effect=slow_apply):
        await reconcile_scheduler._submit_one(s, _queue_item("DOC-1", item_id=1))
        await reconcile_scheduler._submit_one(s, _queue_item("DOC-1", item_id=2))

    jobs = s.jobs.list()
    assert len(jobs) == 1  # second submission returned the already-running job, no new one created
    gate.set()


@pytest.mark.asyncio
async def test_poll_once_submits_a_job_per_pending_item():
    s = _state()
    items = [_queue_item("A"), _queue_item("B", delete=True)]
    status = SearchIndexStatus(status="success", search_provider="semantic", tracking_enabled=True)

    with patch("nada_ai.app.reconcile_scheduler.get_status", return_value=status), \
         patch("nada_ai.app.reconcile_scheduler.list_queue", return_value=items), \
         patch("nada_ai.app.reconcile_scheduler._submit_one", new=AsyncMock()) as mock_submit:
        result = await reconcile_scheduler.poll_once(s)

    assert result == {"polled": 2}
    assert mock_submit.await_count == 2


@pytest.mark.asyncio
async def test_poll_once_warns_but_still_polls_when_tracking_disabled(caplog):
    s = _state()
    status = SearchIndexStatus(status="success", search_provider="db", tracking_enabled=False)

    with patch("nada_ai.app.reconcile_scheduler.get_status", return_value=status), \
         patch("nada_ai.app.reconcile_scheduler.list_queue", return_value=[]) as mock_list, \
         caplog.at_level("WARNING"):
        result = await reconcile_scheduler.poll_once(s)

    assert result == {"polled": 0}
    mock_list.assert_called_once()
    assert any("tracking_enabled=false" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_poll_once_survives_status_check_failure():
    """A NADA API hiccup on the informational status check must not block polling the queue."""
    s = _state()
    with patch("nada_ai.app.reconcile_scheduler.get_status", side_effect=RuntimeError("network blip")), \
         patch("nada_ai.app.reconcile_scheduler.list_queue", return_value=[]) as mock_list:
        result = await reconcile_scheduler.poll_once(s)

    assert result == {"polled": 0}
    mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_reconcile_loop_cancels_cleanly():
    s = _state()
    s.settings.reconcile_search_index_interval_seconds = 30

    with patch("nada_ai.app.reconcile_scheduler.poll_once", new=AsyncMock(return_value={"polled": 0})):
        task = asyncio.create_task(reconcile_scheduler.reconcile_loop(s))
        await asyncio.sleep(0.05)  # let it run one poll and enter the sleep
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert task.cancelled()
