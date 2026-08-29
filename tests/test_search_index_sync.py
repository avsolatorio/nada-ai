"""Tests for the NADA search-index change-queue reconciliation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from nada_ai.ingest.search_index_sync import (
    QueueItemChanged,
    SearchIndexQueueItem,
    ack_item,
    get_status,
    list_queue,
    reconcile_once,
)
from nada_ai.settings import Settings


def _settings(**overrides) -> Settings:
    return Settings(search_index_base_url="https://nada.example.org/index.php/api", **overrides)


def _mock_sync_client(**responses: httpx.Response):
    """Return a context-manager mock for httpx.Client whose .get/.post return the given responses in order."""
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    for method, resp in responses.items():
        getattr(client, method).return_value = resp
    return client


def _resp(json_body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=json_body, request=httpx.Request("GET", "http://test"))


def test_get_status_parses_response():
    payload = {
        "status": "success",
        "search_provider": "semantic",
        "tracking_enabled": True,
        "queue": {"pending": 3, "failed": 1},
        "state": {"indexed": 100, "pending": 3, "failed": 1, "deleted": 0},
    }
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        status = get_status(_settings())
    assert status.tracking_enabled is True
    assert status.queue["pending"] == 3


def test_list_queue_parses_items():
    payload = {
        "status": "success",
        "tracking_enabled": True,
        "total": 1,
        "limit": 50,
        "items": [
            {
                "id": 1,
                "object_type": "survey",
                "object_id": 10,
                "object_key": "WLD_2021_TEST_v01",
                "change_class": "upsert_full",
                "status": "pending",
                "attempts": 0,
                "last_error": None,
                "changed": 1732000000,
                "fetch_document": True,
            }
        ],
    }
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        items = list_queue(_settings())
    assert len(items) == 1
    assert items[0].object_key == "WLD_2021_TEST_v01"
    assert items[0].is_delete is False


def test_delete_queue_item_is_delete_true():
    item = SearchIndexQueueItem(
        id=2, object_type="survey", object_id=11, object_key="X",
        change_class="delete", status="pending", changed=1, fetch_document=False,
    )
    assert item.is_delete is True


def test_ack_item_raises_queue_item_changed_on_409():
    request = httpx.Request("POST", "http://test")
    conflict = httpx.Response(409, json={"status": "failed"}, request=request)
    client = _mock_sync_client(post=conflict)
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        with pytest.raises(QueueItemChanged):
            ack_item(_settings(), 1, result="indexed", changed=123)


def test_ack_item_success():
    client = _mock_sync_client(post=_resp({"status": "success", "applied": True, "result": "indexed"}))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        res = ack_item(_settings(), 1, result="indexed", changed=123)
    assert res["applied"] is True


def _queue_item(idno: str, *, delete: bool = False, item_id: int = 1) -> SearchIndexQueueItem:
    return SearchIndexQueueItem(
        id=item_id, object_type="survey", object_id=item_id, object_key=idno,
        change_class="delete" if delete else "upsert_full",
        status="pending", changed=1700000000, fetch_document=not delete,
    )


def test_reconcile_once_indexes_upsert_and_acks_indexed():
    items = [_queue_item("WLD_2021_TEST_v01")]
    with patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items), \
         patch("nada_ai.ingest.search_index_sync._lookup_metadata_type", return_value="indicator"), \
         patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index, \
         patch("nada_ai.ingest.search_index_sync.delete_by_idno_op") as mock_delete, \
         patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack:
        summary = reconcile_once(_settings(), limit=10)

    mock_index.assert_called_once()
    assert mock_index.call_args.kwargs["idnos"] == ["WLD_2021_TEST_v01"]
    assert mock_index.call_args.kwargs["metadata_type"] == "indicator"
    mock_delete.assert_not_called()
    mock_ack.assert_called_once()
    assert mock_ack.call_args.args[1] == 1
    assert mock_ack.call_args.kwargs == {"result": "indexed", "changed": 1700000000, "error": None}
    assert summary == {"polled": 1, "indexed": 1, "deleted": 0, "failed": 0, "ack_conflicts": 0}


def test_reconcile_once_deletes_tombstone_and_acks_indexed():
    items = [_queue_item("WLD_2021_TEST_v01", delete=True)]
    with patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items), \
         patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index, \
         patch("nada_ai.ingest.search_index_sync.delete_by_idno_op") as mock_delete, \
         patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack:
        summary = reconcile_once(_settings(), limit=10)

    mock_delete.assert_called_once_with(mock_delete.call_args[0][0], "WLD_2021_TEST_v01")
    mock_index.assert_not_called()
    mock_ack.assert_called_once()
    assert mock_ack.call_args.kwargs["result"] == "indexed"
    assert summary == {"polled": 1, "indexed": 0, "deleted": 1, "failed": 0, "ack_conflicts": 0}


def test_reconcile_once_acks_failed_for_unmapped_dataset_type():
    items = [_queue_item("SOME_TABLE_IDNO")]
    with patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items), \
         patch("nada_ai.ingest.search_index_sync._lookup_metadata_type", return_value=None), \
         patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index, \
         patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack:
        summary = reconcile_once(_settings(), limit=10)

    mock_index.assert_not_called()
    assert mock_ack.call_args.kwargs["result"] == "failed"
    assert "mapping" in mock_ack.call_args.kwargs["error"]
    assert summary == {"polled": 1, "indexed": 0, "deleted": 0, "failed": 1, "ack_conflicts": 0}


def test_reconcile_once_acks_failed_when_index_raises():
    items = [_queue_item("WLD_2021_TEST_v01")]
    with patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items), \
         patch("nada_ai.ingest.search_index_sync._lookup_metadata_type", return_value="indicator"), \
         patch("nada_ai.ingest.search_index_sync.index_ids_op", side_effect=RuntimeError("boom")), \
         patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack:
        summary = reconcile_once(_settings(), limit=10)

    assert mock_ack.call_args.kwargs["result"] == "failed"
    assert "boom" in mock_ack.call_args.kwargs["error"]
    assert summary == {"polled": 1, "indexed": 0, "deleted": 0, "failed": 1, "ack_conflicts": 0}


def test_reconcile_once_counts_ack_conflict_without_raising():
    items = [_queue_item("WLD_2021_TEST_v01")]
    with patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items), \
         patch("nada_ai.ingest.search_index_sync._lookup_metadata_type", return_value="indicator"), \
         patch("nada_ai.ingest.search_index_sync.index_ids_op"), \
         patch("nada_ai.ingest.search_index_sync.ack_item", side_effect=QueueItemChanged("conflict")):
        summary = reconcile_once(_settings(), limit=10)

    assert summary["ack_conflicts"] == 1
    assert summary["indexed"] == 1
