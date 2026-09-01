"""Tests for the NADA search-index change-queue reconciliation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from nada_ai.ingest.search_index_sync import (
    QueueItemChanged,
    SearchIndexQueueItem,
    ack_item,
    apply_and_ack_queue_item,
    get_status,
    list_queue,
    lookup_metadata_type,
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


# Real single-study response shape from a live instance
# (https://nada-demo.ihsn.org/index.php/api/admin/search-metadata-extract/studies/{idno}):
# dataset_type lives at study["filters"]["dataset_type"], NOT at the study's top
# level despite the catalog-admin OpenAPI spec documenting a top-level field.
LIVE_STUDY_RESPONSE = {
    "status": "success",
    "study": {
        "core_fields": {"idno": "WB_LSMS_001"},
        "filters": {
            "doctype": 1,
            "published": 1,
            "dataset_type": "document",
            "formid": None,
            "form_model": None,
            "year_start": 2020,
            "year_end": 2020,
            "years": [2020],
            "repositoryid": "central",
            "repositories": ["central"],
            "countries": [],
            "regions": [],
            "data_class_id": None,
            "tags": [],
        },
        "metadata": {},
        "admin_metadata": {},
    },
}


def test_lookup_metadata_type_reads_dataset_type_from_filters():
    with patch(
        "nada_ai.ingest.search_index_sync.catalog_extract.fetch_extract_study",
        return_value=LIVE_STUDY_RESPONSE,
    ):
        result = lookup_metadata_type(_settings(), "WB_LSMS_001")
    assert result == "document"


def test_lookup_metadata_type_none_when_dataset_type_unmapped():
    resp = {"status": "success", "study": {**LIVE_STUDY_RESPONSE["study"], "filters": {"dataset_type": "script"}}}
    with patch("nada_ai.ingest.search_index_sync.catalog_extract.fetch_extract_study", return_value=resp):
        result = lookup_metadata_type(_settings(), "SOME_SCRIPT_IDNO")
    assert result is None


def _queue_item(idno: str, *, delete: bool = False, item_id: int = 1) -> SearchIndexQueueItem:
    return SearchIndexQueueItem(
        id=item_id, object_type="survey", object_id=item_id, object_key=idno,
        change_class="delete" if delete else "upsert_full",
        status="pending", changed=1700000000, fetch_document=not delete,
    )


def test_reconcile_once_indexes_upsert_and_acks_indexed():
    items = [_queue_item("WLD_2021_TEST_v01")]
    with patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items), \
         patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="indicator"), \
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
         patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value=None), \
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
         patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="indicator"), \
         patch("nada_ai.ingest.search_index_sync.index_ids_op", side_effect=RuntimeError("boom")), \
         patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack:
        summary = reconcile_once(_settings(), limit=10)

    assert mock_ack.call_args.kwargs["result"] == "failed"
    assert "boom" in mock_ack.call_args.kwargs["error"]
    assert summary == {"polled": 1, "indexed": 0, "deleted": 0, "failed": 1, "ack_conflicts": 0}


def test_reconcile_once_counts_ack_conflict_without_raising():
    items = [_queue_item("WLD_2021_TEST_v01")]
    with patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items), \
         patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="indicator"), \
         patch("nada_ai.ingest.search_index_sync.index_ids_op"), \
         patch("nada_ai.ingest.search_index_sync.ack_item", side_effect=QueueItemChanged("conflict")):
        summary = reconcile_once(_settings(), limit=10)

    assert summary["ack_conflicts"] == 1
    assert summary["indexed"] == 1


def test_apply_and_ack_queue_item_uses_pre_resolved_metadata_type():
    """The scheduler resolves metadata_type BEFORE calling this (to build a
    matching job-registry key) and must not pay for a second lookup here."""
    item = _queue_item("WLD_2021_TEST_v01")
    with patch("nada_ai.ingest.search_index_sync.lookup_metadata_type") as mock_lookup, \
         patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index, \
         patch("nada_ai.ingest.search_index_sync.ack_item"):
        outcome = apply_and_ack_queue_item(_settings(), item, metadata_type="document")

    mock_lookup.assert_not_called()
    assert mock_index.call_args.kwargs["metadata_type"] == "document"
    assert outcome == {"idno": "WLD_2021_TEST_v01", "action": "indexed", "ack_conflict": False, "error": None}


def test_apply_and_ack_queue_item_falls_back_to_lookup_when_type_omitted():
    item = _queue_item("WLD_2021_TEST_v01")
    with patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="indicator") as mock_lookup, \
         patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index, \
         patch("nada_ai.ingest.search_index_sync.ack_item"):
        apply_and_ack_queue_item(_settings(), item)

    mock_lookup.assert_called_once()
    assert mock_index.call_args.kwargs["metadata_type"] == "indicator"
