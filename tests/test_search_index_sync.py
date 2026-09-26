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
    list_diff_missing,
    list_diff_stale,
    list_queue,
    list_type_breakdown,
    lookup_metadata_type,
    reconcile_diff_once,
    reconcile_once,
    report_state_bulk,
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
        "search_engine": "nada_ai",
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
        id=2,
        object_type="survey",
        object_id=11,
        object_key="X",
        change_class="delete",
        status="pending",
        changed=1,
        fetch_document=False,
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
    resp = {"status": "success", "study": {**LIVE_STUDY_RESPONSE["study"], "filters": {"dataset_type": "citation"}}}
    with patch("nada_ai.ingest.search_index_sync.catalog_extract.fetch_extract_study", return_value=resp):
        result = lookup_metadata_type(_settings(), "SOME_CITATION_IDNO")
    assert result is None


@pytest.mark.parametrize(
    ("dataset_type", "metadata_type"),
    [
        ("survey", "microdata"),
        ("timeseries", "indicator"),
        ("timeseriesdb", "indicator-db"),
        ("timeseries-db", "indicator-db"),
        ("document", "document"),
        ("geospatial", "geospatial"),
        ("table", "table"),
        ("script", "script"),
        ("image", "image"),
        ("video", "video"),
    ],
)
def test_lookup_metadata_type_maps_every_nada_dataset_type(dataset_type, metadata_type):
    resp = {"status": "success", "study": {**LIVE_STUDY_RESPONSE["study"], "filters": {"dataset_type": dataset_type}}}
    with patch("nada_ai.ingest.search_index_sync.catalog_extract.fetch_extract_study", return_value=resp):
        result = lookup_metadata_type(_settings(), "SOME_IDNO")
    assert result == metadata_type


def _queue_item(idno: str, *, delete: bool = False, item_id: int = 1, variables: bool = False) -> SearchIndexQueueItem:
    change_class = "delete" if delete else ("variables" if variables else "upsert_full")
    return SearchIndexQueueItem(
        id=item_id,
        object_type="survey",
        object_id=item_id,
        object_key=idno,
        change_class=change_class,
        status="pending",
        changed=1700000000,
        fetch_document=not delete,
    )


def test_reconcile_once_indexes_upsert_and_acks_indexed():
    items = [_queue_item("WLD_2021_TEST_v01")]
    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items),
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="indicator"),
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.delete_by_idno_op") as mock_delete,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
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
    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items),
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.delete_by_idno_op") as mock_delete,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        summary = reconcile_once(_settings(), limit=10)

    mock_delete.assert_called_once_with(mock_delete.call_args[0][0], "WLD_2021_TEST_v01")
    mock_index.assert_not_called()
    mock_ack.assert_called_once()
    assert mock_ack.call_args.kwargs["result"] == "indexed"
    assert summary == {"polled": 1, "indexed": 0, "deleted": 1, "failed": 0, "ack_conflicts": 0}


def test_reconcile_once_acks_failed_for_unmapped_dataset_type():
    items = [_queue_item("SOME_TABLE_IDNO")]
    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items),
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value=None),
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        summary = reconcile_once(_settings(), limit=10)

    mock_index.assert_not_called()
    assert mock_ack.call_args.kwargs["result"] == "failed"
    assert "mapping" in mock_ack.call_args.kwargs["error"]
    assert summary == {"polled": 1, "indexed": 0, "deleted": 0, "failed": 1, "ack_conflicts": 0}


def test_reconcile_once_acks_failed_when_index_raises():
    items = [_queue_item("WLD_2021_TEST_v01")]
    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items),
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="indicator"),
        patch("nada_ai.ingest.search_index_sync.index_ids_op", side_effect=RuntimeError("boom")),
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        summary = reconcile_once(_settings(), limit=10)

    assert mock_ack.call_args.kwargs["result"] == "failed"
    assert "boom" in mock_ack.call_args.kwargs["error"]
    assert summary == {"polled": 1, "indexed": 0, "deleted": 0, "failed": 1, "ack_conflicts": 0}


def test_reconcile_once_counts_ack_conflict_without_raising():
    items = [_queue_item("WLD_2021_TEST_v01")]
    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items),
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="indicator"),
        patch("nada_ai.ingest.search_index_sync.index_ids_op"),
        patch("nada_ai.ingest.search_index_sync.ack_item", side_effect=QueueItemChanged("conflict")),
    ):
        summary = reconcile_once(_settings(), limit=10)

    assert summary["ack_conflicts"] == 1
    assert summary["indexed"] == 1


def test_apply_and_ack_queue_item_uses_pre_resolved_metadata_type():
    """The scheduler resolves metadata_type BEFORE calling this (to build a
    matching job-registry key) and must not pay for a second lookup here."""
    item = _queue_item("WLD_2021_TEST_v01")
    with (
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type") as mock_lookup,
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.ack_item"),
    ):
        outcome = apply_and_ack_queue_item(_settings(), item, metadata_type="document")

    mock_lookup.assert_not_called()
    assert mock_index.call_args.kwargs["metadata_type"] == "document"
    assert outcome == {"idno": "WLD_2021_TEST_v01", "action": "indexed", "ack_conflict": False, "error": None}


def test_apply_and_ack_queue_item_falls_back_to_lookup_when_type_omitted():
    item = _queue_item("WLD_2021_TEST_v01")
    with (
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="indicator") as mock_lookup,
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.ack_item"),
    ):
        apply_and_ack_queue_item(_settings(), item)

    mock_lookup.assert_called_once()
    assert mock_index.call_args.kwargs["metadata_type"] == "indicator"


# ---------------------------------------------------------------------------
# change_class="variables": only the variable index syncs, not a full
# study/chunk reindex (NADA's Dataset_microdata_model::index_variable_data()
# fires this after a DDI re-import — nothing else about the study changed)
# ---------------------------------------------------------------------------


def test_variables_change_class_syncs_only_variables_on_opensearch():
    item = _queue_item("WLD_2021_TEST_v01", variables=True)
    with (
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type") as mock_lookup,
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        outcome = apply_and_ack_queue_item(_settings(search_backend="opensearch"), item)

    mock_lookup.assert_not_called()
    mock_index.assert_not_called()
    mock_sync.assert_called_once_with(_settings(search_backend="opensearch"), "WLD_2021_TEST_v01")
    assert mock_ack.call_args.kwargs["result"] == "indexed"
    assert outcome["action"] == "indexed"


def test_variables_change_class_is_a_noop_on_qdrant():
    """Qdrant has no separate variable index; the item is still acked (not left pending forever)."""
    item = _queue_item("WLD_2021_TEST_v01", variables=True)
    with (
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        outcome = apply_and_ack_queue_item(_settings(search_backend="qdrant"), item)

    mock_index.assert_not_called()
    mock_sync.assert_not_called()
    assert mock_ack.call_args.kwargs["result"] == "indexed"
    assert outcome["action"] == "indexed"


def test_variables_change_class_acks_failed_when_sync_raises():
    item = _queue_item("WLD_2021_TEST_v01", variables=True)
    with (
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op", side_effect=RuntimeError("boom")),
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        outcome = apply_and_ack_queue_item(_settings(search_backend="opensearch"), item)

    assert mock_ack.call_args.kwargs["result"] == "failed"
    assert "boom" in mock_ack.call_args.kwargs["error"]
    assert outcome["action"] == "failed"


def test_reconcile_once_handles_a_variables_item_end_to_end():
    items = [_queue_item("WLD_2021_TEST_v01", variables=True)]
    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items),
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
        patch("nada_ai.ingest.search_index_sync.ack_item"),
    ):
        summary = reconcile_once(_settings(search_backend="opensearch"), limit=10)

    mock_index.assert_not_called()
    mock_sync.assert_called_once()
    assert summary == {"polled": 1, "indexed": 1, "deleted": 0, "failed": 0, "ack_conflicts": 0}


# ---------------------------------------------------------------------------
# report_state_bulk
# ---------------------------------------------------------------------------


def test_report_state_bulk_empty_items_is_a_noop():
    with patch("nada_ai.ingest.search_index_sync.httpx.Client") as mock_client_cls:
        result = report_state_bulk(_settings(), [])
    mock_client_cls.assert_not_called()
    assert result == {"applied": 0, "results": []}


def test_report_state_bulk_single_chunk():
    items = [{"object_type": "survey", "object_key": "A", "status": "indexed"}]
    payload = {"status": "success", "applied": 1, "results": [{"object_key": "A", "applied": True}]}
    client = _mock_sync_client(post=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        result = report_state_bulk(_settings(), items)
    client.post.assert_called_once()
    assert client.post.call_args.args[0] == "/admin/search-index/state/bulk"
    assert client.post.call_args.kwargs["json"] == {"items": items}
    assert result == {"applied": 1, "results": [{"object_key": "A", "applied": True}]}


def test_report_state_bulk_splits_into_multiple_chunks():
    import nada_ai.ingest.search_index_sync as mod

    items = [{"object_type": "survey", "object_key": f"I{i}", "status": "indexed"} for i in range(3)]
    responses = [
        _resp({"status": "success", "applied": 1, "results": [{"object_key": f"I{i}", "applied": True}]})
        for i in range(3)
    ]
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.side_effect = responses

    with (
        patch.object(mod, "_STATE_BULK_CHUNK_SIZE", 1),
        patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client),
    ):
        result = report_state_bulk(_settings(), items)

    assert client.post.call_count == 3
    assert result["applied"] == 3
    assert len(result["results"]) == 3


# ---------------------------------------------------------------------------
# list_diff_missing / list_diff_stale
# ---------------------------------------------------------------------------


def test_list_diff_missing_parses_page():
    payload = {"status": "success", "items": [{"idno": "A", "type": "survey"}], "total": 5}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        page = list_diff_missing(_settings(), object_type="survey", limit=10, offset=0)
    assert page.total == 5
    assert page.items[0].idno == "A"
    assert page.items[0].type == "survey"
    assert client.get.call_args.kwargs["params"] == {"object_type": "survey", "limit": 10, "offset": 0}


def test_list_diff_missing_parses_last_error():
    payload = {
        "status": "success",
        "items": [{"idno": "A", "type": "survey", "last_error": "boom"}, {"idno": "B", "type": "survey"}],
        "total": 2,
    }
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        page = list_diff_missing(_settings(), object_type="survey", limit=10, offset=0)
    assert page.items[0].last_error == "boom"
    assert page.items[1].last_error is None


def test_list_diff_missing_total_can_exceed_items_when_nothing_indexed():
    """Empty catalog coverage must still report the real total, not look like 'nothing missing'."""
    payload = {"status": "success", "items": [{"idno": "A", "type": "survey"}], "total": 8000}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        page = list_diff_missing(_settings(), object_type="survey", limit=1)
    assert page.total == 8000
    assert len(page.items) == 1


def test_list_type_breakdown_parses_items():
    payload = {
        "status": "success",
        "object_type": "survey",
        "items": [
            {"data_type": "survey", "catalog_total": 1200, "indexed": 120, "missing": 1080, "stale": 0, "errors": 5},
            {"data_type": "geospatial", "catalog_total": 40, "indexed": 38, "missing": 2, "stale": 1, "errors": 2},
        ],
    }
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        items = list_type_breakdown(_settings(), "survey")
    assert len(items) == 2
    assert items[0].data_type == "survey"
    assert items[0].catalog_total == 1200
    assert items[0].errors == 5
    assert items[1].data_type == "geospatial"
    assert items[1].stale == 1
    assert items[1].errors == 2
    assert client.get.call_args.kwargs["params"] == {"object_type": "survey"}


def test_list_diff_stale_parses_page():
    payload = {"status": "success", "items": [{"idno": "Z"}], "total": 1}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        page = list_diff_stale(_settings(), object_type="survey")
    assert page.total == 1
    assert page.items[0].idno == "Z"
    assert page.items[0].type is None


def test_list_diff_missing_sends_data_type_when_given():
    payload = {"status": "success", "items": [], "total": 0}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        list_diff_missing(_settings(), object_type="survey", limit=10, offset=0, data_type="geospatial")
    assert client.get.call_args.kwargs["params"] == {
        "object_type": "survey",
        "limit": 10,
        "offset": 0,
        "data_type": "geospatial",
    }


def test_list_diff_missing_omits_data_type_when_not_given():
    payload = {"status": "success", "items": [], "total": 0}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        list_diff_missing(_settings(), object_type="survey", limit=10, offset=0)
    assert "data_type" not in client.get.call_args.kwargs["params"]


def test_list_diff_missing_sends_has_error_when_true():
    payload = {"status": "success", "items": [], "total": 0}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        list_diff_missing(_settings(), object_type="survey", limit=10, offset=0, has_error=True)
    assert client.get.call_args.kwargs["params"]["has_error"] == "1"


def test_list_diff_missing_omits_has_error_when_false():
    payload = {"status": "success", "items": [], "total": 0}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        list_diff_missing(_settings(), object_type="survey", limit=10, offset=0)
    assert "has_error" not in client.get.call_args.kwargs["params"]


# ---------------------------------------------------------------------------
# reconcile_diff_once
# ---------------------------------------------------------------------------


def _diff_page(items, total=None):
    from nada_ai.ingest.search_index_sync import DiffItem, DiffPage

    return DiffPage(items=[DiffItem(**i) for i in items], total=total if total is not None else len(items))


def test_reconcile_diff_once_indexes_each_missing_item_with_resolved_type():
    page = _diff_page([{"idno": "A", "type": "survey"}, {"idno": "B", "type": "geospatial"}])
    empty_page = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[page, empty_page]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
        patch(
            "nada_ai.ingest.search_index_sync.index_ids_op",
            return_value={"indexed": 1, "errors": [], "load_errors": [], "empty_docs": []},
        ) as mock_index,
    ):
        summary = reconcile_diff_once(_settings())

    assert mock_index.call_count == 2
    calls_by_idno = {c.kwargs["idnos"][0]: c.kwargs["metadata_type"] for c in mock_index.call_args_list}
    assert calls_by_idno == {"A": "microdata", "B": "geospatial"}
    assert summary["missing_total"] == 2
    assert summary["indexed"] == 2
    assert summary["failed"] == 0
    assert summary["skipped"] == 0


def test_reconcile_diff_once_counts_soft_failure_as_failed_not_indexed():
    """index_ids_op doesn't raise for a per-idno failure (load error, empty
    doc, backend write error) — it reports those in its return value. A call
    that returns without raising must not be assumed to mean the idno was
    actually indexed."""
    page = _diff_page([{"idno": "BAD", "type": "geospatial"}])
    empty_page = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[page, empty_page]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
        patch(
            "nada_ai.ingest.search_index_sync.index_ids_op",
            return_value={
                "indexed": 0,
                "errors": [],
                "load_errors": [{"idno": "BAD", "metadata_type": "geospatial", "stage": "load", "error": "boom"}],
                "empty_docs": [],
            },
        ),
    ):
        summary = reconcile_diff_once(_settings())

    assert summary["indexed"] == 0
    assert summary["failed"] == 1


def test_reconcile_diff_once_passes_data_type_through_to_both_diff_calls():
    empty_page = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", return_value=empty_page) as mock_missing,
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page) as mock_stale,
    ):
        reconcile_diff_once(_settings(), data_type="geospatial")

    assert mock_missing.call_args.kwargs["data_type"] == "geospatial"
    assert mock_stale.call_args.kwargs["data_type"] == "geospatial"


def test_reconcile_diff_once_skips_unmapped_dataset_type():
    page = _diff_page([{"idno": "A", "type": "citation"}])  # 'citation' has no metadata_type mapping
    empty_page = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[page, empty_page]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
    ):
        summary = reconcile_diff_once(_settings())

    mock_index.assert_not_called()
    assert summary["skipped"] == 1
    assert summary["indexed"] == 0


def test_reconcile_diff_once_deletes_stale_items_in_one_batch_call():
    empty_missing = _diff_page([])
    stale_page = _diff_page([{"idno": "X"}, {"idno": "Y"}])
    empty_stale = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", return_value=empty_missing),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", side_effect=[stale_page, empty_stale]),
        patch("nada_ai.ingest.search_index_sync.delete_by_idnos_op") as mock_delete,
    ):
        summary = reconcile_diff_once(_settings())

    mock_delete.assert_called_once()
    assert set(mock_delete.call_args.args[1]) == {"X", "Y"}
    assert summary["stale_total"] == 2
    assert summary["deleted"] == 2


def test_reconcile_diff_once_terminates_when_an_item_keeps_failing():
    """The 'missing' diff re-fetches from offset 0 every iteration (the set
    shrinks as items succeed) — an item that keeps failing stays in NADA's
    diff forever (correctly: it's genuinely still not indexed), so this must
    not loop forever retrying it. Simulates 200 re-fetches all returning the
    exact same permanently-broken item, plus one that succeeds and should
    disappear next iteration."""
    stuck_page = _diff_page([{"idno": "STUCK", "type": "survey"}])
    empty_page = _diff_page([])

    with (
        patch(
            "nada_ai.ingest.search_index_sync.list_diff_missing",
            side_effect=[stuck_page] * 200 + [empty_page],
        ),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
        patch("nada_ai.ingest.search_index_sync.index_ids_op", side_effect=RuntimeError("permanently broken")),
    ):
        summary = reconcile_diff_once(_settings())

    # Attempted exactly once despite appearing on every re-fetch, and terminated.
    assert summary["failed"] == 1
    assert summary["indexed"] == 0


def test_reconcile_diff_once_summary_shape():
    empty_page = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", return_value=empty_page),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
    ):
        summary = reconcile_diff_once(_settings())

    assert summary == {
        "missing_total": 0,
        "stale_total": 0,
        "indexed": 0,
        "deleted": 0,
        "failed": 0,
        "skipped": 0,
    }


def test_reconcile_diff_once_reports_progress_per_idno():
    """Jobs reads progress.total/processed/percent/failed/current_idno — the
    same snapshot shape ingest already emits via IngestProgressTracker."""
    missing = _diff_page([{"idno": "A", "type": "survey"}, {"idno": "B", "type": "geospatial"}])
    stale = _diff_page([{"idno": "X"}])
    empty = _diff_page([])
    snapshots: list[dict] = []

    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[missing, empty]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", side_effect=[stale, empty]),
        patch(
            "nada_ai.ingest.search_index_sync.index_ids_op",
            return_value={"indexed": 1, "errors": [], "load_errors": [], "empty_docs": []},
        ),
        patch("nada_ai.ingest.search_index_sync.delete_by_idnos_op"),
    ):
        summary = reconcile_diff_once(_settings(), progress_cb=snapshots.append)

    assert summary["indexed"] == 2
    assert summary["deleted"] == 1
    assert snapshots[0] == {
        "processed": 0,
        "total": 3,
        "failed": 0,
        "current_idno": None,
        "percent": 0.0,
        "phase": "missing",
    }
    assert [s["current_idno"] for s in snapshots[1:]] == ["A", "B", "X"]
    assert snapshots[-1]["processed"] == 3
    assert snapshots[-1]["percent"] == 100.0
    assert snapshots[-1]["phase"] == "stale"
    assert all("total" in s and "failed" in s for s in snapshots)


def test_reconcile_diff_once_stops_when_cancel_token_is_set():
    from nada_ai.ingest.progress import CancelToken

    missing = _diff_page([{"idno": "A", "type": "survey"}, {"idno": "B", "type": "geospatial"}])
    empty = _diff_page([])
    token = CancelToken()
    indexed: list[str] = []

    def index_one(settings, idnos, **kwargs):
        indexed.append(idnos[0])
        token.set()
        return {"indexed": 1, "errors": [], "load_errors": [], "empty_docs": []}

    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[missing, empty]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty),
        patch("nada_ai.ingest.search_index_sync.index_ids_op", side_effect=index_one),
    ):
        summary = reconcile_diff_once(_settings(), cancel_token=token)

    assert indexed == ["A"]
    assert summary["indexed"] == 1
    assert summary["cancelled"] is True
    assert summary["deleted"] == 0


# ---------------------------------------------------------------------------
# change_class="upsert_partial": a change of the study's options (publish state, license, ...). On OpenSearch the flags are applied in place (no metadata
# read, no embedding); a study that is not indexed yet, or another engine, gets the full index.
# ---------------------------------------------------------------------------


def _partial_item(idno: str) -> SearchIndexQueueItem:
    return _queue_item(idno).model_copy(update={"change_class": "upsert_partial"})


def test_partial_change_class_is_applied_in_place_on_opensearch():
    item = _partial_item("WLD_2021_TEST_v01")
    with (
        patch("nada_ai.ingest.service.apply_study_options_op", return_value={"updated": True}) as mock_publish,
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type") as mock_lookup,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        outcome = apply_and_ack_queue_item(_settings(search_backend="opensearch"), item)

    mock_publish.assert_called_once()
    mock_index.assert_not_called()
    mock_lookup.assert_not_called()
    assert mock_ack.call_args.kwargs["result"] == "indexed"
    assert outcome["action"] == "indexed"


def test_partial_change_class_indexes_in_full_when_the_study_is_not_indexed_yet():
    item = _partial_item("WLD_2021_TEST_v01")
    with (
        patch(
            "nada_ai.ingest.service.apply_study_options_op",
            return_value={"updated": False, "reason": "not_indexed"},
        ),
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="microdata"),
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        apply_and_ack_queue_item(_settings(search_backend="opensearch"), item)

    assert mock_index.call_args.kwargs["idnos"] == ["WLD_2021_TEST_v01"]
    assert mock_ack.call_args.kwargs["result"] == "indexed"


def test_partial_change_class_indexes_in_full_on_qdrant():
    item = _partial_item("WLD_2021_TEST_v01")
    with (
        patch("nada_ai.ingest.service.apply_study_options_op") as mock_publish,
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type", return_value="microdata"),
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.ack_item"),
    ):
        apply_and_ack_queue_item(_settings(search_backend="qdrant"), item)

    mock_publish.assert_not_called()
    mock_index.assert_called_once()


def test_partial_change_class_acks_failed_when_the_update_raises():
    item = _partial_item("WLD_2021_TEST_v01")
    with (
        patch("nada_ai.ingest.service.apply_study_options_op", side_effect=RuntimeError("boom")),
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        outcome = apply_and_ack_queue_item(_settings(search_backend="opensearch"), item)

    assert mock_ack.call_args.kwargs["result"] == "failed"
    assert outcome["action"] == "failed"


# ---------------------------------------------------------------------------
# object_type="citation": one citation synced (or removed) in its own lexical index, never a study reindex
# ---------------------------------------------------------------------------


def _citation_item(citation_id: int = 7, *, delete: bool = False) -> SearchIndexQueueItem:
    return SearchIndexQueueItem(
        id=citation_id,
        object_type="citation",
        object_id=citation_id,
        object_key="uuid-7",
        change_class="delete" if delete else "upsert_full",
        status="pending",
        changed=1700000000,
        fetch_document=not delete,
    )


def test_a_citation_item_syncs_that_one_citation_on_opensearch():
    item = _citation_item(7)
    with (
        patch("nada_ai.ingest.citations_index.sync_citation_op") as mock_sync,
        patch("nada_ai.ingest.search_index_sync.index_ids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.lookup_metadata_type") as mock_lookup,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        outcome = apply_and_ack_queue_item(_settings(search_backend="opensearch"), item)

    mock_sync.assert_called_once_with(_settings(search_backend="opensearch"), 7)
    mock_index.assert_not_called()
    mock_lookup.assert_not_called()
    assert mock_ack.call_args.kwargs["result"] == "indexed"
    assert outcome["action"] == "indexed"


def test_a_deleted_citation_is_removed_from_the_index():
    item = _citation_item(7, delete=True)
    with (
        patch("nada_ai.ingest.citations_index.delete_citation_op") as mock_delete,
        patch("nada_ai.ingest.citations_index.sync_citation_op") as mock_sync,
        patch("nada_ai.ingest.search_index_sync.delete_by_idno_op") as mock_study_delete,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        outcome = apply_and_ack_queue_item(_settings(search_backend="opensearch"), item)

    mock_delete.assert_called_once_with(_settings(search_backend="opensearch"), 7)
    mock_sync.assert_not_called()
    mock_study_delete.assert_not_called()
    assert mock_ack.call_args.kwargs["result"] == "indexed"
    assert outcome["action"] == "deleted"


def test_a_citation_item_is_acked_without_work_on_qdrant():
    """Qdrant has no citation index; the item is still acked rather than left pending forever."""
    item = _citation_item(7)
    with (
        patch("nada_ai.ingest.citations_index.sync_citation_op") as mock_sync,
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        outcome = apply_and_ack_queue_item(_settings(search_backend="qdrant"), item)

    mock_sync.assert_not_called()
    assert mock_ack.call_args.kwargs["result"] == "indexed"
    assert outcome["action"] == "indexed"


def test_a_failed_citation_sync_acks_failed():
    item = _citation_item(7)
    with (
        patch("nada_ai.ingest.citations_index.sync_citation_op", side_effect=RuntimeError("boom")),
        patch("nada_ai.ingest.search_index_sync.ack_item") as mock_ack,
    ):
        outcome = apply_and_ack_queue_item(_settings(search_backend="opensearch"), item)

    assert mock_ack.call_args.kwargs["result"] == "failed"
    assert "boom" in mock_ack.call_args.kwargs["error"]
    assert outcome["action"] == "failed"


def test_reconcile_once_polls_studies_and_citations_together():
    items = [_queue_item("WLD_2021_TEST_v01"), _citation_item(7)]
    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=items) as mock_list,
        patch("nada_ai.ingest.search_index_sync.apply_and_ack_queue_item") as mock_apply,
    ):
        mock_apply.return_value = {"action": "indexed", "ack_conflict": False}
        summary = reconcile_once(_settings(search_backend="opensearch"))

    assert "object_type" not in mock_list.call_args.kwargs
    assert summary["polled"] == 2 and summary["indexed"] == 2
