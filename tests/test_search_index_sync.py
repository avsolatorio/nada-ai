"""Tests for the NADA search-index change-queue reconciliation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from nada_ai.ingest.search_index_sync import (
    QueueItemChanged,
    QueuePage,
    SearchIndexQueueItem,
    SearchIndexSyncError,
    ack_item,
    get_status,
    list_diff_missing,
    list_diff_stale,
    list_queue,
    list_type_breakdown,
    process_item,
    reconcile_diff_once,
    reconcile_once,
    report_state_bulk,
)
from nada_ai.nada.gate import NadaUnavailable
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
        items = list_queue(_settings()).items
    assert len(items) == 1
    assert (items[0].object_id, items[0].object_key) == (10, "WLD_2021_TEST_v01")
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


#: What the ops return when everything they were asked to write was written.
_INDEXED_OK = {"indexed": 1, "errors": [], "load_errors": [], "empty_docs": []}
_VARIABLES_OK = {"sid": 1, "indexed": 3, "errors": [], "cancelled": False}
_CITATION_OK = {"citation_id": 7, "indexed": 1, "deleted": 0, "errors": []}


def _queue_item(
    sid: int, *, delete: bool = False, variables: bool = False, data_type: str | None = None, idno: str | None = None
) -> SearchIndexQueueItem:
    change_class = "delete" if delete else ("variables" if variables else "upsert_full")
    return SearchIndexQueueItem(
        id=sid,
        object_type="survey",
        object_id=sid,
        object_key=idno or f"IDNO-{sid}",
        change_class=change_class,
        status="pending",
        changed=1700000000,
        data_type=data_type,
        fetch_document=not delete,
    )


def _full(sid: int, title: str = "T") -> dict:
    return {"core_fields": {"catalog_id": sid}, "filters": {"dataset_type": "document"}, "metadata": {"title": title}}


@pytest.fixture(autouse=True)
def _full_records_by_id():
    """A full index fetches the study's record by id: no NADA here."""
    with patch(
        "nada_ai.ingest.search_index_sync.full_record", side_effect=lambda settings, sid: _full(sid)
    ) as full_record:
        yield full_record


def test_a_full_index_reads_the_record_fetched_by_id_not_by_idno(_full_records_by_id):
    item = _queue_item(516, data_type="document", idno="gov.noaa.ncdc:C00994")
    with patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value=_INDEXED_OK) as mock_index:
        process_item(_settings(), item)

    _full_records_by_id.assert_called_once_with(_settings(), 516)
    assert mock_index.call_args.args[1] == [516]
    assert mock_index.call_args.kwargs["records"] == [_full(516)]  # indexed from this record: nothing by idno
    assert mock_index.call_args.kwargs["force"] is False  # unchanged chunks keep their vectors


def test_a_full_index_uses_the_record_the_scheduler_already_fetched(_full_records_by_id):
    record = _full(1, "from the page")
    with patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value=_INDEXED_OK) as mock_index:
        process_item(_settings(), _queue_item(1, data_type="document"), extract=record)
    _full_records_by_id.assert_not_called()
    assert mock_index.call_args.kwargs["records"] == [record]


def test_a_partial_record_from_the_scheduler_is_fetched_again_in_full(_full_records_by_id):
    """A record fetched without metadata (for a partial update) cannot be indexed in full."""
    with patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value=_INDEXED_OK) as mock_index:
        process_item(_settings(), _queue_item(1), extract={"core_fields": {"catalog_id": 1}, "filters": {}})
    _full_records_by_id.assert_called_once()
    assert mock_index.call_args.kwargs["records"] == [_full(1)]


def _page(items: list[SearchIndexQueueItem]) -> QueuePage:
    return QueuePage(items=items, total=len(items), tracking_enabled=True)


def test_list_queue_sends_exclude_and_reads_the_refusal_reason():
    payload = {"status": "success", "tracking_enabled": False, "reason": "other_nada_ai", "total": 0, "items": []}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        page = list_queue(_settings(), exclude=[3, 1])
    assert client.get.call_args.kwargs["params"]["exclude"] == "3,1"
    assert page.tracking_enabled is False
    assert page.reason == "other_nada_ai"


def test_a_queue_item_needs_no_idno():
    item = SearchIndexQueueItem(
        id=1, object_type="survey", object_id=5, change_class="delete", status="pending", changed=1
    )
    assert item.object_key is None


def test_a_delete_item_removes_the_study_by_its_id():
    item = _queue_item(42, delete=True)
    with (
        patch("nada_ai.ingest.search_index_sync.index_sids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.delete_by_sids_op") as mock_delete,
    ):
        action = process_item(_settings(), item)

    assert mock_delete.call_args.args[1] == [42]
    mock_index.assert_not_called()
    assert action == "deleted"


@pytest.mark.parametrize(
    ("report", "reason"),
    [
        ({"load_errors": [{"sid": 1, "error": "metadata 404"}]}, "metadata 404"),
        ({"load_errors": [{"sid": 1, "error": "no ingest path for dataset type 'x'"}]}, "no ingest path"),
        ({"empty_docs": [{"sid": 1, "reason": "no_langdocs"}]}, "no documents produced (no_langdocs)"),
        ({"indexed": 0, "errors": [{"sid": 1, "error": "disk full"}]}, "disk full"),
        ({"variables": {"indexed": 2, "errors": [{}], "failed": {1: "boom"}}}, "variables: boom"),
    ],
)
def test_a_full_item_whose_index_reports_no_success_raises(report, reason):
    """index_sids_op reports a per-study failure in its return value rather than raising. Acking such an item
    indexed would drop it from the queue and overwrite its failed state in NADA, so nothing would ever retry it."""
    item = _queue_item(1, data_type="survey")
    with patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value={**_INDEXED_OK, **report}):
        with pytest.raises(SearchIndexSyncError, match=reason.split(" (")[0]):
            process_item(_settings(), item)


def test_reconcile_once_processes_a_page_and_acks_it_in_one_batch():
    items = [
        _queue_item(1, data_type="timeseries"),
        _queue_item(2, delete=True),
        _queue_item(3, data_type="timeseries"),
    ]

    def index(settings, sids, **kw):
        if sids == [3]:
            raise RuntimeError("boom")
        return _INDEXED_OK

    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=_page(items)),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", side_effect=index),
        patch("nada_ai.ingest.search_index_sync.delete_by_sids_op"),
        patch("nada_ai.ingest.search_index_sync.ack_batch") as mock_ack,
    ):
        summary = reconcile_once(_settings(), limit=10)

    acks = mock_ack.call_args.args[1]
    assert [(a["id"], a["result"]) for a in acks] == [(1, "indexed"), (2, "indexed"), (3, "failed")]
    assert "boom" in acks[2]["error"]
    assert summary == {"polled": 3, "indexed": 1, "deleted": 1, "failed": 1, "not_processed": 0}


def test_reconcile_once_stops_when_nada_cannot_serve_and_acks_what_was_done():
    items = [_queue_item(1, data_type="timeseries"), _queue_item(2, data_type="timeseries")]

    def index(settings, sids, **kw):
        if sids == [2]:
            raise NadaUnavailable("GET /studies: timeout")
        return _INDEXED_OK

    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=_page(items)),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", side_effect=index),
        patch("nada_ai.ingest.search_index_sync.ack_batch") as mock_ack,
    ):
        summary = reconcile_once(_settings(), limit=10)

    assert [a["id"] for a in mock_ack.call_args.args[1]] == [1]
    assert summary["not_processed"] == 1


def test_reconcile_once_does_nothing_while_nada_does_not_track_for_this_deployment():
    page = QueuePage(items=[], tracking_enabled=False, reason="not_tracking_for_nada_ai")
    with (
        patch("nada_ai.ingest.search_index_sync.list_queue", return_value=page),
        patch("nada_ai.ingest.search_index_sync.ack_batch") as mock_ack,
    ):
        summary = reconcile_once(_settings(), limit=10)
    mock_ack.assert_not_called()
    assert summary["reason"] == "not_tracking_for_nada_ai"


# ---------------------------------------------------------------------------
# change_class="variables": only the variable index syncs, not a full
# study/chunk reindex (NADA's Dataset_microdata_model::index_variable_data()
# fires this after a DDI re-import — nothing else about the study changed)
# ---------------------------------------------------------------------------


def test_variables_change_class_syncs_only_variables_on_opensearch():
    item = _queue_item(1, variables=True)
    with (
        patch("nada_ai.ingest.search_index_sync.index_sids_op") as mock_index,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op", return_value=_VARIABLES_OK) as mock_sync,
    ):
        action = process_item(_settings(search_backend="opensearch"), item)

    mock_index.assert_not_called()
    mock_sync.assert_called_once_with(_settings(search_backend="opensearch"), 1)
    assert action == "indexed"


def test_variables_change_class_is_a_noop_on_qdrant():
    """Qdrant has no separate variable index; the item is still acked (not left pending forever)."""
    item = _queue_item(1, variables=True)
    with (
        patch("nada_ai.ingest.search_index_sync.index_sids_op") as mock_index,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op", return_value=_VARIABLES_OK) as mock_sync,
    ):
        action = process_item(_settings(search_backend="qdrant"), item)

    mock_index.assert_not_called()
    mock_sync.assert_not_called()
    assert action == "indexed"


def test_variables_change_class_raises_when_sync_reports_write_errors():
    item = _queue_item(1, variables=True)
    with patch(
        "nada_ai.ingest.variables_index.sync_survey_variables_op",
        return_value={**_VARIABLES_OK, "indexed": 2, "errors": [{"index": {"error": "version_conflict"}}]},
    ):
        with pytest.raises(SearchIndexSyncError, match="version_conflict"):
            process_item(_settings(search_backend="opensearch"), item)


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
    payload = {"status": "success", "items": [{"object_id": 5, "idno": "A", "type": "survey"}], "total": 5}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        page = list_diff_missing(_settings(), object_type="survey", limit=10, offset=0)
    assert page.total == 5
    assert (page.items[0].object_id, page.items[0].idno) == (5, "A")
    assert page.items[0].type == "survey"
    assert client.get.call_args.kwargs["params"] == {"object_type": "survey", "limit": 10, "offset": 0}


def test_list_diff_missing_parses_last_error():
    payload = {
        "status": "success",
        "items": [
            {"object_id": 1, "idno": "A", "type": "survey", "last_error": "boom"},
            {"object_id": 2, "idno": "B", "type": "survey"},
        ],
        "total": 2,
    }
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        page = list_diff_missing(_settings(), object_type="survey", limit=10, offset=0)
    assert page.items[0].last_error == "boom"
    assert page.items[1].last_error is None


def test_list_diff_missing_total_can_exceed_items_when_nothing_indexed():
    """Empty catalog coverage must still report the real total, not look like 'nothing missing'."""
    payload = {"status": "success", "items": [{"object_id": 1, "idno": "A", "type": "survey"}], "total": 8000}
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
    payload = {"status": "success", "items": [{"object_id": 9, "idno": "Z"}], "total": 1}
    client = _mock_sync_client(get=_resp(payload))
    with patch("nada_ai.ingest.search_index_sync.httpx.Client", return_value=client):
        page = list_diff_stale(_settings(), object_type="survey")
    assert page.total == 1
    assert page.items[0].object_id == 9
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


def _diff_settings(**overrides) -> Settings:
    """reconcile_diff_once refuses to run without state reporting (see its docstring)."""
    return _settings(report_search_index_state_enabled=True, **overrides)


def _diff_page(items, total=None):
    from nada_ai.ingest.search_index_sync import DiffItem, DiffPage

    return DiffPage(items=[DiffItem(**i) for i in items], total=total if total is not None else len(items))


def _indexed_sids(mock_index) -> list[int]:
    return [c.args[1][0] for c in mock_index.call_args_list]


def test_reconcile_diff_once_indexes_each_missing_study_by_its_id():
    page = _diff_page([{"object_id": 1, "type": "survey"}, {"object_id": 2, "type": "geospatial"}])
    empty_page = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[page, empty_page]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value=_INDEXED_OK) as mock_index,
    ):
        summary = reconcile_diff_once(_diff_settings())

    assert _indexed_sids(mock_index) == [1, 2]
    assert all(c.kwargs["force"] is True for c in mock_index.call_args_list)
    assert summary["missing_total"] == 2
    assert summary["indexed"] == 2
    assert summary["failed"] == 0
    assert summary["skipped"] == 0


def test_reconcile_diff_once_counts_soft_failure_as_failed_not_indexed():
    """index_sids_op doesn't raise for a per-study failure (load error, empty doc, backend write error) — it reports
    those in its return value. A call that returns without raising must not be assumed to mean the study was
    actually indexed."""
    page = _diff_page([{"object_id": 1, "type": "geospatial"}])
    empty_page = _diff_page([])
    report = {**_INDEXED_OK, "indexed": 0, "load_errors": [{"sid": 1, "stage": "load", "error": "boom"}]}
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[page, empty_page]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value=report),
    ):
        summary = reconcile_diff_once(_diff_settings())

    assert summary["indexed"] == 0
    assert summary["failed"] == 1


def test_reconcile_diff_once_counts_a_study_whose_variables_failed_as_failed():
    page = _diff_page([{"object_id": 1, "type": "survey"}])
    report = {**_INDEXED_OK, "variables": {"indexed": 0, "errors": [{}], "failed": {1: "boom"}}}
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[page, _diff_page([])]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=_diff_page([])),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value=report),
    ):
        summary = reconcile_diff_once(_diff_settings())

    assert summary["indexed"] == 0
    assert summary["failed"] == 1


def test_reconcile_diff_once_passes_data_type_through_to_both_diff_calls():
    empty_page = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", return_value=empty_page) as mock_missing,
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page) as mock_stale,
    ):
        reconcile_diff_once(_diff_settings(), data_type="geospatial")

    assert mock_missing.call_args.kwargs["data_type"] == "geospatial"
    assert mock_stale.call_args.kwargs["data_type"] == "geospatial"


def test_reconcile_diff_once_skips_unmapped_dataset_type():
    page = _diff_page([{"object_id": 1, "type": "citation"}])  # 'citation' has no metadata_type mapping
    empty_page = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[page, empty_page]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
        patch("nada_ai.ingest.search_index_sync.index_sids_op") as mock_index,
    ):
        summary = reconcile_diff_once(_diff_settings())

    mock_index.assert_not_called()
    assert summary["skipped"] == 1
    assert summary["indexed"] == 0


def test_reconcile_diff_once_deletes_stale_studies_by_id_in_one_batch_call():
    stale_page = _diff_page([{"object_id": 8, "idno": "SAME"}, {"object_id": 9, "idno": "SAME"}])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", return_value=_diff_page([])),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", side_effect=[stale_page, _diff_page([])]),
        patch("nada_ai.ingest.search_index_sync.delete_by_sids_op") as mock_delete,
    ):
        summary = reconcile_diff_once(_diff_settings())

    mock_delete.assert_called_once()
    assert mock_delete.call_args.args[1] == [8, 9]  # two studies that shared an idno are two studies
    assert summary["stale_total"] == 2
    assert summary["deleted"] == 2


def test_reconcile_diff_once_remove_only_deletes_stale_and_indexes_nothing():
    stale_page = _diff_page([{"object_id": 8}])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing") as mock_missing,
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", side_effect=[stale_page, _diff_page([])]),
        patch("nada_ai.ingest.search_index_sync.index_sids_op") as mock_index,
        patch("nada_ai.ingest.search_index_sync.delete_by_sids_op") as mock_delete,
    ):
        summary = reconcile_diff_once(_diff_settings(), remove_only=True)

    mock_missing.assert_not_called()
    mock_index.assert_not_called()
    assert mock_delete.call_args.args[1] == [8]
    assert summary["missing_total"] == 0
    assert summary["deleted"] == 1


def test_reconcile_diff_once_terminates_when_an_item_keeps_failing():
    """The 'missing' diff is re-fetched live every iteration (the set shrinks
    as items succeed) — an item that keeps failing stays in NADA's diff
    forever (correctly: it's genuinely still not indexed), so this must not
    loop forever retrying it. Simulates 200 re-fetches all returning the exact
    same permanently-broken item."""
    stuck_page = _diff_page([{"object_id": 1, "type": "survey"}])
    empty_page = _diff_page([])

    with (
        patch(
            "nada_ai.ingest.search_index_sync.list_diff_missing",
            side_effect=[stuck_page] * 200 + [empty_page],
        ),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", side_effect=RuntimeError("permanently broken")),
    ):
        summary = reconcile_diff_once(_diff_settings())

    # Attempted exactly once despite appearing on every re-fetch, and terminated.
    assert summary["failed"] == 1
    assert summary["indexed"] == 0


def test_reconcile_diff_once_summary_shape():
    empty_page = _diff_page([])
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", return_value=empty_page),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty_page),
    ):
        summary = reconcile_diff_once(_diff_settings())

    assert summary == {
        "missing_total": 0,
        "stale_total": 0,
        "indexed": 0,
        "deleted": 0,
        "failed": 0,
        "skipped": 0,
    }


def test_reconcile_diff_once_reports_progress_per_study():
    """Jobs reads progress.total/processed/percent/failed/current_sid — the
    same snapshot shape ingest already emits via IngestProgressTracker."""
    missing = _diff_page([{"object_id": 1, "type": "survey"}, {"object_id": 2, "type": "geospatial"}])
    stale = _diff_page([{"object_id": 9}])
    empty = _diff_page([])
    snapshots: list[dict] = []

    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[missing, empty]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", side_effect=[stale, empty]),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value=_INDEXED_OK),
        patch("nada_ai.ingest.search_index_sync.delete_by_sids_op"),
    ):
        summary = reconcile_diff_once(_diff_settings(), progress_cb=snapshots.append)

    assert summary["indexed"] == 2
    assert summary["deleted"] == 1
    assert snapshots[0] == {
        "processed": 0,
        "total": 3,
        "failed": 0,
        "current_sid": None,
        "percent": 0.0,
        "phase": "missing",
    }
    assert [s["current_sid"] for s in snapshots[1:]] == [1, 2, 9]
    assert snapshots[-1]["processed"] == 3
    assert snapshots[-1]["percent"] == 100.0
    assert snapshots[-1]["phase"] == "stale"
    assert all("total" in s and "failed" in s for s in snapshots)


def test_reconcile_diff_once_stops_when_cancel_token_is_set():
    from nada_ai.ingest.progress import CancelToken

    missing = _diff_page([{"object_id": 1, "type": "survey"}, {"object_id": 2, "type": "geospatial"}])
    empty = _diff_page([])
    token = CancelToken()
    indexed: list[int] = []

    def index_one(settings, sids, **kwargs):
        indexed.append(sids[0])
        token.set()
        return _INDEXED_OK

    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=[missing, empty]),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=empty),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", side_effect=index_one),
    ):
        summary = reconcile_diff_once(_diff_settings(), cancel_token=token)

    assert indexed == [1]
    assert summary["indexed"] == 1
    assert summary["cancelled"] is True
    assert summary["deleted"] == 0


def test_reconcile_diff_once_refuses_to_run_without_state_reporting():
    """NADA's diff only changes when nada-ai reports what it indexed; without that the run would stop after one
    page and still report success."""
    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing") as mock_missing,
        pytest.raises(SearchIndexSyncError, match="NADA_REPORT_SEARCH_INDEX_STATE_ENABLED"),
    ):
        reconcile_diff_once(_settings(report_search_index_state_enabled=False))

    mock_missing.assert_not_called()


class _FakeDiff:
    """NADA's diff as a live list: fetched by offset/limit, and an item leaves it once handled."""

    def __init__(self, items):
        self.items = list(items)

    def fetch(self, settings, *, object_type, limit, offset, data_type):
        return _diff_page(self.items[offset : offset + limit], total=len(self.items))

    def remove(self, sids):
        self.items = [it for it in self.items if it["object_id"] not in sids]


def test_reconcile_diff_once_gets_past_a_full_page_of_items_that_keep_failing():
    """A page whose items all stay in the diff (failed, or skipped for an unmapped type) must not hide the rest."""
    diff = _FakeDiff(
        [
            {"object_id": 1, "type": "survey"},  # broken
            {"object_id": 2, "type": "citation"},  # unmapped
            *({"object_id": 10 + i, "type": "survey"} for i in range(5)),
        ]
    )

    def index_one(settings, sids, **kwargs):
        if sids == [1]:
            return {**_INDEXED_OK, "indexed": 0, "load_errors": [{"sid": 1, "error": "boom"}]}
        diff.remove(sids)
        return _INDEXED_OK

    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=diff.fetch),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=_diff_page([])),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", side_effect=index_one) as mock_index,
    ):
        summary = reconcile_diff_once(_diff_settings(), page_size=2)

    assert sorted(_indexed_sids(mock_index)) == [1, 10, 11, 12, 13, 14]
    assert summary["indexed"] == 5
    assert summary["failed"] == 1
    assert summary["skipped"] == 1
    assert [it["object_id"] for it in diff.items] == [1, 2]


def test_reconcile_diff_once_gets_past_a_full_page_of_stale_items_that_fail_to_delete():
    diff = _FakeDiff([{"object_id": i} for i in (1, 2, 3, 4, 5)])

    def delete(settings, sids):
        if 1 in sids:
            raise RuntimeError("delete failed")
        diff.remove(sids)

    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", return_value=_diff_page([])),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", side_effect=diff.fetch),
        patch("nada_ai.ingest.search_index_sync.delete_by_sids_op", side_effect=delete),
    ):
        summary = reconcile_diff_once(_diff_settings(), page_size=2)

    assert summary["deleted"] == 3
    assert summary["failed"] == 2
    assert [it["object_id"] for it in diff.items] == [1, 2]


def test_reconcile_diff_once_moves_past_an_item_that_stays_after_a_reported_success():
    """A success whose state report never reached NADA leaves the item in the diff; it is fetched again, recognised
    as seen, and stepped over rather than stopping the run."""
    diff = _FakeDiff([{"object_id": i, "type": "survey"} for i in range(4)])

    def index_one(settings, sids, **kwargs):
        if sids != [0]:  # study 0's report is lost: it stays in the diff
            diff.remove(sids)
        return _INDEXED_OK

    with (
        patch("nada_ai.ingest.search_index_sync.list_diff_missing", side_effect=diff.fetch),
        patch("nada_ai.ingest.search_index_sync.list_diff_stale", return_value=_diff_page([])),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", side_effect=index_one) as mock_index,
    ):
        summary = reconcile_diff_once(_diff_settings(), page_size=2)

    assert _indexed_sids(mock_index) == [0, 1, 2, 3]
    assert summary["indexed"] == 4


# ---------------------------------------------------------------------------
# change_class="upsert_partial": a change of the study's options (publish state, license, ...). On OpenSearch the
# flags are applied in place (no metadata read, no embedding); a study that is not indexed yet, or another engine,
# gets the full index.
# ---------------------------------------------------------------------------


def _partial_item(sid: int) -> SearchIndexQueueItem:
    return _queue_item(sid, data_type="survey").model_copy(update={"change_class": "upsert_partial"})


def test_partial_change_class_is_applied_in_place_from_the_fetched_extract():
    extract = {"core_fields": {"catalog_id": 1, "idno": "WLD_2021_TEST_v01"}, "filters": {"published": 1}}
    with (
        patch("nada_ai.ingest.service.apply_study_options_from_extract", return_value={"updated": True}) as mock_apply,
        patch("nada_ai.ingest.service.apply_study_options_op") as mock_fetching,
        patch("nada_ai.ingest.search_index_sync.index_sids_op") as mock_index,
    ):
        action = process_item(_settings(search_backend="opensearch"), _partial_item(1), extract=extract)

    assert mock_apply.call_args.args[1:] == (1, extract["core_fields"], {"published": 1})
    mock_fetching.assert_not_called()
    mock_index.assert_not_called()
    assert action == "indexed"


def test_partial_change_class_fetches_the_extract_by_id_when_not_given():
    with (
        patch("nada_ai.ingest.service.apply_study_options_op", return_value={"updated": True}) as mock_publish,
        patch("nada_ai.ingest.search_index_sync.index_sids_op") as mock_index,
    ):
        process_item(_settings(search_backend="opensearch"), _partial_item(4))

    assert mock_publish.call_args.args[1] == 4
    mock_index.assert_not_called()


def test_partial_change_class_indexes_in_full_when_the_study_is_not_indexed_yet():
    with (
        patch(
            "nada_ai.ingest.service.apply_study_options_op", return_value={"updated": False, "reason": "not_indexed"}
        ),
        patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value=_INDEXED_OK) as mock_index,
    ):
        action = process_item(_settings(search_backend="opensearch"), _partial_item(4))

    assert mock_index.call_args.args[1] == [4]
    assert action == "indexed"


def test_partial_change_class_indexes_in_full_on_qdrant():
    with (
        patch("nada_ai.ingest.service.apply_study_options_op") as mock_publish,
        patch("nada_ai.ingest.search_index_sync.index_sids_op", return_value=_INDEXED_OK) as mock_index,
    ):
        process_item(_settings(search_backend="qdrant"), _partial_item(4))

    mock_publish.assert_not_called()
    mock_index.assert_called_once()


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
        patch("nada_ai.ingest.citations_index.sync_citation_op", return_value=_CITATION_OK) as mock_sync,
        patch("nada_ai.ingest.search_index_sync.index_sids_op") as mock_index,
    ):
        action = process_item(_settings(search_backend="opensearch"), item)

    mock_sync.assert_called_once_with(_settings(search_backend="opensearch"), 7)
    mock_index.assert_not_called()
    assert action == "indexed"


def test_a_deleted_citation_is_removed_from_the_index():
    item = _citation_item(7, delete=True)
    with (
        patch("nada_ai.ingest.citations_index.delete_citation_op") as mock_delete,
        patch("nada_ai.ingest.citations_index.sync_citation_op") as mock_sync,
        patch("nada_ai.ingest.search_index_sync.delete_by_sids_op") as mock_study_delete,
    ):
        action = process_item(_settings(search_backend="opensearch"), item)

    mock_delete.assert_called_once_with(_settings(search_backend="opensearch"), 7)
    mock_sync.assert_not_called()
    mock_study_delete.assert_not_called()
    assert action == "deleted"


def test_a_citation_item_is_handled_without_work_on_qdrant():
    """Qdrant has no citation index; the item is still acked rather than left pending forever."""
    item = _citation_item(7)
    with patch("nada_ai.ingest.citations_index.sync_citation_op") as mock_sync:
        action = process_item(_settings(search_backend="qdrant"), item)

    mock_sync.assert_not_called()
    assert action == "indexed"


def test_a_citation_sync_with_write_errors_raises():
    item = _citation_item(7)
    with patch(
        "nada_ai.ingest.citations_index.sync_citation_op",
        return_value={**_CITATION_OK, "indexed": 0, "errors": [{"index": {"error": "mapper_parsing_exception"}}]},
    ):
        with pytest.raises(SearchIndexSyncError, match="mapper_parsing_exception"):
            process_item(_settings(search_backend="opensearch"), item)
