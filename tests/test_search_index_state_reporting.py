"""Tests for reporting indexing/deletion outcomes to NADA's search_index_state
(service._report_state_bulk_best_effort), wired into index_sids_op,
index_from_catalog_op and delete_by_sids_op. Every report is keyed by ``object_id`` (the study's sid).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nada_ai.settings import Settings


def _settings(**overrides) -> Settings:
    overrides.setdefault("report_search_index_state_enabled", True)
    return Settings(**overrides)


def _record(sid: int, dataset_type: str = "timeseries", idno: str | None = None) -> dict:
    """A full extract record, as NADA's metadata-extract API sends it."""
    idno = idno or f"IDNO-{sid}"
    return {
        "core_fields": {"catalog_id": sid, "idno": idno, "all_text": []},
        "filters": {"dataset_type": dataset_type},
        "metadata": {"idno": idno},
    }


def _fetch(records: dict[int, dict]):
    """A fake ``fetch_studies_by_sids`` serving ``records`` by sid."""

    def fetch(settings, sids, **kwargs):
        return {
            "studies": [records[s] for s in sids if s in records],
            "missing": [s for s in sids if s not in records],
            "errors": [],
        }

    return fetch


def _bulk(*, failing=(), empty=(), write_errors=None):
    """A fake ``run_bulk_index``: consumes the studies; ``failing`` sids fail to load, ``empty`` produce nothing."""

    def fn(settings, sources, **kwargs):
        load_errors, empty_docs, progress = kwargs.get("load_errors"), kwargs.get("empty_docs"), kwargs.get("progress")
        n = 0
        for source in sources:
            if source.sid in failing:
                load_errors.append({"sid": source.sid, "idno": source.idno, "stage": "load", "error": "boom"})
            elif source.sid in empty:
                empty_docs.append({"sid": source.sid, "idno": source.idno, "reason": "no_langdocs"})
            else:
                n += 1
            if progress is not None:
                progress.mark(source.sid, ok=source.sid not in failing)
        return n, list(write_errors) if write_errors else None

    return fn


def _index(settings, sids, records, bulk, **kwargs):
    import nada_ai.ingest.service as service_module

    with (
        patch("nada_ai.ingest.study_sources.fetch_studies_by_sids", _fetch(records)),
        patch.object(service_module, "run_bulk_index", bulk),
    ):
        return service_module.index_sids_op(settings, sids, show_progress_bar=False, **kwargs)


def _reported(mock_report) -> dict[int, tuple[str, str | None]]:
    items = mock_report.call_args.args[1]
    assert all(i["object_type"] == "survey" for i in items)
    return {i["object_id"]: (i["status"], i.get("error")) for i in items}


def _reported_any(mock_report) -> dict[int, tuple[str, str | None]]:
    items = mock_report.call_args.args[1]
    return {i["object_id"]: (i["status"], i.get("error")) for i in items}


# ---------------------------------------------------------------------------
# _error_by_sid
# ---------------------------------------------------------------------------


def test_error_by_sid_prefers_load_error_over_empty_reason():
    import nada_ai.ingest.service as service_module

    load_errors = [{"sid": 1, "error": "boom"}]
    empty_docs = [{"sid": 1, "reason": "no_langdocs"}, {"sid": 2, "reason": "all_empty_content"}]

    result = service_module._error_by_sid(load_errors, empty_docs)

    assert result == {1: "boom", 2: "no documents produced (all_empty_content)"}


# ---------------------------------------------------------------------------
# _report_state_bulk_best_effort itself
# ---------------------------------------------------------------------------


def test_report_state_bulk_best_effort_noop_when_disabled():
    import nada_ai.ingest.service as service_module

    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module._report_state_bulk_best_effort(
            _settings(report_search_index_state_enabled=False),
            [{"object_type": "survey", "object_id": 1, "status": "indexed"}],
        )
    mock_report.assert_not_called()


def test_report_state_bulk_best_effort_noop_when_no_items():
    import nada_ai.ingest.service as service_module

    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module._report_state_bulk_best_effort(_settings(), [])
    mock_report.assert_not_called()


def test_report_state_bulk_best_effort_calls_through_when_enabled():
    import nada_ai.ingest.service as service_module

    settings = _settings()
    items = [{"object_type": "survey", "object_id": 1, "status": "indexed"}]
    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module._report_state_bulk_best_effort(settings, items)
    mock_report.assert_called_once_with(settings, items)


def test_report_state_bulk_best_effort_swallows_errors():
    import nada_ai.ingest.service as service_module

    with patch("nada_ai.ingest.search_index_sync.report_state_bulk", side_effect=RuntimeError("network down")):
        # Must not raise.
        service_module._report_state_bulk_best_effort(
            _settings(), [{"object_type": "survey", "object_id": 1, "status": "indexed"}]
        )


# ---------------------------------------------------------------------------
# index_sids_op
# ---------------------------------------------------------------------------


def test_index_sids_op_reports_success_and_failure_separately_by_sid():
    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        _index(_settings(), [1, 2], {1: _record(1), 2: _record(2)}, _bulk(failing={2}))

    assert _reported(mock_report) == {1: ("indexed", None), 2: ("failed", "boom")}
    items = mock_report.call_args.args[1]
    assert {i["object_id"]: i.get("object_key") for i in items} == {1: "IDNO-1", 2: "IDNO-2"}


def test_index_sids_op_reports_a_study_nada_does_not_have_as_failed():
    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        result = _index(_settings(), [1, 9], {1: _record(1)}, _bulk())

    assert _reported(mock_report) == {1: ("indexed", None), 9: ("failed", "not in NADA's catalog")}
    assert result["load_errors"][0]["stage"] == "fetch"


def test_a_cancelled_run_reports_only_the_studies_it_got_through(tmp_path, monkeypatch):
    """Like the pipeline: it takes the next study, sees the cancel and stops without loading it. That study is not
    reported indexed (a reconcile picks it up)."""
    from nada_ai.ingest.progress import CancelToken

    token = CancelToken()

    def bulk(settings, sources, **kwargs):
        for source in sources:
            if kwargs["cancel_token"].is_set():
                break
            if source.sid == 1:
                token.set()  # cancelled while study 1 is in hand; it is still written
        return 1, None

    records = [_record(1), _record(2), _record(3)]
    result, _, mock_report = _run_catalog(
        tmp_path, monkeypatch, records, settings=_settings(search_backend="opensearch"), bulk=bulk, cancel_token=token
    )
    assert result["cancelled"] is True
    assert _reported_any(mock_report) == {1: ("indexed", None)}


def test_applying_study_options_says_why_nada_could_not_build_the_extract():
    import nada_ai.ingest.service as service_module

    batch = {"studies": [], "missing": [], "errors": [{"catalog_id": 7, "message": "metadata is not valid JSON"}]}
    with (
        patch("nada_ai.ingest.study_sources.fetch_studies_by_sids", return_value=batch),
        pytest.raises(ValueError, match="could not build the extract of study 7: metadata is not valid JSON"),
    ):
        service_module.apply_study_options_op(_settings(search_backend="opensearch"), 7)


def test_index_sids_op_reports_a_record_with_no_ingest_path_as_failed():
    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        _index(_settings(), [1], {1: _record(1, dataset_type="unknown-type")}, _bulk())

    assert _reported(mock_report) == {1: ("failed", "no ingest path for dataset type 'unknown-type'")}


def test_index_sids_op_does_not_report_when_disabled():
    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        _index(_settings(report_search_index_state_enabled=False), [1], {1: _record(1)}, _bulk())
    mock_report.assert_not_called()


def test_index_sids_op_uses_records_the_caller_already_has():
    import nada_ai.ingest.service as service_module

    with (
        patch("nada_ai.ingest.study_sources.fetch_studies_by_sids") as fetch,
        patch.object(service_module, "run_bulk_index", _bulk()),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.index_sids_op(_settings(), [1], show_progress_bar=False, records=[_record(1)])

    fetch.assert_not_called()
    assert _reported(mock_report) == {1: ("indexed", None)}


# ---------------------------------------------------------------------------
# index_sids_op also syncs variables of microdata studies (a full study index is study + chunks + variables
# together — see docs/variables-search-contract.md)
# ---------------------------------------------------------------------------


def test_index_sids_op_syncs_variables_of_microdata_studies_only():
    records = {1: _record(1, "survey"), 2: _record(2, "document"), 3: _record(3, "survey")}
    with (
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        mock_sync.return_value = {"indexed": 3, "errors": []}
        result = _index(_settings(search_backend="opensearch"), [1, 2, 3], records, _bulk())

    assert [c.args[1] for c in mock_sync.call_args_list] == [1, 3]
    assert result["variables"] == {"indexed": 6, "errors": [], "failed": {}}


def test_index_sids_op_skips_variables_of_a_study_that_failed_to_load():
    records = {1: _record(1, "survey"), 2: _record(2, "survey")}
    with (
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        mock_sync.return_value = {"indexed": 1, "errors": []}
        _index(_settings(search_backend="opensearch"), [1, 2], records, _bulk(failing={2}))

    assert [c.args[1] for c in mock_sync.call_args_list] == [1]


def test_index_sids_op_does_not_sync_variables_for_qdrant():
    """Variable search is OpenSearch-only; a qdrant deployment has no separate variable index to sync."""
    with (
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        result = _index(_settings(search_backend="qdrant"), [1], {1: _record(1, "survey")}, _bulk())

    mock_sync.assert_not_called()
    assert result["variables"] is None


def test_index_sids_op_variable_sync_failure_does_not_fail_the_call():
    with (
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op", side_effect=RuntimeError("boom")),
    ):
        result = _index(_settings(search_backend="opensearch"), [1], {1: _record(1, "survey")}, _bulk())

    assert result["indexed"] == 1  # the study/chunk indexing itself still succeeded
    assert result["variables"]["errors"] == [{"sid": 1, "error": "boom"}]


def test_index_sids_op_reports_a_study_whose_variables_failed_as_failed():
    """A full index of a study is study + chunks + variables: a study whose variables did not all land is not
    indexed, or NADA drops it from its "missing" diff and nothing retries its variables."""

    def sync(settings, sid, **_):
        return {"indexed": 1, "errors": [{"index": {"status": 400}}] if sid == 1 else []}

    records = {1: _record(1, "survey"), 2: _record(2, "survey")}
    with (
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op", side_effect=sync),
    ):
        result = _index(_settings(search_backend="opensearch"), [1, 2], records, _bulk())

    assert _reported(mock_report) == {1: ("failed", "variables: 1 write error(s)"), 2: ("indexed", None)}
    assert result["variables"]["failed"] == {1: "1 write error(s)"}


def test_a_cancelled_variable_sync_counts_every_study_not_synced_as_failed():
    import nada_ai.ingest.service as service_module
    from nada_ai.ingest.progress import CancelToken

    token = CancelToken()

    def sync(settings, sid, **_):
        token.set()  # cancelled during the first study, after it finished
        return {"indexed": 1, "errors": []}

    with patch("nada_ai.ingest.variables_index.sync_survey_variables_op", side_effect=sync):
        result = service_module._sync_variables_best_effort(_settings(), [1, 2, 3], cancel_token=token)

    assert result["failed"] == {
        2: "cancelled before its variables were synced",
        3: "cancelled before its variables were synced",
    }


# ---------------------------------------------------------------------------
# delete_by_sids_op
# ---------------------------------------------------------------------------


def test_delete_by_sids_op_reports_deleted_by_object_id():
    import nada_ai.ingest.service as service_module

    with (
        patch.object(service_module, "_delete_qdrant_by_sids", return_value={"backend": "qdrant"}),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.delete_by_sids_op(_settings(search_backend="qdrant"), [5, 6, 5])

    assert mock_report.call_args.args[1] == [
        {"object_type": "survey", "object_id": 5, "status": "deleted"},
        {"object_type": "survey", "object_id": 6, "status": "deleted"},
    ]


def test_delete_by_sids_op_deletes_every_document_of_the_studies_by_sid():
    from unittest.mock import MagicMock

    import nada_ai.ingest.service as service_module

    client = MagicMock()
    client.delete_by_query.return_value = {"deleted": 2, "total": 2}
    settings = _settings(search_backend="opensearch")
    with (
        patch.object(service_module, "build_client", return_value=client),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
    ):
        service_module.delete_by_sids_op(settings, [7])

    queries = {c.kwargs["index"]: c.kwargs["body"]["query"] for c in client.delete_by_query.call_args_list}
    assert queries == {
        settings.index_name: {"terms": {"metadata.sid": [7]}},
        settings.studies_index: {"terms": {"sid": [7]}},
        settings.variables_index: {"terms": {"sid": [7]}},
    }


# ---------------------------------------------------------------------------
# index_from_catalog_op: pages NADA's records by id
# ---------------------------------------------------------------------------


def _run_catalog(tmp_path, monkeypatch, records, *, settings, bulk=None, cancel_token=None, catalog_type="survey"):
    import nada_ai.ingest.service as service_module

    monkeypatch.setenv("NADA_INGEST_CHECKPOINT_DIR", str(tmp_path))

    def page(settings, *, offset, limit, nada_type):
        return {
            "studies": records[offset : offset + limit],
            "total": len(records),
            "has_more": offset + limit < len(records),
        }

    with (
        patch("nada_ai.ingest.study_sources.fetch_study_page", page),
        patch.object(service_module, "run_bulk_index", bulk or _bulk()),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        mock_sync.return_value = {"indexed": 2, "errors": []}
        result = service_module.index_from_catalog_op(
            settings, catalog_type=catalog_type, ps=2, show_progress_bar=False, cancel_token=cancel_token
        )
    return result, mock_sync, mock_report


def test_index_from_catalog_op_pages_every_record_and_reports_by_sid(tmp_path, monkeypatch):
    records = [_record(1, "geospatial"), _record(2, "geospatial"), _record(3, "geospatial")]
    result, _, mock_report = _run_catalog(
        tmp_path, monkeypatch, records, settings=_settings(), bulk=_bulk(empty={2}), catalog_type="geospatial"
    )

    assert result["rows"] == 3
    assert _reported(mock_report) == {
        1: ("indexed", None),
        2: ("failed", "no documents produced (no_langdocs)"),
        3: ("indexed", None),
    }


def test_index_from_catalog_op_syncs_variables_of_microdata_studies_only(tmp_path, monkeypatch):
    records = [_record(1, "survey"), _record(2, "document"), _record(3, "survey")]
    result, mock_sync, _ = _run_catalog(tmp_path, monkeypatch, records, settings=_settings(search_backend="opensearch"))

    assert [c.args[1] for c in mock_sync.call_args_list] == [1, 3]  # never the document
    assert result["variables"] == {"indexed": 4, "errors": [], "failed": {}}


def test_index_from_catalog_op_skips_variables_of_a_study_that_failed_to_load(tmp_path, monkeypatch):
    records = [_record(1, "survey"), _record(2, "survey")]
    _, mock_sync, _ = _run_catalog(
        tmp_path, monkeypatch, records, settings=_settings(search_backend="opensearch"), bulk=_bulk(failing={2})
    )
    assert [c.args[1] for c in mock_sync.call_args_list] == [1]


def test_index_from_catalog_op_does_not_sync_variables_on_qdrant(tmp_path, monkeypatch):
    result, mock_sync, _ = _run_catalog(
        tmp_path, monkeypatch, [_record(1, "survey")], settings=_settings(search_backend="qdrant")
    )
    mock_sync.assert_not_called()
    assert result["variables"] is None


def test_index_from_catalog_op_skips_variables_when_cancelled(tmp_path, monkeypatch):
    from nada_ai.ingest.progress import CancelToken

    token = CancelToken()
    token.set()
    result, mock_sync, _ = _run_catalog(
        tmp_path,
        monkeypatch,
        [_record(1, "survey")],
        settings=_settings(search_backend="opensearch"),
        cancel_token=token,
    )
    mock_sync.assert_not_called()
    assert result["cancelled"] is True


def test_a_cancelled_catalog_run_reports_its_microdata_studies_failed_not_indexed(tmp_path, monkeypatch):
    """Their variables were never synced, so they are not fully indexed: a reconcile must retry them."""
    from nada_ai.ingest.progress import CancelToken

    token = CancelToken()
    inner = _bulk()

    def bulk(settings, sources, **kwargs):
        out = inner(settings, sources, **kwargs)
        token.set()  # cancelled once the studies were written, before the variables
        return out

    records = [_record(1, "survey"), _record(2, "document")]
    _, mock_sync, mock_report = _run_catalog(
        tmp_path, monkeypatch, records, settings=_settings(search_backend="opensearch"), bulk=bulk, cancel_token=token
    )

    mock_sync.assert_not_called()
    assert _reported(mock_report) == {
        1: ("failed", "variables: cancelled before its variables were synced"),
        2: ("indexed", None),
    }


def test_index_from_catalog_op_reports_write_failure_as_failed_not_indexed(tmp_path, monkeypatch):
    records = [_record(1, "geospatial"), _record(2, "geospatial")]
    _, _, mock_report = _run_catalog(
        tmp_path,
        monkeypatch,
        records,
        settings=_settings(),
        bulk=_bulk(write_errors=[{"id": "u1", "sid": 2, "error": "payload rejected"}]),
        catalog_type="geospatial",
    )
    assert _reported(mock_report) == {1: ("indexed", None), 2: ("failed", "write failed: payload rejected")}


# ---------------------------------------------------------------------------
# Backend write errors must never be reported as 'indexed'
# ---------------------------------------------------------------------------


def test_attribute_write_error_qdrant_shape():
    import nada_ai.ingest.service as service_module

    assert service_module._attribute_write_error({"id": "u1", "sid": 4, "error": "rejected"}) == (4, "rejected")


def test_attribute_write_error_opensearch_chunk_and_study_documents():
    import nada_ai.ingest.service as service_module

    chunk = {"index": {"_id": "u1", "status": 400, "error": {"type": "mapper"}, "data": {"metadata": {"sid": 4}}}}
    sid, msg = service_module._attribute_write_error(chunk)
    assert sid == 4 and "mapper" in msg
    study = {"index": {"_id": "4", "status": 400, "error": "boom", "data": {"sid": 4, "idno": "A"}}}
    assert service_module._attribute_write_error(study)[0] == 4


def test_attribute_write_error_unattributable():
    import nada_ai.ingest.service as service_module

    assert service_module._attribute_write_error({"error": "run aborted"})[0] is None
    assert service_module._attribute_write_error({"id": "u1", "sid": None, "error": "x"})[0] is None
    assert service_module._attribute_write_error("plain string")[0] is None
    assert service_module._attribute_write_error({"index": {"error": "x"}})[0] is None


def test_index_sids_op_reports_write_failure_as_failed_not_indexed():
    err = {"index": {"_id": "u1", "status": 400, "error": "boom", "data": {"metadata": {"sid": 2}}}}
    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        _index(_settings(), [1, 2], {1: _record(1), 2: _record(2)}, _bulk(write_errors=[err]))

    assert _reported(mock_report) == {1: ("indexed", None), 2: ("failed", "write failed: boom")}


def test_index_sids_op_reports_nothing_indexed_when_write_error_unattributable():
    # e.g. Qdrant went away mid-run: the writer reports one un-attributable error for the whole run.
    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        _index(_settings(), [1, 2], {1: _record(1), 2: _record(2)}, _bulk(write_errors=[{"error": "refused"}]))

    # Nothing to report at all: they must stay 'missing' in NADA so a reconcile retries them.
    mock_report.assert_not_called()


def test_unattributable_write_error_still_reports_known_load_failures():
    import nada_ai.ingest.service as service_module

    items = service_module._state_report_items(
        {1: "A", 2: "B"},
        load_errors=[{"sid": 2, "error": "boom"}],
        empty_docs=[],
        write_errors=[{"error": "connection refused"}],
    )
    assert items == [{"object_type": "survey", "object_id": 2, "object_key": "B", "status": "failed", "error": "boom"}]


def test_load_error_message_wins_over_write_error_for_same_study():
    import nada_ai.ingest.service as service_module

    items = service_module._state_report_items(
        {1: "A"},
        load_errors=[{"sid": 1, "error": "load boom"}],
        empty_docs=[],
        write_errors=[{"sid": 1, "error": "write boom"}],
    )
    assert [(i["object_id"], i["status"], i["error"]) for i in items] == [(1, "failed", "load boom")]


def test_qdrant_writer_attaches_the_sid_to_write_errors():
    """End to end through QdrantIngestWriter: a point Qdrant rejects is reported with its study's sid."""
    from unittest.mock import MagicMock

    import nada_ai.ingest.qdrant_writer as qw

    settings = _settings(qdrant_sparse_lexical=False)

    def record(sid):
        return (f"uuid-{sid}", [0.1, 0.2], {"page_content": "x", "metadata": {"idno": f"I{sid}", "sid": sid}})

    client = MagicMock()

    def upsert(collection_name, points, wait):
        if any(p.id == "uuid-2" for p in points):
            raise RuntimeError("payload rejected")

    client.upsert.side_effect = upsert
    embedding = MagicMock()
    embedding.embedding_dimension.return_value = 2

    with (
        patch.object(qw, "_client", return_value=client),
        patch.object(qw.QdrantIngestWriter, "ensure_target"),
        patch.object(qw, "iter_langdoc_records", return_value=iter([record(1), record(2)])),
    ):
        success, errors = qw.QdrantIngestWriter(settings).run_bulk([], embedding=embedding, show_progress_bar=False)

    assert success == 1  # 1 written via the one-at-a-time retry
    assert errors == [{"id": "uuid-2", "sid": 2, "error": "payload rejected"}]


# ---------------------------------------------------------------------------
# sync_variables_op: variables only, no study document, no chunks, no embedding
# ---------------------------------------------------------------------------


def test_sync_variables_op_for_sids_reports_progress_and_never_indexes_studies():
    import nada_ai.ingest.service as service_module

    progress: list[dict] = []
    with (
        patch.object(service_module, "run_bulk_index") as mock_studies,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        mock_sync.side_effect = [{"sid": 1, "indexed": 3, "errors": []}, RuntimeError("boom")]
        result = service_module.sync_variables_op(_settings(), [1, 2], progress_cb=progress.append)

    mock_studies.assert_not_called()  # no study document, no chunks, so nothing is embedded
    assert result["scope"] == "sids" and result["requested"] == 2 and result["indexed"] == 3
    assert result["error_count"] == 1 and result["errors"][0]["sid"] == 2
    assert [p["processed"] for p in progress] == [1, 2] and progress[-1]["failed"] == 1
    assert progress[-1]["current_sid"] == 2 and progress[-1]["percent"] == 100.0


def test_sync_variables_op_for_the_whole_catalog_uses_the_backfill_and_caps_the_errors():
    import nada_ai.ingest.service as service_module

    many = [{"index": {"_id": str(i)}} for i in range(200)]
    with patch("nada_ai.ingest.variables_index.backfill_variables_op") as mock_backfill:
        mock_backfill.return_value = {"seen": 9, "indexed": 9, "errors": many, "total": 9, "cancelled": False}
        result = service_module.sync_variables_op(_settings(), None)

    assert result["scope"] == "all" and result["seen"] == 9
    assert result["error_count"] == 200 and len(result["errors"]) == 50  # a bulk failure echoes whole documents


def test_sync_variables_op_stops_when_cancelled():
    import nada_ai.ingest.service as service_module
    from nada_ai.ingest.progress import CancelToken

    token = CancelToken()
    token.set()
    with patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync:
        result = service_module.sync_variables_op(_settings(), [1, 2], cancel_token=token)
    mock_sync.assert_not_called()
    assert result["cancelled"] is True


# ---------------------------------------------------------------------------
# apply_study_options_op: flags only, no metadata read and no embedding
# ---------------------------------------------------------------------------


def _publish_state(
    *,
    indexed: bool,
    published: str = "0",
    core: dict | None = None,
    stored_all_text: list[str] | None = None,
):
    from unittest.mock import MagicMock

    import nada_ai.ingest.service as service_module

    client = MagicMock()
    client.exists.return_value = indexed
    client.update_by_query.return_value = {"updated": 4}
    client.get.return_value = {"fields": {"all_text": stored_all_text}} if stored_all_text else {}
    record = {
        "core_fields": core or {"catalog_id": 7, "idno": "A", "title": "T"},
        "filters": {"published": [published], "countries": ["1"]},
    }
    with (
        patch(
            "nada_ai.ingest.study_sources.fetch_studies_by_sids",
            return_value={"studies": [record], "missing": [], "errors": []},
        ) as fetch,
        patch.object(service_module, "build_client", return_value=client),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op", return_value={"errors": []}) as sync,
    ):
        result = service_module.apply_study_options_op(_settings(search_backend="opensearch"), 7)
    assert fetch.call_args.args[1] == [7]  # the record is fetched by id, without its metadata
    return result, client, sync


def test_apply_study_options_rewrites_the_study_and_its_chunks_in_place():
    result, client, sync = _publish_state(indexed=True, published="0")

    assert result == {"sid": 7, "updated": True, "chunks_updated": 4}
    written = client.index.call_args.kwargs
    assert written["id"] == "7" and written["body"]["filter_facets"]["published"] == ["0"]
    (chunks,) = (c.kwargs for c in client.update_by_query.call_args_list)  # the variable index is never updated
    assert chunks["body"]["script"]["params"]["facets"]["published"] == ["0"]
    assert chunks["body"]["query"] == {"term": {"metadata.sid": 7}}
    sync.assert_not_called()


def test_apply_study_options_keeps_the_all_text_nada_sends_only_with_full_updates():
    """A partial record has no all_text: the stored one is written again, not dropped."""
    _, client, _ = _publish_state(indexed=True, stored_all_text=["Women and men aged 15 to 49", "Kabul"])
    assert client.get.call_args.kwargs == {
        "index": _settings(search_backend="opensearch").studies_index,
        "id": "7",
        "stored_fields": "all_text",
        "_source": False,
    }
    assert client.index.call_args.kwargs["body"]["all_text"] == ["Women and men aged 15 to 49", "Kabul"]


def test_apply_study_options_writes_the_all_text_nada_sent_without_reading_the_stored_one():
    core = {"catalog_id": 7, "idno": "A", "title": "T", "all_text": ["New notes"]}
    _, client, _ = _publish_state(indexed=True, core=core, stored_all_text=["Old notes"])
    client.get.assert_not_called()
    assert client.index.call_args.kwargs["body"]["all_text"] == ["New notes"]


def test_apply_study_options_writes_no_all_text_when_none_is_stored():
    _, client, _ = _publish_state(indexed=True)
    assert "all_text" not in client.index.call_args.kwargs["body"]


def test_publishing_a_study_touches_none_of_its_variables():
    """Variable search reads the study's published state from the study index, so its variables are not rewritten."""
    _, client, sync = _publish_state(indexed=True, published="1")
    sync.assert_not_called()
    client.search.assert_not_called()


def test_apply_study_options_leaves_a_study_that_is_not_indexed_alone():
    result, client, _ = _publish_state(indexed=False)
    assert result == {"sid": 7, "updated": False, "reason": "not_indexed"}
    client.index.assert_not_called()
    client.update_by_query.assert_not_called()
