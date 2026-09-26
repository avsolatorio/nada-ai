"""Tests for reporting indexing/deletion outcomes to NADA's search_index_state
(service._report_state_bulk_best_effort), wired into index_ids_op,
index_from_catalog_op, delete_by_idno_op, and delete_by_idnos_op.
"""

from __future__ import annotations

from unittest.mock import patch

from nada_ai.settings import Settings


def _settings(**overrides) -> Settings:
    overrides.setdefault("report_search_index_state_enabled", True)
    return Settings(**overrides)


# ---------------------------------------------------------------------------
# _error_by_idno
# ---------------------------------------------------------------------------


def test_error_by_idno_prefers_load_error_over_empty_reason():
    import nada_ai.ingest.service as service_module

    load_errors = [{"idno": "A", "error": "boom"}]
    empty_docs = [{"idno": "A", "reason": "no_langdocs"}, {"idno": "B", "reason": "all_empty_content"}]

    result = service_module._error_by_idno(load_errors, empty_docs)

    assert result == {"A": "boom", "B": "no documents produced (all_empty_content)"}


# ---------------------------------------------------------------------------
# _report_state_bulk_best_effort itself
# ---------------------------------------------------------------------------


def test_report_state_bulk_best_effort_noop_when_disabled():
    import nada_ai.ingest.service as service_module

    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module._report_state_bulk_best_effort(
            _settings(report_search_index_state_enabled=False),
            [{"object_type": "survey", "object_key": "A", "status": "indexed"}],
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
    items = [{"object_type": "survey", "object_key": "A", "status": "indexed"}]
    with patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module._report_state_bulk_best_effort(settings, items)
    mock_report.assert_called_once_with(settings, items)


def test_report_state_bulk_best_effort_swallows_errors():
    import nada_ai.ingest.service as service_module

    with patch("nada_ai.ingest.search_index_sync.report_state_bulk", side_effect=RuntimeError("network down")):
        # Must not raise.
        service_module._report_state_bulk_best_effort(
            _settings(), [{"object_type": "survey", "object_key": "A", "status": "indexed"}]
        )


# ---------------------------------------------------------------------------
# index_ids_op
# ---------------------------------------------------------------------------


def _fake_run_bulk_index_with_one_failure(failing_idno: str):
    def _fn(settings, pairs, **kwargs):
        load_errors = kwargs.get("load_errors")
        n = 0
        for idno, _ in pairs:
            if idno == failing_idno and load_errors is not None:
                load_errors.append({"idno": idno, "metadata_type": "indicator", "stage": "load", "error": "boom"})
            else:
                n += 1
        return n, None

    return _fn


def test_index_ids_op_reports_success_and_failure_separately():
    import nada_ai.ingest.service as service_module

    fake = _fake_run_bulk_index_with_one_failure("BAD")
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.index_ids_op(_settings(), ["GOOD", "BAD"], "indicator")

    mock_report.assert_called_once()
    items = mock_report.call_args.args[1]
    by_key = {i["object_key"]: i["status"] for i in items}
    assert by_key == {"GOOD": "indexed", "BAD": "failed"}
    assert all(i["object_type"] == "survey" for i in items)
    by_key_error = {i["object_key"]: i.get("error") for i in items}
    assert by_key_error == {"GOOD": None, "BAD": "boom"}


def test_index_ids_op_does_not_report_when_disabled():
    import nada_ai.ingest.service as service_module

    fake = _fake_run_bulk_index_with_one_failure("BAD")
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.index_ids_op(_settings(report_search_index_state_enabled=False), ["GOOD", "BAD"], "indicator")
    mock_report.assert_not_called()


# ---------------------------------------------------------------------------
# index_ids_op also syncs variables for metadata_type=microdata (a full study
# index is study + chunks + variables together — see
# docs/variables-search-contract.md)
# ---------------------------------------------------------------------------


def test_index_ids_op_syncs_variables_for_microdata():
    import nada_ai.ingest.service as service_module

    fake = _fake_run_bulk_index_with_one_failure("__none__")  # nothing fails
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        mock_sync.return_value = {"indexed": 3, "errors": []}
        result = service_module.index_ids_op(_settings(search_backend="opensearch"), ["A", "B"], "microdata")

    assert mock_sync.call_count == 2
    synced = {c.args[1] for c in mock_sync.call_args_list}
    assert synced == {"A", "B"}
    assert result["variables"] == {"indexed": 6, "errors": [], "failed": {}}


def test_index_ids_op_skips_variables_for_a_failed_idno():
    import nada_ai.ingest.service as service_module

    fake = _fake_run_bulk_index_with_one_failure("BAD")
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        mock_sync.return_value = {"indexed": 1, "errors": []}
        service_module.index_ids_op(_settings(search_backend="opensearch"), ["GOOD", "BAD"], "microdata")

    assert [c.args[1] for c in mock_sync.call_args_list] == ["GOOD"]


def test_index_ids_op_does_not_sync_variables_for_other_metadata_types():
    import nada_ai.ingest.service as service_module

    fake = _fake_run_bulk_index_with_one_failure("__none__")
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        result = service_module.index_ids_op(_settings(search_backend="opensearch"), ["A"], "indicator")

    mock_sync.assert_not_called()
    assert result["variables"] is None


def test_index_ids_op_does_not_sync_variables_for_qdrant():
    """Variable search is OpenSearch-only; a qdrant deployment has no separate variable index to sync."""
    import nada_ai.ingest.service as service_module

    fake = _fake_run_bulk_index_with_one_failure("__none__")
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        result = service_module.index_ids_op(_settings(search_backend="qdrant"), ["A"], "microdata")

    mock_sync.assert_not_called()
    assert result["variables"] is None


def test_index_ids_op_variable_sync_failure_does_not_fail_the_call():
    import nada_ai.ingest.service as service_module

    fake = _fake_run_bulk_index_with_one_failure("__none__")
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op", side_effect=RuntimeError("boom")),
    ):
        result = service_module.index_ids_op(_settings(search_backend="opensearch"), ["A"], "microdata")

    assert result["indexed"] == 1  # the study/chunk indexing itself still succeeded
    assert result["variables"]["errors"] == [{"idno": "A", "error": "boom"}]


def _reported(mock_report) -> dict[str, tuple[str, str | None]]:
    items = mock_report.call_args.args[1]
    return {i["object_key"]: (i["status"], i.get("error")) for i in items}


def test_index_ids_op_reports_a_study_whose_variables_failed_as_failed():
    """A full index of a study is study + chunks + variables: a study whose variables did not all land is not
    indexed, or NADA drops it from its "missing" diff and nothing retries its variables."""
    import nada_ai.ingest.service as service_module

    def sync(settings, idno, **_):
        return {"indexed": 1, "errors": [{"index": {"status": 400}}] if idno == "A" else []}

    fake = _fake_run_bulk_index_with_one_failure("__none__")
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op", side_effect=sync),
    ):
        result = service_module.index_ids_op(_settings(search_backend="opensearch"), ["A", "B"], "microdata")

    assert _reported(mock_report) == {"A": ("failed", "variables: 1 write error(s)"), "B": ("indexed", None)}
    assert result["variables"]["failed"] == {"A": "1 write error(s)"}


def test_index_ids_op_reports_a_study_whose_variable_sync_raised_as_failed():
    import nada_ai.ingest.service as service_module

    fake = _fake_run_bulk_index_with_one_failure("__none__")
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op", side_effect=RuntimeError("boom")),
    ):
        service_module.index_ids_op(_settings(search_backend="opensearch"), ["A"], "microdata")

    assert _reported(mock_report) == {"A": ("failed", "variables: boom")}


def test_a_cancelled_variable_sync_counts_every_study_not_synced_as_failed():
    import nada_ai.ingest.service as service_module
    from nada_ai.ingest.progress import CancelToken

    token = CancelToken()

    def sync(settings, idno, **_):
        token.set()  # cancelled during the first study, after it finished
        return {"indexed": 1, "errors": []}

    with patch("nada_ai.ingest.variables_index.sync_survey_variables_op", side_effect=sync):
        result = service_module._sync_variables_best_effort(_settings(), ["A", "B", "C"], cancel_token=token)

    assert result["failed"] == {
        "B": "cancelled before its variables were synced",
        "C": "cancelled before its variables were synced",
    }


# ---------------------------------------------------------------------------
# delete_by_idno_op / delete_by_idnos_op
# ---------------------------------------------------------------------------


def test_delete_by_idno_op_reports_deleted_status():
    import nada_ai.ingest.service as service_module

    with (
        patch.object(service_module, "_delete_qdrant", return_value={"backend": "qdrant"}),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.delete_by_idno_op(_settings(search_backend="qdrant"), "IDNO-1")

    mock_report.assert_called_once()
    items = mock_report.call_args.args[1]
    assert items == [{"object_type": "survey", "object_key": "IDNO-1", "status": "deleted"}]


def test_delete_by_idnos_op_reports_all_deleted():
    import nada_ai.ingest.service as service_module

    with (
        patch.object(service_module, "_delete_qdrant_batch", return_value={"backend": "qdrant"}),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.delete_by_idnos_op(_settings(search_backend="qdrant"), ["A", "B"])

    mock_report.assert_called_once()
    items = mock_report.call_args.args[1]
    assert {i["object_key"] for i in items} == {"A", "B"}
    assert all(i["status"] == "deleted" for i in items)


# ---------------------------------------------------------------------------
# index_from_catalog_op: empty_docs must not be reported as 'indexed'
# ---------------------------------------------------------------------------


def test_index_from_catalog_op_excludes_empty_docs_from_indexed_report(tmp_path, monkeypatch):
    import nada_ai.ingest.service as service_module

    monkeypatch.setenv("NADA_INGEST_CHECKPOINT_DIR", str(tmp_path))
    settings = _settings()

    def fake_get_metadata_ids(params, **kwargs):
        return [{"idno": "GOOD", "type": "geospatial"}, {"idno": "EMPTY", "type": "geospatial"}]

    def fake_run_bulk_index(settings, pairs, **kwargs):
        progress = kwargs.get("progress")
        empty_docs = kwargs.get("empty_docs")
        for idno, _ in pairs:
            if idno == "EMPTY":
                if empty_docs is not None:
                    empty_docs.append({"idno": idno, "metadata_type": "geospatial", "reason": "no_langdocs"})
                if progress is not None:
                    progress.mark(idno, ok=True)
            else:
                if progress is not None:
                    progress.mark(idno, ok=True)
        return 1, None

    with (
        patch("ai4data.discovery.catalog.get_metadata_ids", fake_get_metadata_ids),
        patch("ai4data.discovery.catalog.is_extract_mode", return_value=False),
        patch.object(service_module, "run_bulk_index", fake_run_bulk_index),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.index_from_catalog_op(settings, catalog_type="geospatial", show_progress_bar=False)

    mock_report.assert_called_once()
    items = mock_report.call_args.args[1]
    by_key = {i["object_key"]: i["status"] for i in items}
    assert by_key == {"GOOD": "indexed", "EMPTY": "failed"}
    by_key_error = {i["object_key"]: i.get("error") for i in items}
    assert by_key_error == {"GOOD": None, "EMPTY": "no documents produced (no_langdocs)"}


# ---------------------------------------------------------------------------
# index_from_catalog_op syncs variables too, for the microdata studies of the run only
# ---------------------------------------------------------------------------


def _run_catalog(tmp_path, monkeypatch, rows, *, settings, load_error_idnos=(), cancel_token=None):
    import nada_ai.ingest.service as service_module

    monkeypatch.setenv("NADA_INGEST_CHECKPOINT_DIR", str(tmp_path))

    def fake_run_bulk_index(settings, pairs, **kwargs):
        for idno, _ in pairs:
            if idno in load_error_idnos:
                kwargs["load_errors"].append({"idno": idno, "stage": "load", "error": "boom"})
            if kwargs.get("progress") is not None:
                kwargs["progress"].mark(idno, ok=True)
        return len(pairs), None

    with (
        patch("ai4data.discovery.catalog.get_metadata_ids", lambda params, **kw: rows),
        patch("ai4data.discovery.catalog.is_extract_mode", return_value=False),
        patch.object(service_module, "run_bulk_index", fake_run_bulk_index),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk"),
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        mock_sync.return_value = {"indexed": 2, "errors": []}
        result = service_module.index_from_catalog_op(
            settings, catalog_type="microdata", show_progress_bar=False, cancel_token=cancel_token
        )
    return result, mock_sync


def test_index_from_catalog_op_syncs_variables_of_microdata_rows_only(tmp_path, monkeypatch):
    rows = [
        {"idno": "M1", "type": "microdata"},
        {"idno": "D1", "type": "document"},
        {"idno": "M2", "type": "microdata"},
    ]
    result, mock_sync = _run_catalog(tmp_path, monkeypatch, rows, settings=_settings(search_backend="opensearch"))

    assert {c.args[1] for c in mock_sync.call_args_list} == {"M1", "M2"}  # never the document row
    assert result["variables"] == {"indexed": 4, "errors": [], "failed": {}}


def test_index_from_catalog_op_skips_variables_of_a_study_that_failed_to_load(tmp_path, monkeypatch):
    rows = [{"idno": "M1", "type": "microdata"}, {"idno": "BAD", "type": "microdata"}]
    _, mock_sync = _run_catalog(
        tmp_path, monkeypatch, rows, settings=_settings(search_backend="opensearch"), load_error_idnos={"BAD"}
    )
    assert [c.args[1] for c in mock_sync.call_args_list] == ["M1"]


def test_index_from_catalog_op_makes_no_variable_call_for_a_catalog_without_microdata(tmp_path, monkeypatch):
    rows = [{"idno": "D1", "type": "document"}, {"idno": "D2", "type": "document"}]
    result, mock_sync = _run_catalog(tmp_path, monkeypatch, rows, settings=_settings(search_backend="opensearch"))
    mock_sync.assert_not_called()
    assert result["variables"] is None


def test_index_from_catalog_op_does_not_sync_variables_on_qdrant(tmp_path, monkeypatch):
    rows = [{"idno": "M1", "type": "microdata"}]
    result, mock_sync = _run_catalog(tmp_path, monkeypatch, rows, settings=_settings(search_backend="qdrant"))
    mock_sync.assert_not_called()
    assert result["variables"] is None


def test_index_from_catalog_op_skips_variables_when_cancelled(tmp_path, monkeypatch):
    from nada_ai.ingest.progress import CancelToken

    token = CancelToken()
    token.set()
    rows = [{"idno": "M1", "type": "microdata"}]
    result, mock_sync = _run_catalog(
        tmp_path, monkeypatch, rows, settings=_settings(search_backend="opensearch"), cancel_token=token
    )
    mock_sync.assert_not_called()
    assert result["cancelled"] is True


def test_a_cancelled_catalog_run_reports_its_microdata_studies_failed_not_indexed(tmp_path, monkeypatch):
    """Their variables were never synced, so they are not fully indexed: a reconcile must retry them."""
    import nada_ai.ingest.service as service_module
    from nada_ai.ingest.progress import CancelToken

    monkeypatch.setenv("NADA_INGEST_CHECKPOINT_DIR", str(tmp_path))
    token = CancelToken()

    def fake_run_bulk_index(settings, pairs, **kwargs):
        for idno, _ in pairs:
            kwargs["progress"].mark(idno, ok=True)
        token.set()  # cancelled once the studies were written, before the variables
        return len(pairs), None

    rows = [{"idno": "M1", "type": "microdata"}, {"idno": "D1", "type": "document"}]
    with (
        patch("ai4data.discovery.catalog.get_metadata_ids", lambda params, **kw: rows),
        patch("ai4data.discovery.catalog.is_extract_mode", return_value=False),
        patch.object(service_module, "run_bulk_index", fake_run_bulk_index),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        service_module.index_from_catalog_op(
            _settings(search_backend="opensearch"),
            catalog_type="microdata",
            show_progress_bar=False,
            cancel_token=token,
        )

    mock_sync.assert_not_called()
    assert _reported(mock_report) == {
        "D1": ("indexed", None),
        "M1": ("failed", "variables: cancelled before its variables were synced"),
    }


# ---------------------------------------------------------------------------
# Backend write errors must never be reported as 'indexed'
# ---------------------------------------------------------------------------


def _fake_run_bulk_index_with_write_errors(write_errors):
    def _fn(settings, pairs, **kwargs):
        return len(pairs) - len(write_errors), list(write_errors)

    return _fn


def test_attribute_write_error_qdrant_shape():
    import nada_ai.ingest.service as service_module

    idno, msg = service_module._attribute_write_error({"id": "u1", "idno": "A", "error": "rejected"})
    assert (idno, msg) == ("A", "rejected")


def test_attribute_write_error_opensearch_bulk_shape():
    import nada_ai.ingest.service as service_module

    err = {"index": {"_id": "u1", "status": 400, "error": {"type": "mapper"}, "data": {"metadata": {"idno": "A"}}}}
    idno, msg = service_module._attribute_write_error(err)
    assert idno == "A"
    assert "mapper" in msg


def test_attribute_write_error_unattributable():
    import nada_ai.ingest.service as service_module

    assert service_module._attribute_write_error({"error": "run aborted"})[0] is None
    assert service_module._attribute_write_error({"id": "u1", "idno": None, "error": "x"})[0] is None
    assert service_module._attribute_write_error("plain string")[0] is None
    assert service_module._attribute_write_error({"index": {"error": "x"}})[0] is None


def test_index_ids_op_reports_write_failure_as_failed_not_indexed():
    import nada_ai.ingest.service as service_module

    fake = _fake_run_bulk_index_with_write_errors([{"id": "u1", "idno": "BAD", "error": "payload rejected"}])
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.index_ids_op(_settings(), ["GOOD", "BAD"], "indicator")

    items = mock_report.call_args.args[1]
    assert {i["object_key"]: i["status"] for i in items} == {"GOOD": "indexed", "BAD": "failed"}
    bad = next(i for i in items if i["object_key"] == "BAD")
    assert bad["error"] == "write failed: payload rejected"


def test_index_ids_op_reports_opensearch_bulk_write_failure_as_failed():
    import nada_ai.ingest.service as service_module

    err = {"index": {"_id": "u1", "status": 400, "error": "boom", "data": {"metadata": {"idno": "BAD"}}}}
    fake = _fake_run_bulk_index_with_write_errors([err])
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.index_ids_op(_settings(), ["GOOD", "BAD"], "indicator")

    items = mock_report.call_args.args[1]
    assert {i["object_key"]: i["status"] for i in items} == {"GOOD": "indexed", "BAD": "failed"}


def test_index_ids_op_reports_nothing_indexed_when_write_error_unattributable():
    import nada_ai.ingest.service as service_module

    # e.g. Qdrant went away mid-run: the writer reports one un-attributable error for the whole run.
    fake = _fake_run_bulk_index_with_write_errors([{"error": "connection refused"}])
    with (
        patch.object(service_module, "run_bulk_index", fake),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.index_ids_op(_settings(), ["A", "B"], "indicator")

    # Nothing to report at all: A/B must stay 'missing' in NADA so a reconcile retries them.
    mock_report.assert_not_called()


def test_unattributable_write_error_still_reports_known_load_failures():
    import nada_ai.ingest.service as service_module

    items = service_module._state_report_items(
        ["A", "B"],
        load_errors=[{"idno": "B", "error": "boom"}],
        empty_docs=[],
        write_errors=[{"error": "connection refused"}],
    )
    assert items == [{"object_type": "survey", "object_key": "B", "status": "failed", "error": "boom"}]


def test_load_error_message_wins_over_write_error_for_same_idno():
    import nada_ai.ingest.service as service_module

    items = service_module._state_report_items(
        ["A"],
        load_errors=[{"idno": "A", "error": "load boom"}],
        empty_docs=[],
        write_errors=[{"idno": "A", "error": "write boom"}],
    )
    assert [(i["object_key"], i["status"], i["error"]) for i in items] == [("A", "failed", "load boom")]


def test_index_from_catalog_op_reports_write_failure_as_failed_not_indexed(tmp_path, monkeypatch):
    import nada_ai.ingest.service as service_module

    monkeypatch.setenv("NADA_INGEST_CHECKPOINT_DIR", str(tmp_path))

    def fake_get_metadata_ids(params, **kwargs):
        return [{"idno": "GOOD", "type": "geospatial"}, {"idno": "BAD", "type": "geospatial"}]

    def fake_run_bulk_index(settings, pairs, **kwargs):
        progress = kwargs.get("progress")
        for idno, _ in pairs:
            if progress is not None:
                progress.mark(idno, ok=True)
        return 1, [{"id": "u1", "idno": "BAD", "error": "payload rejected"}]

    with (
        patch("ai4data.discovery.catalog.get_metadata_ids", fake_get_metadata_ids),
        patch("ai4data.discovery.catalog.is_extract_mode", return_value=False),
        patch.object(service_module, "run_bulk_index", fake_run_bulk_index),
        patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report,
    ):
        service_module.index_from_catalog_op(_settings(), catalog_type="geospatial", show_progress_bar=False)

    items = mock_report.call_args.args[1]
    assert {i["object_key"]: i["status"] for i in items} == {"GOOD": "indexed", "BAD": "failed"}


def test_qdrant_writer_attaches_idno_to_write_errors():
    """End to end through QdrantIngestWriter: a point Qdrant rejects is reported with its idno."""
    from unittest.mock import MagicMock

    import nada_ai.ingest.qdrant_writer as qw

    settings = _settings(qdrant_sparse_lexical=False)

    def record(idno):
        return (f"uuid-{idno}", [0.1, 0.2], {"page_content": "x", "metadata": {"idno": idno}})

    client = MagicMock()

    def upsert(collection_name, points, wait):
        if any(p.id == "uuid-BAD" for p in points):
            raise RuntimeError("payload rejected")

    client.upsert.side_effect = upsert
    embedding = MagicMock()
    embedding.embedding_dimension.return_value = 2

    with (
        patch.object(qw, "_client", return_value=client),
        patch.object(qw.QdrantIngestWriter, "ensure_target"),
        patch.object(qw, "iter_langdoc_records", return_value=iter([record("GOOD"), record("BAD")])),
    ):
        success, errors = qw.QdrantIngestWriter(settings).run_bulk(
            [("GOOD", "geospatial"), ("BAD", "geospatial")], embedding=embedding, show_progress_bar=False
        )

    assert success == 1  # GOOD written via the one-at-a-time retry
    assert errors == [{"id": "uuid-BAD", "idno": "BAD", "error": "payload rejected"}]


# ---------------------------------------------------------------------------
# sync_variables_op: variables only, no study document, no chunks, no embedding
# ---------------------------------------------------------------------------


def test_sync_variables_op_for_idnos_reports_progress_and_never_indexes_studies():
    import nada_ai.ingest.service as service_module

    progress: list[dict] = []
    with (
        patch.object(service_module, "run_bulk_index") as mock_studies,
        patch("nada_ai.ingest.variables_index.sync_survey_variables_op") as mock_sync,
    ):
        mock_sync.side_effect = [
            {"idno": "A", "indexed": 3, "errors": []},
            RuntimeError("boom"),
        ]
        result = service_module.sync_variables_op(_settings(), ["A", "B"], progress_cb=progress.append)

    mock_studies.assert_not_called()  # no study document, no chunks, so nothing is embedded
    assert result["scope"] == "idnos" and result["requested"] == 2 and result["indexed"] == 3
    assert result["error_count"] == 1 and result["errors"][0]["idno"] == "B"
    assert [p["processed"] for p in progress] == [1, 2] and progress[-1]["failed"] == 1
    assert progress[-1]["current_idno"] == "B" and progress[-1]["percent"] == 100.0


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
        result = service_module.sync_variables_op(_settings(), ["A", "B"], cancel_token=token)
    mock_sync.assert_not_called()
    assert result["cancelled"] is True


# ---------------------------------------------------------------------------
# apply_study_options_op: flags only, no metadata read and no embedding
# ---------------------------------------------------------------------------


def _publish_state(*, indexed: bool, published: str = "0"):
    from unittest.mock import MagicMock

    import nada_ai.ingest.service as service_module

    client = MagicMock()
    client.update_by_query.side_effect = [{"updated": 4}, {"updated": 9}]
    extract = (7, {"idno": "A", "title": "T"}, {"published": [published], "countries": ["1"]})
    with (
        patch("nada_ai.filters.metadata_extract.fetch_study_extract", return_value=extract),
        patch.object(service_module, "build_client", return_value=client),
        patch.object(service_module, "sids_for_idnos", return_value={"A": 7} if indexed else {}),
    ):
        result = service_module.apply_study_options_op(_settings(search_backend="opensearch"), "A")
    return result, client


def test_apply_study_options_rewrites_the_flags_of_the_study_its_chunks_and_its_variables():
    result, client = _publish_state(indexed=True, published="0")

    assert result == {"idno": "A", "updated": True, "sid": 7, "chunks_updated": 4, "variables_updated": 9}
    written = client.index.call_args.kwargs
    assert written["id"] == "7" and written["body"]["filter_facets"]["published"] == ["0"]
    chunks, variables = (c.kwargs for c in client.update_by_query.call_args_list)
    assert chunks["body"]["script"]["params"]["facets"]["published"] == ["0"]
    assert chunks["body"]["query"] == {"term": {"metadata.sid": 7}}
    assert variables["body"]["script"]["params"] == {"published": 0}
    assert variables["body"]["query"] == {"term": {"sid": 7}}


def test_apply_study_options_publishes_variables_as_1():
    _, client = _publish_state(indexed=True, published="1")
    variables = client.update_by_query.call_args_list[1].kwargs
    assert variables["body"]["script"]["params"] == {"published": 1}


def test_apply_study_options_leaves_a_study_that_is_not_indexed_alone():
    result, client = _publish_state(indexed=False)
    assert result == {"idno": "A", "updated": False, "reason": "not_indexed"}
    client.index.assert_not_called()
    client.update_by_query.assert_not_called()
