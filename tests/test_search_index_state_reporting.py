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
    with patch.object(service_module, "run_bulk_index", fake), \
         patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
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
    with patch.object(service_module, "run_bulk_index", fake), \
         patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module.index_ids_op(
            _settings(report_search_index_state_enabled=False), ["GOOD", "BAD"], "indicator"
        )
    mock_report.assert_not_called()


# ---------------------------------------------------------------------------
# delete_by_idno_op / delete_by_idnos_op
# ---------------------------------------------------------------------------


def test_delete_by_idno_op_reports_deleted_status():
    import nada_ai.ingest.service as service_module

    with patch.object(service_module, "_delete_qdrant", return_value={"backend": "qdrant"}), \
         patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module.delete_by_idno_op(_settings(search_backend="qdrant"), "IDNO-1")

    mock_report.assert_called_once()
    items = mock_report.call_args.args[1]
    assert items == [{"object_type": "survey", "object_key": "IDNO-1", "status": "deleted"}]


def test_delete_by_idnos_op_reports_all_deleted():
    import nada_ai.ingest.service as service_module

    with patch.object(service_module, "_delete_qdrant_batch", return_value={"backend": "qdrant"}), \
         patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
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

    with patch("ai4data.discovery.catalog.get_metadata_ids", fake_get_metadata_ids), \
         patch("ai4data.discovery.catalog.is_extract_mode", return_value=False), \
         patch.object(service_module, "run_bulk_index", fake_run_bulk_index), \
         patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module.index_from_catalog_op(settings, catalog_type="geospatial", show_progress_bar=False)

    mock_report.assert_called_once()
    items = mock_report.call_args.args[1]
    by_key = {i["object_key"]: i["status"] for i in items}
    assert by_key == {"GOOD": "indexed", "EMPTY": "failed"}
    by_key_error = {i["object_key"]: i.get("error") for i in items}
    assert by_key_error == {"GOOD": None, "EMPTY": "no documents produced (no_langdocs)"}


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
    with patch.object(service_module, "run_bulk_index", fake), \
         patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module.index_ids_op(_settings(), ["GOOD", "BAD"], "indicator")

    items = mock_report.call_args.args[1]
    assert {i["object_key"]: i["status"] for i in items} == {"GOOD": "indexed", "BAD": "failed"}
    bad = next(i for i in items if i["object_key"] == "BAD")
    assert bad["error"] == "write failed: payload rejected"


def test_index_ids_op_reports_opensearch_bulk_write_failure_as_failed():
    import nada_ai.ingest.service as service_module

    err = {"index": {"_id": "u1", "status": 400, "error": "boom", "data": {"metadata": {"idno": "BAD"}}}}
    fake = _fake_run_bulk_index_with_write_errors([err])
    with patch.object(service_module, "run_bulk_index", fake), \
         patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
        service_module.index_ids_op(_settings(), ["GOOD", "BAD"], "indicator")

    items = mock_report.call_args.args[1]
    assert {i["object_key"]: i["status"] for i in items} == {"GOOD": "indexed", "BAD": "failed"}


def test_index_ids_op_reports_nothing_indexed_when_write_error_unattributable():
    import nada_ai.ingest.service as service_module

    # e.g. Qdrant went away mid-run: the writer reports one un-attributable error for the whole run.
    fake = _fake_run_bulk_index_with_write_errors([{"error": "connection refused"}])
    with patch.object(service_module, "run_bulk_index", fake), \
         patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
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

    with patch("ai4data.discovery.catalog.get_metadata_ids", fake_get_metadata_ids), \
         patch("ai4data.discovery.catalog.is_extract_mode", return_value=False), \
         patch.object(service_module, "run_bulk_index", fake_run_bulk_index), \
         patch("nada_ai.ingest.search_index_sync.report_state_bulk") as mock_report:
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

    with patch.object(qw, "_client", return_value=client), \
         patch.object(qw.QdrantIngestWriter, "ensure_target"), \
         patch.object(qw, "iter_langdoc_records", return_value=iter([record("GOOD"), record("BAD")])):
        success, errors = qw.QdrantIngestWriter(settings).run_bulk(
            [("GOOD", "geospatial"), ("BAD", "geospatial")], embedding=embedding, show_progress_bar=False
        )

    assert success == 1  # GOOD written via the one-at-a-time retry
    assert errors == [{"id": "uuid-BAD", "idno": "BAD", "error": "payload rejected"}]
