"""Tests for filter sync helpers (mocked backends): a study is found by its NADA internal id (sid)."""

from unittest.mock import MagicMock, patch

from nada_ai.filters.sync import sync_filters_for_sid
from nada_ai.settings import Settings


@patch("nada_ai.filters.sync.qdrant_client")
def test_sync_qdrant_not_found(mock_client_fn):
    client = MagicMock()
    mock_client_fn.return_value = client
    client.count.return_value = MagicMock(count=0)

    settings = Settings(search_backend="qdrant")
    res = sync_filters_for_sid(settings, 404, {"countries": [181]})

    assert res["found"] is False
    assert res["updated_points"] == 0
    client.set_payload.assert_not_called()
    client.close.assert_called_once()


@patch("nada_ai.filters.sync.qdrant_client")
def test_sync_qdrant_merges_into_metadata(mock_client_fn):
    client = MagicMock()
    mock_client_fn.return_value = client
    client.count.return_value = MagicMock(count=2)

    settings = Settings(search_backend="qdrant")
    res = sync_filters_for_sid(settings, 9, {"countries": [181]})

    assert res == {"sid": 9, "updated_points": 2, "found": True}
    assert client.count.call_args.kwargs["count_filter"].must[0].key == "metadata.sid"
    client.set_payload.assert_called_once()
    kwargs = client.set_payload.call_args.kwargs
    assert kwargs["key"] == "metadata"
    assert kwargs["payload"] == {
        "filter_fields": [{"key": "countries", "value": ["181"]}],
        "filter_facets": {"countries": ["181"]},
    }
    assert "metadata" not in kwargs["payload"]


@patch("nada_ai.filters.sync.build_client")
def test_sync_opensearch_updates_chunks_and_the_study_document(mock_build_client):
    client = MagicMock()
    mock_build_client.return_value = client
    client.exists.return_value = True
    client.count.return_value = {"count": 3}
    client.update_by_query.return_value = {"updated": 3}

    settings = Settings(search_backend="opensearch", index_name="chunks")
    with patch("nada_ai.ingest.service.apply_study_options_op") as rewrite_study:
        res = sync_filters_for_sid(settings, 9, {"countries": [181]})

    assert res == {"sid": 9, "updated_points": 3, "found": True}
    (chunk_call,) = client.update_by_query.call_args_list
    # chunks are matched by sid and get the flat map, not nested rows
    assert chunk_call.kwargs["index"] == "chunks"
    assert chunk_call.kwargs["body"]["query"] == {"term": {"metadata.sid": 9}}
    assert chunk_call.kwargs["body"]["script"]["params"] == {"facets": {"countries": ["181"]}}
    assert "filter_fields" not in chunk_call.kwargs["body"]["script"]["source"]
    # the study document is never updated in place (its index stores few fields): it is rewritten from NADA
    rewrite_study.assert_called_once_with(settings, 9)
    client.transport.close.assert_called_once()


@patch("nada_ai.filters.sync.build_client")
def test_sync_opensearch_reports_not_found_when_nothing_is_indexed(mock_build_client):
    client = MagicMock()
    mock_build_client.return_value = client
    client.exists.return_value = False
    client.count.return_value = {"count": 0}

    res = sync_filters_for_sid(Settings(search_backend="opensearch"), 404, {"countries": [1]})

    assert res["found"] is False
    client.update_by_query.assert_not_called()
