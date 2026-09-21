"""Tests for filter sync helpers (mocked backends)."""

from unittest.mock import MagicMock, patch

from nada_ai.filters.sync import sync_filters_for_idno
from nada_ai.settings import Settings


@patch("nada_ai.filters.sync.qdrant_client")
def test_sync_qdrant_not_found(mock_client_fn):
    client = MagicMock()
    mock_client_fn.return_value = client
    client.count.return_value = MagicMock(count=0)

    settings = Settings(search_backend="qdrant")
    res = sync_filters_for_idno(settings, "MISSING", {"countries": [181]})

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
    res = sync_filters_for_idno(settings, "DOC-1", {"countries": [181]})

    assert res == {"idno": "DOC-1", "updated_points": 2, "found": True}
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
    client.search.return_value = {"hits": {"hits": [{"_source": {"idno": "DOC-1", "sid": 9}}]}}
    client.count.return_value = {"count": 3}
    client.update_by_query.return_value = {"updated": 3}

    settings = Settings(search_backend="opensearch", index_name="chunks")
    res = sync_filters_for_idno(settings, "DOC-1", {"countries": [181]})

    assert res["found"] is True
    assert res["updated_points"] == 3
    chunk_call, study_call = client.update_by_query.call_args_list
    # chunks are matched by stored idno and by sid, and get the flat map, not nested rows
    assert chunk_call.kwargs["index"] == "chunks"
    should = chunk_call.kwargs["body"]["query"]["bool"]["should"]
    assert {"term": {"metadata.idno": "DOC-1"}} in should
    assert {"terms": {"metadata.sid": [9]}} in should
    assert chunk_call.kwargs["body"]["script"]["params"] == {"facets": {"countries": ["181"]}}
    assert "filter_fields" not in chunk_call.kwargs["body"]["script"]["source"]
    # the study document is matched by NADA idno
    assert study_call.kwargs["index"] == "chunks-studies"
    assert study_call.kwargs["body"]["query"] == {"term": {"idno": "DOC-1"}}
    assert study_call.kwargs["body"]["script"]["params"] == {"facets": {"countries": ["181"]}}
    client.transport.close.assert_called_once()


@patch("nada_ai.filters.sync.build_client")
def test_sync_opensearch_reports_not_found_when_nothing_is_indexed(mock_build_client):
    client = MagicMock()
    mock_build_client.return_value = client
    client.search.return_value = {"hits": {"hits": []}}
    client.count.return_value = {"count": 0}

    res = sync_filters_for_idno(Settings(search_backend="opensearch"), "GONE", {"countries": [1]})

    assert res["found"] is False
    client.update_by_query.assert_not_called()
