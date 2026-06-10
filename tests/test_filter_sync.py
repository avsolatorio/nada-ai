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


@patch("nada_ai.filters.sync.build_client")
def test_sync_opensearch_updates(mock_build_client):
    client = MagicMock()
    mock_build_client.return_value = client
    client.count.return_value = {"count": 3}
    client.update_by_query.return_value = {"updated": 3}

    settings = Settings(search_backend="opensearch")
    res = sync_filters_for_idno(settings, "DOC-1", {"countries": [181]})

    assert res["found"] is True
    assert res["updated_points"] == 3
    client.update_by_query.assert_called_once()
    client.transport.close.assert_called_once()
