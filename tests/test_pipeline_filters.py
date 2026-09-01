"""Tests for filters/facets being baked into content ingest (ingest/pipeline.py)
and the shared fetch helpers in nada_ai.filters.sync — see the "index_from_catalog
should also index filters" design discussion.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from nada_ai.filters.sync import fetch_filters_for_idno, sync_filters_for_idno_from_nada
from nada_ai.settings import Settings


def _settings(tmp_path, **overrides) -> Settings:
    path = tmp_path / "dynamic_filter_facets.json"
    return Settings(dynamic_filter_facets_path=str(path), **overrides)


# ---------------------------------------------------------------------------
# fetch_filters_for_idno
# ---------------------------------------------------------------------------

def test_fetch_filters_prefers_cached_extract_filters(tmp_path):
    settings = _settings(tmp_path)
    raw_metadata = {"_extract_filters": {"countries": ["181"]}}
    with patch("nada_ai.filters.metadata_extract.fetch_study_records") as mock_fetch:
        result = fetch_filters_for_idno(settings, "DOC-1", raw_metadata=raw_metadata)
    assert result == {"countries": ["181"]}
    mock_fetch.assert_not_called()


def test_fetch_filters_falls_back_to_explicit_call_when_no_cached_filters(tmp_path):
    settings = _settings(tmp_path)
    with patch(
        "nada_ai.filters.metadata_extract.fetch_study_records",
        return_value=[{"idno": "DOC-1", "filters": {"tags": ["health"]}}],
    ) as mock_fetch:
        result = fetch_filters_for_idno(settings, "DOC-1", raw_metadata=None)
    assert result == {"tags": ["health"]}
    mock_fetch.assert_called_once()


def test_fetch_filters_returns_none_when_extract_raises(tmp_path):
    from nada_ai.filters.metadata_extract import MetadataExtractError

    settings = _settings(tmp_path)
    with patch(
        "nada_ai.filters.metadata_extract.fetch_study_records",
        side_effect=MetadataExtractError("no filters found"),
    ):
        result = fetch_filters_for_idno(settings, "DOC-1")
    assert result is None


def test_fetch_filters_returns_none_on_unexpected_exception(tmp_path):
    settings = _settings(tmp_path)
    with patch("nada_ai.filters.metadata_extract.fetch_study_records", side_effect=RuntimeError("boom")):
        result = fetch_filters_for_idno(settings, "DOC-1")
    assert result is None


def test_fetch_filters_returns_none_when_no_records(tmp_path):
    settings = _settings(tmp_path)
    with patch("nada_ai.filters.metadata_extract.fetch_study_records", return_value=[]):
        result = fetch_filters_for_idno(settings, "DOC-1")
    assert result is None


# ---------------------------------------------------------------------------
# sync_filters_for_idno_from_nada
# ---------------------------------------------------------------------------

@patch("nada_ai.filters.sync.qdrant_client")
def test_sync_from_nada_syncs_when_filters_found(mock_client_fn, tmp_path):
    client = MagicMock()
    mock_client_fn.return_value = client
    client.count.return_value = MagicMock(count=1)
    settings = _settings(tmp_path, search_backend="qdrant")

    with patch(
        "nada_ai.filters.metadata_extract.fetch_study_records",
        return_value=[{"idno": "DOC-1", "filters": {"countries": ["181"]}}],
    ):
        result = sync_filters_for_idno_from_nada(settings, "DOC-1")

    assert result is not None
    assert result["found"] is True
    client.set_payload.assert_called_once()


def test_sync_from_nada_returns_none_when_no_filters_available(tmp_path):
    settings = _settings(tmp_path, search_backend="qdrant")
    with patch("nada_ai.filters.metadata_extract.fetch_study_records", return_value=[]):
        result = sync_filters_for_idno_from_nada(settings, "DOC-1")
    assert result is None


# ---------------------------------------------------------------------------
# iter_langdoc_records bakes filter_fields/filter_facets into the payload
# ---------------------------------------------------------------------------

class _FakeDoc:
    def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata


class _FakeHandler:
    def __init__(self, docs: list[_FakeDoc]) -> None:
        self._docs = docs

    def get_langdocs(self) -> list[_FakeDoc]:
        return self._docs


class _FakeLoader:
    """Stand-in for ai4data.discovery.metadata.handler.MetadataLoader."""

    _by_idno: dict[str, list[_FakeDoc]] = {}
    _raw_by_idno: dict[str, dict[str, Any]] = {}

    def __init__(self, idno: str, metadata_type: str, force: bool = False, include_resources: bool = True) -> None:
        self.idno = idno
        self.metadata_type = metadata_type
        self.metadata = self._raw_by_idno.get(idno, {})

    def get_metadata_handler(self) -> _FakeHandler:
        return _FakeHandler(self._by_idno.get(self.idno, []))


class _FakeVec:
    def tolist(self) -> list[float]:
        return [0.1, 0.2]


class _FakeEmbedding:
    """Stand-in for EmbeddingService — avoids loading a real model in tests."""

    def encode_corpus(self, texts: list[str], show_progress_bar: bool = True) -> list[_FakeVec]:
        return [_FakeVec() for _ in texts]


def test_iter_langdoc_records_bakes_in_cached_extract_filters(tmp_path):
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "DOC-1": [_FakeDoc("a perfectly fine and long enough description", {"idno": "DOC-1", "type": "document"})],
    }
    _FakeLoader._raw_by_idno = {"DOC-1": {"_extract_filters": {"brand_new_facet_key": ["x"]}}}

    with (
        patch.object(pipeline_module, "MetadataLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="opensearch")
        results = list(
            pipeline_module.iter_langdoc_records(
                settings, _FakeEmbedding(), [("DOC-1", "document")], show_progress_bar=False
            )
        )

    assert len(results) == 1
    _, _, source = results[0]
    assert source["metadata"]["filter_fields"] == [{"key": "brand_new_facet_key", "value": ["x"]}]
    # OpenSearch backend never gets the flat facets map
    assert "filter_facets" not in source["metadata"]

    from nada_ai.search.dynamic_filters import load_dynamic_facet_keys

    assert "brand_new_facet_key" in load_dynamic_facet_keys(settings)


def test_iter_langdoc_records_skips_filters_when_disabled(tmp_path):
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "DOC-1": [_FakeDoc("a perfectly fine and long enough description", {"idno": "DOC-1", "type": "document"})],
    }
    _FakeLoader._raw_by_idno = {"DOC-1": {"_extract_filters": {"some_key": ["x"]}}}

    with (
        patch.object(pipeline_module, "MetadataLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="opensearch", sync_filters_during_ingest=False)
        results = list(
            pipeline_module.iter_langdoc_records(
                settings, _FakeEmbedding(), [("DOC-1", "document")], show_progress_bar=False
            )
        )

    _, _, source = results[0]
    assert "filter_fields" not in source["metadata"]


def test_iter_langdoc_records_falls_back_to_explicit_fetch_and_includes_qdrant_facets(tmp_path):
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "DOC-1": [_FakeDoc("a perfectly fine and long enough description", {"idno": "DOC-1", "type": "document"})],
    }
    _FakeLoader._raw_by_idno = {"DOC-1": {}}  # no _extract_filters cached — must fall back

    with (
        patch.object(pipeline_module, "MetadataLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
        patch(
            "nada_ai.filters.metadata_extract.fetch_study_records",
            return_value=[{"idno": "DOC-1", "filters": {"tags": ["health"]}}],
        ),
    ):
        settings = _settings(tmp_path, search_backend="qdrant")
        results = list(
            pipeline_module.iter_langdoc_records(
                settings, _FakeEmbedding(), [("DOC-1", "document")], show_progress_bar=False
            )
        )

    _, _, source = results[0]
    assert source["metadata"]["filter_fields"] == [{"key": "tags", "value": ["health"]}]
    assert source["metadata"]["filter_facets"] == {"tags": ["health"]}
