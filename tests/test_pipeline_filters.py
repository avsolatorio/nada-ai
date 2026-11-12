"""Tests for filters/facets being baked into content ingest (ingest/pipeline.py): each study's record carries
its filters, so a bulk-indexed document is filterable immediately.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from nada_ai.ingest.study_sources import StudySource
from nada_ai.settings import Settings


def _settings(tmp_path, **overrides) -> Settings:
    path = tmp_path / "dynamic_filter_facets.json"
    return Settings(dynamic_filter_facets_path=str(path), **overrides)


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
    """Stand-in for the pipeline's ``_RecordLoader`` (ai4data's loader handed the study's record)."""

    _by_idno: dict[str, list[_FakeDoc]] = {}
    _raw_by_idno: dict[str, dict[str, Any]] = {}

    def __init__(self, source: StudySource) -> None:
        self.idno = source.idno
        self.metadata = self._raw_by_idno.get(source.idno, {})

    def get_metadata_handler(self) -> _FakeHandler:
        return _FakeHandler(self._by_idno.get(self.idno, []))


#: The study these tests index (sid 5).
_DOC_1 = StudySource(sid=5, idno="DOC-1", metadata_type="document", metadata={})


class _FakeVec:
    def tolist(self) -> list[float]:
        return [0.1, 0.2]


class _FakeEmbedding:
    """Stand-in for EmbeddingService — avoids loading a real model in tests."""

    input_cap = 1024

    def encode_corpus(self, texts: list[str], show_progress_bar: bool = True) -> list[_FakeVec]:
        return [_FakeVec() for _ in texts]


def test_iter_langdoc_records_bakes_in_cached_extract_filters(tmp_path):
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "DOC-1": [_FakeDoc("a perfectly fine and long enough description", {"idno": "DOC-1", "type": "document"})],
    }
    _FakeLoader._raw_by_idno = {
        "DOC-1": {
            "_extract_filters": {"brand_new_facet_key": ["x"]},
            "_extract_core_fields": {"catalog_id": 5, "idno": "DOC-1"},
        }
    }

    with (
        patch.object(pipeline_module, "_RecordLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="opensearch")
        results = list(
            pipeline_module.iter_langdoc_records(settings, _FakeEmbedding(), [_DOC_1], show_progress_bar=False)
        )

    assert len(results) == 1
    _, _, source = results[0]
    assert source["metadata"]["sid"] == 5
    # OpenSearch stores the flat facets map only; the nested rows are a Qdrant payload shape
    assert source["metadata"]["filter_facets"] == {"brand_new_facet_key": ["x"]}
    assert "filter_fields" not in source["metadata"]

    from nada_ai.search.dynamic_filters import load_dynamic_facet_keys

    assert "brand_new_facet_key" in load_dynamic_facet_keys(settings)


def test_iter_langdoc_records_includes_qdrant_facets(tmp_path):
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "DOC-1": [_FakeDoc("a perfectly fine and long enough description", {"idno": "DOC-1", "type": "document"})],
    }
    _FakeLoader._raw_by_idno = {
        "DOC-1": {"_extract_filters": {"tags": ["health"]}, "_extract_core_fields": {"catalog_id": 5, "idno": "DOC-1"}}
    }

    with (
        patch.object(pipeline_module, "_RecordLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="qdrant")
        results = list(
            pipeline_module.iter_langdoc_records(settings, _FakeEmbedding(), [_DOC_1], show_progress_bar=False)
        )

    _, _, source = results[0]
    assert source["metadata"]["filter_fields"] == [{"key": "tags", "value": ["health"]}]
    assert source["metadata"]["filter_facets"] == {"tags": ["health"]}


def test_iter_langdoc_records_needs_no_embedding_service_when_disabled(tmp_path):
    """embedding_backend=none: yields vec=None for every chunk, like the opensearch_ml path, but with no
    EmbeddingService at all — a None passed as ``embedding`` must never be touched, let alone loaded."""
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "DOC-1": [_FakeDoc("a perfectly fine and long enough description", {"idno": "DOC-1", "type": "document"})],
    }
    _FakeLoader._raw_by_idno = {
        "DOC-1": {"_extract_filters": {}, "_extract_core_fields": {"catalog_id": 5, "idno": "DOC-1"}}
    }

    with (
        patch.object(pipeline_module, "_RecordLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="opensearch", embedding_backend="none")
        results = list(pipeline_module.iter_langdoc_records(settings, None, [_DOC_1], show_progress_bar=False))

    assert len(results) == 1
    doc_id, vec, source = results[0]
    assert vec is None
    assert "embedding" not in source


def test_iter_bulk_actions_attaches_no_pipeline_when_embeddings_are_disabled(tmp_path):
    """Only opensearch_ml needs the ingest pipeline attached (server-side embedding); embedding_backend=none has
    no pipeline to attach — its bulk action is the same plain shape as the local backend's, just with no vector."""
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "DOC-1": [_FakeDoc("a perfectly fine and long enough description", {"idno": "DOC-1", "type": "document"})],
    }
    _FakeLoader._raw_by_idno = {
        "DOC-1": {"_extract_filters": {}, "_extract_core_fields": {"catalog_id": 5, "idno": "DOC-1"}}
    }

    with (
        patch.object(pipeline_module, "_RecordLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="opensearch", embedding_backend="none")
        actions = list(pipeline_module.iter_bulk_actions(settings, None, [_DOC_1], show_progress_bar=False))

    assert len(actions) == 1
    assert "pipeline" not in actions[0]
    assert "embedding" not in actions[0]["_source"]


def test_iter_langdoc_records_skips_a_study_without_extract_data(tmp_path):
    """No cached extract data means no filters and no sid; there is no fallback lookup, so the study is skipped."""
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "DOC-1": [_FakeDoc("a perfectly fine and long enough description", {"idno": "DOC-1", "type": "document"})],
    }
    _FakeLoader._raw_by_idno = {"DOC-1": {}}
    load_errors: list[dict[str, Any]] = []

    with (
        patch.object(pipeline_module, "_RecordLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
        patch("nada_ai.ingest.study_sources.fetch_studies_by_sids", side_effect=AssertionError("no fallback")),
    ):
        results = list(
            pipeline_module.iter_langdoc_records(
                _settings(tmp_path, search_backend="qdrant"),
                _FakeEmbedding(),
                [_DOC_1],
                show_progress_bar=False,
                load_errors=load_errors,
            )
        )

    assert results == []
    assert [e["stage"] for e in load_errors] == ["extract"]


# ---------------------------------------------------------------------------
# stored vectors are reused, so only new or changed chunks are embedded
# ---------------------------------------------------------------------------


class _CountingEmbedding(_FakeEmbedding):
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode_corpus(self, texts: list[str], show_progress_bar: bool = True) -> list[_FakeVec]:
        self.encoded.extend(texts)
        return super().encode_corpus(texts, show_progress_bar)


def _two_chunk_study() -> None:
    _FakeLoader._by_idno = {
        "DOC-1": [
            _FakeDoc("unchanged chunk text", {"idno": "DOC-1", "type": "document", "qfield": "a"}),
            _FakeDoc("brand new chunk text", {"idno": "DOC-1", "type": "document", "qfield": "b"}),
        ]
    }
    _FakeLoader._raw_by_idno = {
        "DOC-1": {"_extract_filters": {}, "_extract_core_fields": {"catalog_id": 5, "idno": "DOC-1"}}
    }


def _run(tmp_path, stored_vectors):
    import nada_ai.ingest.pipeline as pipeline_module

    embedding = _CountingEmbedding()
    with (
        patch.object(pipeline_module, "_RecordLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["qfield"]),
    ):
        results = list(
            pipeline_module.iter_langdoc_records(
                _settings(tmp_path, search_backend="opensearch"),
                embedding,
                [_DOC_1],
                show_progress_bar=False,
                stored_vectors=stored_vectors,
            )
        )
    return embedding, {doc_id: vec for doc_id, vec, _ in results}


def test_stored_vectors_are_reused_and_only_new_chunks_are_embedded(tmp_path):
    _two_chunk_study()
    asked: list[list[str]] = []

    def stored(ids: list[str]) -> dict[str, list[float]]:
        asked.append(ids)
        return {"a": [9.0, 9.0]}

    embedding, vectors = _run(tmp_path, stored)

    assert asked == [["a", "b"]]
    assert embedding.encoded == ["brand new chunk text"]
    assert vectors == {"a": [9.0, 9.0], "b": [0.1, 0.2]}


def test_every_chunk_is_embedded_when_nothing_is_stored(tmp_path):
    _two_chunk_study()
    embedding, vectors = _run(tmp_path, None)
    assert embedding.encoded == ["unchanged chunk text", "brand new chunk text"]
    assert vectors == {"a": [0.1, 0.2], "b": [0.1, 0.2]}


def test_no_embedding_call_when_every_chunk_is_stored(tmp_path):
    _two_chunk_study()
    embedding, vectors = _run(tmp_path, lambda ids: {i: [1.0, 1.0] for i in ids})
    assert embedding.encoded == []
    assert vectors == {"a": [1.0, 1.0], "b": [1.0, 1.0]}
