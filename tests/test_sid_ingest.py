"""Step 2 of the OpenSearch plan: the NADA internal study id (``sid``) on every indexed document.

``sid`` comes from NADA's metadata-extract API (``core_fields.survey_uid``), which ai4data attaches to the
loaded metadata as ``_extract_core_fields``. Ingest requires it: a study without one is not indexed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ai4data.discovery.catalog import get_langdoc_uuid
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from nada_ai.filters.metadata_extract import MetadataExtractNotConfigured
from nada_ai.ingest import pipeline
from nada_ai.ingest.qdrant_writer import _INTEGER_INDEX_FIELDS
from nada_ai.ingest.service import delete_by_idno_op, delete_by_sid_op, delete_by_sids_op
from nada_ai.search.backend.opensearch.mapping import index_body
from nada_ai.search.documents import langdoc_to_source
from nada_ai.settings import Settings

EXTRACT_MODE = "nada_ai.ingest.pipeline.catalog_extract.is_extract_mode"


def _settings(tmp_path, **overrides: Any) -> Settings:
    return Settings(dynamic_filter_facets_path=str(tmp_path / "facets.json"), **overrides)


def _doc(idno: str, qfield: str = "title", text: str = "some text") -> Document:
    return Document(page_content=text, metadata={"type": "microdata", "idno": idno, "qfield": qfield})


def _raw(sid: Any = 11, filters: Any = None) -> dict[str, Any]:
    """Metadata as ai4data returns it in extract mode."""
    raw: dict[str, Any] = {"_extract_filters": {"countries": [1]} if filters is None else filters}
    if sid is not None:
        raw["_extract_core_fields"] = {"idno": "X", "survey_uid": sid}
    return raw


# ---------------------------------------------------------------------------------------
# metadata.sid on the stored document
# ---------------------------------------------------------------------------------------


def test_langdoc_to_source_stores_sid_as_integer() -> None:
    source = langdoc_to_source(_doc("A"), None, sid=42)
    assert source["metadata"]["sid"] == 42
    assert source["metadata"]["idno"] == "A"


def test_langdoc_to_source_coerces_numeric_sid() -> None:
    assert langdoc_to_source(_doc("A"), None, sid="42")["metadata"]["sid"] == 42  # type: ignore[arg-type]


def test_langdoc_to_source_requires_sid() -> None:
    with pytest.raises(TypeError):
        langdoc_to_source(_doc("A"), None)  # type: ignore[call-arg]


def test_mapping_and_qdrant_index_know_sid() -> None:
    props = index_body(4)["mappings"]["properties"]["metadata"]["properties"]
    assert props["sid"] == {"type": "integer"}
    assert "sid" in _INTEGER_INDEX_FIELDS


# ---------------------------------------------------------------------------------------
# Reading the study's identity from the metadata-extract data
# ---------------------------------------------------------------------------------------


def test_study_extract_returns_sid_core_fields_and_filters() -> None:
    study = pipeline._study_extract(_raw(sid=9))
    assert study == pipeline.StudyExtract(sid=9, core_fields={"idno": "X", "survey_uid": 9}, filters={"countries": [1]})


def test_study_extract_accepts_a_numeric_string_sid() -> None:
    assert pipeline._study_extract(_raw(sid="9")).sid == 9


@pytest.mark.parametrize("sid", [None, 0, -3, "abc", ""])
def test_study_extract_requires_a_valid_sid(sid: Any) -> None:
    with pytest.raises(pipeline.StudyExtractError, match="survey_uid"):
        pipeline._study_extract(_raw(sid=sid))


@pytest.mark.parametrize("raw", [None, {}, {"_extract_core_fields": "nope"}, {"_extract_core_fields": None}])
def test_study_extract_requires_core_fields(raw: Any) -> None:
    with pytest.raises(pipeline.StudyExtractError):
        pipeline._study_extract(raw)


def test_study_extract_requires_an_idno() -> None:
    raw = {"_extract_core_fields": {"survey_uid": 9}, "_extract_filters": {}}
    with pytest.raises(pipeline.StudyExtractError, match="idno"):
        pipeline._study_extract(raw)


def test_study_extract_requires_filters() -> None:
    raw = {"_extract_core_fields": {"survey_uid": 9, "idno": "X"}}
    with pytest.raises(pipeline.StudyExtractError, match="filters"):
        pipeline._study_extract(raw)


def test_indexing_requires_extract_mode() -> None:
    with patch(EXTRACT_MODE, return_value=False), pytest.raises(MetadataExtractNotConfigured, match="internal id"):
        pipeline.require_extract_mode()
    with patch(EXTRACT_MODE, return_value=True):
        pipeline.require_extract_mode()


def test_run_bulk_index_refuses_before_touching_the_index(tmp_path) -> None:
    """The check must come before the writer exists, or ``recreate_index`` would drop the index first."""
    with (
        patch(EXTRACT_MODE, return_value=False),
        patch("nada_ai.ingest.factory.create_ingest_writer") as make_writer,
        pytest.raises(MetadataExtractNotConfigured),
    ):
        pipeline.run_bulk_index(_settings(tmp_path), [("A", "microdata")], recreate_index=True)
    make_writer.assert_not_called()


# ---------------------------------------------------------------------------------------
# End to end through iter_langdoc_records
# ---------------------------------------------------------------------------------------


class _FakeEmbedding:
    def encode_corpus(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts])


def _fake_loader_factory(docs_by_idno: dict[str, list[Document]], raw_by_idno: dict[str, dict]):
    def factory(idno: str, metadata_type: str, force: bool = False, include_resources: bool = True):
        handler = SimpleNamespace(get_langdocs=lambda: docs_by_idno[idno])
        return SimpleNamespace(metadata=raw_by_idno[idno], get_metadata_handler=lambda: handler)

    return factory


def test_every_document_of_a_study_carries_its_sid_and_a_study_without_one_is_not_indexed(tmp_path) -> None:
    docs = {
        "A": [_doc("A", "title", "alpha title"), _doc("A", "abstract", "alpha abstract")],
        "B": [_doc("B", "title", "beta title")],
        "C": [_doc("C", "title", "gamma title")],
    }
    raw = {"A": _raw(sid=11), "B": _raw(sid=22), "C": _raw(sid=None)}
    load_errors: list[dict[str, Any]] = []
    studies: list[pipeline.StudyExtract] = []
    progress = MagicMock()

    with (
        patch.object(pipeline, "MetadataLoader", _fake_loader_factory(docs, raw)),
        # There is no fallback lookup: nothing may call the extract API from the pipeline.
        patch("nada_ai.filters.metadata_extract.fetch_study_records", side_effect=AssertionError("no fallback")),
    ):
        records = list(
            pipeline.iter_langdoc_records(
                _settings(tmp_path, search_backend="opensearch"),
                _FakeEmbedding(),  # type: ignore[arg-type]
                [("A", "microdata"), ("B", "microdata"), ("C", "microdata")],
                show_progress_bar=False,
                progress=progress,
                load_errors=load_errors,
                studies=studies,
            )
        )

    assert [study.sid for study in studies] == [11, 22]  # one per loaded study; C has no sid
    sids = {(source["metadata"]["idno"], source["metadata"]["sid"]) for _id, _vec, source in records}
    assert sids == {("A", 11), ("B", 22)}
    assert len(records) == 3  # C contributed nothing
    assert len(load_errors) == 1
    assert load_errors[0]["idno"] == "C"
    assert load_errors[0]["stage"] == "extract"
    assert "survey_uid" in load_errors[0]["error"]
    progress.mark.assert_any_call("C", ok=False, error=load_errors[0]["error"])
    progress.mark.assert_any_call("A", ok=True)


def test_a_study_without_chunk_documents_still_gets_a_study_record(tmp_path) -> None:
    studies: list[pipeline.StudyExtract] = []
    empty_docs: list[dict[str, Any]] = []
    with patch.object(pipeline, "MetadataLoader", _fake_loader_factory({"A": []}, {"A": _raw(sid=11)})):
        records = list(
            pipeline.iter_langdoc_records(
                _settings(tmp_path),
                _FakeEmbedding(),  # type: ignore[arg-type]
                [("A", "microdata")],
                show_progress_bar=False,
                empty_docs=empty_docs,
                studies=studies,
            )
        )
    assert records == []
    assert [study.sid for study in studies] == [11]
    assert [e["reason"] for e in empty_docs] == ["no_langdocs"]


def test_document_ids_are_unchanged_by_sid(tmp_path) -> None:
    """Ids stay content-based (Qdrant needs UUIDs; a new scheme would duplicate points on re-ingest)."""
    doc = _doc("A", "title", "alpha title")
    with patch.object(pipeline, "MetadataLoader", _fake_loader_factory({"A": [doc]}, {"A": _raw(sid=11)})):
        ((doc_id, _vec, _source),) = pipeline.iter_langdoc_records(
            _settings(tmp_path),
            _FakeEmbedding(),  # type: ignore[arg-type]
            [("A", "microdata")],
            show_progress_bar=False,
        )
    assert doc_id == get_langdoc_uuid(doc)


# ---------------------------------------------------------------------------------------
# Delete by sid
# ---------------------------------------------------------------------------------------


class _KeepOpen:
    """Wraps a local in-memory Qdrant client so the code under test can ``close()`` it without losing data."""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def close(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _qdrant_with_studies() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection("nada-test", vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE))
    points = [
        qm.PointStruct(id=1, vector=[1, 0, 0, 0], payload={"metadata": {"sid": 1, "idno": "A", "qfield": "title"}}),
        qm.PointStruct(id=2, vector=[0, 1, 0, 0], payload={"metadata": {"sid": 1, "idno": "A", "qfield": "abstract"}}),
        qm.PointStruct(id=3, vector=[0, 0, 1, 0], payload={"metadata": {"sid": 2, "idno": "B", "qfield": "title"}}),
        qm.PointStruct(id=4, vector=[0, 0, 0, 1], payload={"metadata": {"sid": 3, "idno": "C", "qfield": "title"}}),
    ]
    client.upsert("nada-test", points=points)
    return client


def _remaining_ids(client: QdrantClient) -> set[int]:
    points, _ = client.scroll("nada-test", limit=100)
    return {int(p.id) for p in points}


def test_qdrant_delete_by_sid_removes_only_that_study() -> None:
    client = _qdrant_with_studies()
    settings = Settings(search_backend="qdrant", qdrant_collection_name="nada-test")
    with patch("nada_ai.ingest.qdrant_writer._client", return_value=_KeepOpen(client)):
        result = delete_by_sid_op(settings, 1)
    assert result["backend"] == "qdrant"
    assert result["sids"] == [1]
    assert _remaining_ids(client) == {3, 4}  # the other studies are untouched


def test_qdrant_delete_by_several_sids() -> None:
    client = _qdrant_with_studies()
    settings = Settings(search_backend="qdrant", qdrant_collection_name="nada-test")
    with patch("nada_ai.ingest.qdrant_writer._client", return_value=_KeepOpen(client)):
        delete_by_sids_op(settings, [1, 2, 2])
    assert _remaining_ids(client) == {4}


@pytest.mark.parametrize("sids", [[], [0], [-4], [1, 0]])
def test_delete_by_sids_rejects_empty_or_non_positive(sids: list[int]) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        delete_by_sids_op(Settings(search_backend="qdrant"), sids)


def test_opensearch_delete_by_sid_removes_chunks_study_and_variable_documents() -> None:
    client = MagicMock()
    client.delete_by_query.side_effect = [{"deleted": 3, "total": 3}, {"deleted": 2, "total": 2}, {"deleted": 9}]
    settings = Settings(search_backend="opensearch", index_name="nada-test")
    with patch("nada_ai.ingest.service.build_client", return_value=client):
        result = delete_by_sids_op(settings, [5, 7])
    calls = client.delete_by_query.call_args_list
    assert [c.kwargs["index"] for c in calls] == ["nada-test", "nada-test-studies", "nada-test-variables"]
    assert calls[0].kwargs["body"]["query"]["bool"]["should"] == [{"terms": {"metadata.sid": [5, 7]}}]
    assert calls[1].kwargs["body"]["query"]["bool"]["should"] == [{"terms": {"sid": [5, 7]}}]
    assert calls[2].kwargs["body"]["query"]["bool"]["should"] == [{"terms": {"sid": [5, 7]}}]
    client.search.assert_not_called()  # no idno given, nothing to look up
    assert result == {
        "backend": "opensearch",
        "index": "nada-test",
        "sids": [5, 7],
        "deleted": 3,
        "total": 3,
        "studies_index": "nada-test-studies",
        "studies_deleted": 2,
        "variables_index": "nada-test-variables",
        "variables_deleted": 9,
    }


def test_opensearch_delete_by_idno_also_removes_chunks_stored_under_another_idno() -> None:
    """The chunk idno comes from the record's schema and can differ from NADA's; the study index knows the sid."""
    client = MagicMock()
    client.search.return_value = {"hits": {"hits": [{"_source": {"idno": "PC11_A02-28-v22", "sid": 4}}]}}
    client.delete_by_query.side_effect = [{"deleted": 1, "total": 1}, {"deleted": 1, "total": 1}, {"deleted": 4}]
    settings = Settings(search_backend="opensearch", index_name="nada-test")
    with patch("nada_ai.ingest.service.build_client", return_value=client):
        result = delete_by_idno_op(settings, "PC11_A02-28-v22")
    assert client.search.call_args.kwargs["index"] == "nada-test-studies"
    chunk_should = client.delete_by_query.call_args_list[0].kwargs["body"]["query"]["bool"]["should"]
    assert {"terms": {"metadata.idno": ["PC11_A02-28-v22"]}} in chunk_should
    assert {"terms": {"metadata.sid": [4]}} in chunk_should
    assert result["studies_deleted"] == 1 and result["idno"] == "PC11_A02-28-v22"


def test_opensearch_delete_by_idno_also_removes_the_studys_variables() -> None:
    """A deleted study's variables must not stay searchable: they carry the same NADA idno and sid."""
    client = MagicMock()
    client.search.return_value = {"hits": {"hits": [{"_source": {"idno": "PC11_A02-28-v22", "sid": 4}}]}}
    client.delete_by_query.side_effect = [{"deleted": 1}, {"deleted": 1}, {"deleted": 4}]
    settings = Settings(search_backend="opensearch", index_name="nada-test")
    with patch("nada_ai.ingest.service.build_client", return_value=client):
        result = delete_by_idno_op(settings, "PC11_A02-28-v22")
    call = client.delete_by_query.call_args_list[2]
    assert call.kwargs["index"] == "nada-test-variables"
    assert call.kwargs["ignore_unavailable"] is True  # a deployment that never indexed variables has no such index
    should = call.kwargs["body"]["query"]["bool"]["should"]
    assert {"terms": {"idno": ["PC11_A02-28-v22"]}} in should and {"terms": {"sid": [4]}} in should
    assert result["variables_deleted"] == 4
