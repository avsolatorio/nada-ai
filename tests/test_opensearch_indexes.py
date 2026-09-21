"""Step 3 of the OpenSearch plan: the study index, the chunk index, and flat filter fields (unit tests)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nada_ai.ingest import pipeline
from nada_ai.ingest.opensearch_writer import OpenSearchIngestWriter
from nada_ai.search.backend.opensearch.mapping import (
    filter_facets_mapping,
    index_body,
    new_index_generation,
    studies_index_body,
)
from nada_ai.search.backend.opensearch.studies import study_bulk_action, study_to_source
from nada_ai.settings import Settings

CORE = {
    "survey_uid": 4,
    "idno": "PC11_A02-28-v22",
    "title": "Census of India 2011",
    "nation": "India",
    "authoring_entity": "Registrar General",
    "abstract": "  Population counts.  ",
    "keywords": None,
    "methodology": "",
    "var_keywords": "age sex",
    "year_start": "2011",
    "year_end": 2011,
    "created": 1700000000,
    "changed": "1700000500",
    "total_views": 12,
    "total_downloads": None,
    "varcount": 30,
}
FILTERS = {"dataset_type": "survey", "countries": [102], "years": [2011], "fq_author": [7, 8], "tags": []}


# ---------------------------------------------------------------------------------------
# Study document
# ---------------------------------------------------------------------------------------


def test_study_document_fields() -> None:
    source = study_to_source(4, CORE, FILTERS)
    assert source["sid"] == 4
    assert source["idno"] == "PC11_A02-28-v22"  # NADA's idno
    assert source["abstract"] == "Population counts."
    assert source["title_sort"] == "Census of India 2011"
    assert source["nation_sort"] == "India"
    assert (source["year_start"], source["year_end"], source["changed"]) == (2011, 2011, 1700000500)
    assert source["varcount"] == 30


def test_study_document_omits_empty_fields() -> None:
    source = study_to_source(4, CORE, FILTERS)
    for field in ("keywords", "methodology", "total_downloads"):
        assert field not in source


def test_study_document_filters_are_one_flat_field_per_key() -> None:
    facets = study_to_source(4, CORE, FILTERS)["filter_facets"]
    assert facets == {"dataset_type": ["survey"], "countries": ["102"], "years": ["2011"], "fq_author": ["7", "8"]}


def test_study_document_needs_an_idno() -> None:
    with pytest.raises(ValueError, match="idno"):
        study_to_source(4, {"survey_uid": 4}, {})


def test_study_bulk_action_id_is_the_sid() -> None:
    action = study_bulk_action("idx-studies", 4, CORE, FILTERS)
    assert action["_index"] == "idx-studies"
    assert action["_id"] == "4"
    assert action["_op_type"] == "index"


# ---------------------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------------------


def _types(node: Any) -> list[str]:
    """Every mapped ``type`` in a mapping tree."""
    found: list[str] = []
    if isinstance(node, dict):
        if isinstance(node.get("type"), str):
            found.append(node["type"])
        for value in node.values():
            found.extend(_types(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_types(value))
    return found


def test_no_nested_fields_in_either_index() -> None:
    assert "nested" not in _types(index_body(4))
    assert "nested" not in _types(studies_index_body())


def test_chunk_index_has_flat_filter_facets_and_no_filter_fields() -> None:
    metadata = index_body(4)["mappings"]["properties"]["metadata"]["properties"]
    assert "filter_fields" not in metadata
    facets = metadata["filter_facets"]["properties"]
    assert facets["countries"] == {"type": "integer"}
    assert facets["dataset_type"] == {"type": "keyword"}


def test_study_index_is_strict_and_typed() -> None:
    body = studies_index_body()
    mappings = body["mappings"]
    assert mappings["dynamic"] == "strict"
    props = mappings["properties"]
    assert props["sid"] == {"type": "integer"}
    assert props["title"]["analyzer"] == "nada_text"
    assert props["title_sort"]["normalizer"] == "nada_sort"
    assert "embedding" not in props
    assert "knn" not in body["settings"]["index"]


def test_filter_templates_type_user_facets_as_integers_and_the_rest_as_keywords() -> None:
    _, templates = filter_facets_mapping("metadata.")
    by_name = {name: spec for template in templates for name, spec in template.items()}
    assert by_name["filter_facets_user_facets"]["path_match"] == "metadata.filter_facets.fq_*"
    assert by_name["filter_facets_user_facets"]["mapping"]["type"] == "integer"
    assert by_name["filter_facets_other_keys"]["mapping"]["type"] == "keyword"
    # the user-facet rule must come first or the catch-all would win
    assert list(by_name) == ["filter_facets_user_facets", "filter_facets_other_keys"]


def test_both_indexes_share_one_filter_mapping() -> None:
    chunk_props = index_body(4)["mappings"]["properties"]["metadata"]["properties"]["filter_facets"]
    study_props = studies_index_body()["mappings"]["properties"]["filter_facets"]
    assert chunk_props == study_props


def test_index_generations_are_unique() -> None:
    assert new_index_generation() != new_index_generation()


# ---------------------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------------------


def _client(existing: set[str] | None = None) -> MagicMock:
    """A cluster stand-in that remembers which indexes exist, so deletes and creates take effect."""
    present = set(existing or ())
    client = MagicMock()
    client.indices.exists.side_effect = lambda index: index in present
    client.indices.delete.side_effect = lambda index: present.discard(index)
    client.indices.create.side_effect = lambda index, body: present.add(index)
    client.indices.get_mapping.return_value = {}
    return client


def test_ensure_index_stamps_generation_and_embedding_info() -> None:
    client = _client()
    settings = Settings(index_name="chunks", embedding_model_id="model-x")
    pipeline.ensure_index(client, settings, 384)
    body = client.indices.create.call_args.kwargs["body"]
    meta = body["mappings"]["_meta"]
    assert meta["embedding_model"] == "model-x"
    assert meta["embedding_dim"] == 384
    assert meta["generation"]


def test_ensure_studies_index_creates_once() -> None:
    settings = Settings(index_name="chunks")
    client = _client()
    pipeline.ensure_studies_index(client, settings)
    kwargs = client.indices.create.call_args.kwargs
    assert kwargs["index"] == "chunks-studies"
    assert kwargs["body"]["mappings"]["_meta"]["generation"]

    existing = _client({"chunks-studies"})
    pipeline.ensure_studies_index(existing, settings)
    existing.indices.create.assert_not_called()


# ---------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------


class _Embedding:
    def embedding_dimension(self) -> int:
        return 4


def _studies() -> list[pipeline.StudyExtract]:
    return [
        pipeline.StudyExtract(sid=4, core_fields=CORE, filters=FILTERS),
        pipeline.StudyExtract(sid=2, core_fields={**CORE, "survey_uid": 2, "idno": "EGY"}, filters={}),
    ]


def _run_writer(settings: Settings, client: MagicMock, *, recreate: bool, bulk_results: list[Any]):
    bulk_calls: list[list[dict[str, Any]]] = []

    def fake_bulk(_client, actions, **_):
        bulk_calls.append(list(actions))
        return bulk_results[len(bulk_calls) - 1]

    def fake_iter_bulk_actions(_settings, _embedding, _pairs, *, studies, **_):
        studies.extend(_studies())
        yield {
            "_op_type": "index",
            "_index": settings.index_name,
            "_id": "chunk-1",
            "_source": {"metadata": {"sid": 4}},
        }
        yield {
            "_op_type": "index",
            "_index": settings.index_name,
            "_id": "chunk-2",
            "_source": {"metadata": {"sid": 4}},
        }

    with (
        patch("nada_ai.ingest.opensearch_writer.build_client", return_value=client),
        patch("nada_ai.ingest.opensearch_writer.bulk", side_effect=fake_bulk),
        patch("nada_ai.ingest.opensearch_writer.iter_bulk_actions", side_effect=fake_iter_bulk_actions),
    ):
        result = OpenSearchIngestWriter(settings).run_bulk(
            [("PC11_A02-28-v22", "microdata")],
            embedding=_Embedding(),
            recreate_target=recreate,  # type: ignore[arg-type]
        )
    return result, bulk_calls


def test_writer_indexes_chunks_then_one_study_document_per_study() -> None:
    settings = Settings(index_name="chunks", opensearch_put_composable_index_template=False)
    client = _client()
    (success, errors), calls = _run_writer(settings, client, recreate=False, bulk_results=[(5, []), (2, [])])

    assert (success, errors) == (5, None)
    chunk_actions, study_actions = calls
    assert [a["_id"] for a in chunk_actions] == ["chunk-1", "chunk-2"]
    assert {a["_index"] for a in study_actions} == {"chunks-studies"}
    assert sorted(a["_id"] for a in study_actions) == ["2", "4"]
    assert {c.kwargs["index"] for c in client.indices.create.call_args_list} == {"chunks", "chunks-studies"}


def test_writer_recreate_drops_both_indexes_first() -> None:
    settings = Settings(index_name="chunks", opensearch_put_composable_index_template=False)
    client = _client({"chunks", "chunks-studies"})
    _run_writer(settings, client, recreate=True, bulk_results=[(0, []), (0, [])])
    deleted = [c.kwargs["index"] for c in client.indices.delete.call_args_list]
    assert deleted == ["chunks", "chunks-studies"]
    assert client.indices.create.call_count == 2


def test_writer_reports_errors_from_both_indexes() -> None:
    settings = Settings(index_name="chunks", opensearch_put_composable_index_template=False)
    (success, errors), _ = _run_writer(
        settings,
        _client(),
        recreate=False,
        bulk_results=[(4, [{"index": {"_id": "c"}}]), (1, [{"index": {"_id": "4"}}])],
    )
    assert success == 4
    assert errors == [{"index": {"_id": "c"}}, {"index": {"_id": "4"}}]


def test_writer_installs_both_templates_when_enabled() -> None:
    settings = Settings(index_name="chunks", opensearch_put_composable_index_template=True)
    client = _client()
    _run_writer(settings, client, recreate=False, bulk_results=[(0, []), (0, [])])
    assert client.indices.put_index_template.call_count == 2


def test_writer_prunes_chunks_that_are_not_in_this_run() -> None:
    settings = Settings(index_name="chunks", opensearch_put_composable_index_template=False)
    client = _client()
    client.delete_by_query.return_value = {"deleted": 7}
    _run_writer(settings, client, recreate=False, bulk_results=[(2, []), (2, [])])

    client.delete_by_query.assert_called_once()
    call = client.delete_by_query.call_args.kwargs
    assert call["index"] == "chunks"
    clauses = {
        clause["bool"]["filter"][0]["term"]["metadata.sid"]: clause["bool"]["must_not"][0]["ids"]["values"]
        for clause in call["body"]["query"]["bool"]["should"]
    }
    # study 4 wrote two chunks this run, so only other chunks of study 4 go; study 2 wrote none, so all of its go
    assert clauses == {4: ["chunk-1", "chunk-2"], 2: []}


def test_pruning_is_batched() -> None:
    settings = Settings(index_name="chunks")
    client = _client()
    client.delete_by_query.return_value = {"deleted": 0}
    writer = OpenSearchIngestWriter(settings)
    assert writer._prune_stale_chunks(client, {sid: {f"c{sid}"} for sid in range(1, 121)}) == 0
    assert client.delete_by_query.call_count == 3  # 120 studies in batches of 50
