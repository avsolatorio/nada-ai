"""Tests for OpenSearch dynamic filter query translation."""

from nada_ai.search.backend.opensearch.queries import build_filters, merge_facets_into_body
from nada_ai.search.dynamic_filters import dynamic_facet_aggs, dynamic_filters_to_opensearch_clauses


def test_dynamic_filters_are_flat_clauses():
    clauses = dynamic_filters_to_opensearch_clauses({"countries": [181, 182], "tags": ["health"]})
    assert clauses == [
        {"terms": {"metadata.filter_facets.countries": ["181", "182"]}},
        {"term": {"metadata.filter_facets.tags": "health"}},
    ]


def test_no_nested_queries_anywhere():
    clauses = build_filters({"type": "document", "countries": [181]})
    assert "nested" not in str(clauses)


def test_build_filters_includes_dynamic():
    clauses = build_filters({"type": "document", "countries": [181]})
    assert {"term": {"metadata.type": "document"}} in clauses
    assert {"term": {"metadata.filter_facets.countries": "181"}} in clauses


def test_dynamic_facet_aggs_shape():
    aggs = dynamic_facet_aggs(["countries", "regions"])
    assert aggs["countries"] == {"terms": {"field": "metadata.filter_facets.countries", "size": 200}}
    assert set(aggs) == {"countries", "regions"}


def test_merge_facets_static_and_dynamic():
    body: dict = {"query": {"match_all": {}}}
    merge_facets_into_body(body, ["type"], ["countries"])
    assert "type" in body["aggs"]
    assert "countries" in body["aggs"]
