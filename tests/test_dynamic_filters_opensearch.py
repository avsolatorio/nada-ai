"""Tests for OpenSearch dynamic filter query translation."""

from nada_ai.search.backend.opensearch.queries import build_filters, merge_facets_into_body
from nada_ai.search.dynamic_filters import dynamic_facet_aggs, dynamic_filters_to_opensearch_clauses


def test_dynamic_filters_nested_clauses():
    clauses = dynamic_filters_to_opensearch_clauses({"countries": [181, 182]})
    assert len(clauses) == 1
    assert "nested" in clauses[0]
    assert clauses[0]["nested"]["path"] == "metadata.filter_fields"


def test_build_filters_includes_dynamic():
    clauses = build_filters({"type": "document", "countries": [181]})
    assert len(clauses) == 2
    assert any("nested" in c for c in clauses)
    assert any("term" in c for c in clauses)


def test_dynamic_facet_aggs_shape():
    aggs = dynamic_facet_aggs(["countries", "regions"])
    assert "countries" in aggs
    assert aggs["countries"]["nested"]["path"] == "metadata.filter_fields"


def test_merge_facets_static_and_dynamic():
    body: dict = {"query": {"match_all": {}}}
    merge_facets_into_body(body, ["type"], ["countries"])
    assert "type" in body["aggs"]
    assert "countries" in body["aggs"]
