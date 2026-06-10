"""Facet bucket coercion and OpenSearch facet wiring."""

from nada_ai.app.schemas import FacetBucket, coerce_search_facets
from nada_ai.search.backend.opensearch.queries import merge_facets_into_body


def test_coerce_search_facets():
    raw = {"type": [{"value": "indicator", "count": 3}, {"value": "document", "count": 1}]}
    out = coerce_search_facets(raw)
    assert out is not None
    assert isinstance(out["type"][0], FacetBucket)
    assert out["type"][0].value == "indicator"
    assert out["type"][0].count == 3


def test_merge_facets_into_body():
    body: dict = {"query": {"match_all": {}}}
    merge_facets_into_body(body, ["type", "source"], None)
    assert "aggs" in body
    assert "type" in body["aggs"]
    assert "terms" in body["aggs"]["type"]
