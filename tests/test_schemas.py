import pytest
from pydantic import ValidationError

from nada_ai.app.schemas import CollapseInnerHits, SearchRequest


def test_collapse_inner_hits_requires_collapse_field():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", collapse_inner_hits=CollapseInnerHits())


def test_collapse_inner_hits_with_field_ok():
    r = SearchRequest(
        query="test",
        collapse_field="idno",
        collapse_inner_hits=CollapseInnerHits(name="variants", size=3),
    )
    assert r.collapse_field == "idno"
    assert r.collapse_inner_hits is not None
    assert r.collapse_inner_hits.name == "variants"
    assert r.collapse_inner_hits.size == 3


def test_include_opensearch_body_defaults_false():
    r = SearchRequest(query="x")
    assert r.include_opensearch_body is False
