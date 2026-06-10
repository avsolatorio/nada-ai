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


def test_include_debug_request_defaults_false():
    r = SearchRequest(query="x")
    assert r.include_debug_request is False


def test_include_opensearch_body_alias_sets_debug_flag():
    r = SearchRequest(query="x", include_opensearch_body=True)
    assert r.include_debug_request is True


def test_search_request_accepts_query_prompt_overrides():
    r = SearchRequest(
        query="gdp",
        query_prompt_name="web_search_query",
        query_prompt="Instruct: x\nQuery: ",
    )
    assert r.query_prompt == "Instruct: x\nQuery: "
    assert r.query_prompt_name == "web_search_query"


def test_search_request_empty_query_prompt_to_none():
    r = SearchRequest(query="gdp", query_prompt="   ")
    assert r.query_prompt is None
