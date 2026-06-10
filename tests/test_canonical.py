"""Canonical path helpers."""

from nada_ai.search.canonical import facet_field_whitelist, stored_filter_field_name


def test_stored_paths_under_metadata():
    assert stored_filter_field_name("idno") == "metadata.idno"
    assert stored_filter_field_name("page_content") == "page_content"


def test_whitelist_contains_core_facets():
    w = facet_field_whitelist()
    assert "type" in w
    assert "geographies" in w
