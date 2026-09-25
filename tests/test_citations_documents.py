"""``citations.py``: the citation document built from NADA's extract data."""

from __future__ import annotations

import json

import pytest

from nada_ai.search.backend.opensearch.citations import citation_bulk_action, citation_to_source, format_authors
from nada_ai.search.backend.opensearch.mapping import CITATION_TEXT_FIELDS


def test_authors_json_is_flattened_to_searchable_names() -> None:
    raw = json.dumps(
        [
            {"lname": "Bentley", "fname": "Olivia Victoria", "initial": ""},
            {"lname": "Smith", "fname": "Ann", "initial": "B"},
        ]
    )
    assert format_authors(raw) == "Olivia Victoria Bentley; Ann B Smith"


def test_authors_that_are_plain_text_or_empty_are_kept_or_dropped() -> None:
    assert format_authors("Smith, J.; Doe, A.") == "Smith, J.; Doe, A."
    assert format_authors('{"not": "a list"}') == '{"not": "a list"}'
    assert format_authors("") == ""
    assert format_authors(None) == ""
    assert format_authors("[{}]") == ""


def _citation() -> dict:
    return {
        "metadata": {
            "title": "  Poverty and health  ",
            "subtitle": "A study",
            "authors": json.dumps([{"lname": "Doe", "fname": "Jane", "initial": ""}]),
            "abstract": "About poverty.",
            "keywords": "poverty, health",
            "notes": None,
            "doi": "10.1/ABC",
        },
        "core_fields": {"citation_id": 7, "citation_uuid": "u-7", "doi": "10.1/ABC"},
        "filters": {"doctype": 3, "published": 1, "ctype": "book", "pub_date": 2019},
    }


def test_a_citation_document_holds_the_text_filters_and_sort_fields() -> None:
    c = _citation()
    source = citation_to_source(c["core_fields"], c["metadata"], c["filters"])
    assert source == {
        "citation_id": 7,
        "uuid": "u-7",
        "title": "Poverty and health",
        "subtitle": "A study",
        "authors": "Jane Doe",
        "abstract": "About poverty.",
        "keywords": "poverty, health",
        "title_sort": "Poverty and health",
        "doi": "10.1/ABC",
        "ctype": "book",
        "pub_year": 2019,
        "published": 1,
    }
    assert set(source) >= {f for f in CITATION_TEXT_FIELDS if f != "notes"}  # empty notes are omitted


def test_an_unpublished_or_unflagged_citation_is_stored_as_not_published() -> None:
    c = _citation()
    assert citation_to_source(c["core_fields"], c["metadata"], {"published": 0})["published"] == 0
    assert citation_to_source(c["core_fields"], c["metadata"], {})["published"] == 0


def test_a_citation_without_an_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="citation_id"):
        citation_to_source({}, {}, {})


def test_the_bulk_action_is_keyed_by_the_citation_id_so_a_reindex_replaces_it() -> None:
    action = citation_bulk_action("idx-citations", _citation())
    assert action["_id"] == "7" and action["_index"] == "idx-citations" and action["_op_type"] == "index"
    assert action["_source"]["citation_id"] == 7
