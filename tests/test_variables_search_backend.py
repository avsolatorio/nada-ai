"""``variables_search.py``: the lexical query builder and executor for ``POST /variables/search``."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from opensearchpy.exceptions import NotFoundError

from nada_ai.search.backend.opensearch.mapping import VARIABLE_TEXT_FIELDS
from nada_ai.search.backend.opensearch.variables_search import (
    LEXICAL_FIELDS,
    IndexNotReady,
    VariableFilters,
    VariableSearchJob,
    filter_clauses,
    search_body,
    search_variables,
)


def test_lexical_fields_name_every_variable_text_field_exactly_once() -> None:
    named = [f.split("^")[0] for f in LEXICAL_FIELDS]
    assert sorted(named) == sorted(VARIABLE_TEXT_FIELDS)
    assert len(named) == len(set(named))


def test_filter_clauses_always_requires_published() -> None:
    assert filter_clauses(VariableFilters()) == [{"term": {"published": 1}}]


def test_filter_clauses_add_sids_and_types_when_given() -> None:
    clauses = filter_clauses(VariableFilters(sids=(1, 2), types=("survey",)))
    assert {"term": {"published": 1}} in clauses
    assert {"terms": {"sid": [1, 2]}} in clauses
    assert {"terms": {"dataset_type": ["survey"]}} in clauses


def _job(**overrides: object) -> VariableSearchJob:
    defaults: dict[str, object] = dict(
        client=None,
        index="nada-ai-variables",
        query="diarrhea",
        filters=VariableFilters(),
        limit=15,
        offset=0,
        sort_by="relevance",
    )
    defaults.update(overrides)
    return VariableSearchJob(**defaults)  # type: ignore[arg-type]


def test_search_body_is_a_bool_query_of_the_lexical_multi_match_and_filters() -> None:
    body = search_body(_job())
    assert body["size"] == 15
    assert body["from"] == 0
    assert body["track_total_hits"] is True
    assert body["query"]["bool"]["must"] == [
        {"multi_match": {"query": "diarrhea", "fields": list(LEXICAL_FIELDS), "type": "best_fields"}}
    ]
    assert body["query"]["bool"]["filter"] == [{"term": {"published": 1}}]
    assert "sort" not in body


def test_search_body_sorts_on_the_sortable_fields_when_not_relevance() -> None:
    """``name`` is a text field (OpenSearch refuses to sort on it): the sort goes to its ``name.sort`` keyword."""
    assert search_body(_job(sort_by="name"))["sort"] == [{"name.sort": {"order": "asc", "missing": "_last"}}, "_score"]
    assert search_body(_job(sort_by="title", order="desc"))["sort"] == [
        {"title": {"order": "desc", "missing": "_last"}},
        "_score",
    ]


def test_every_sort_field_is_a_normalized_keyword_in_the_variables_mapping() -> None:
    """What the body sorts on must be sortable in the mapping — the check the body test alone could not make."""
    from nada_ai.search.backend.opensearch.mapping import variables_index_body
    from nada_ai.search.backend.opensearch.variables_search import _SORT_FIELDS

    properties = variables_index_body()["mappings"]["properties"]
    for path in _SORT_FIELDS.values():
        field, _, sub = path.partition(".")
        mapping = properties[field]["fields"][sub] if sub else properties[field]
        assert mapping["type"] == "keyword", path
        assert mapping["normalizer"] == "nada_sort", path


def test_search_variables_parses_hits_and_total() -> None:
    client = AsyncMock()
    client.search.return_value = {
        "took": 4,
        "hits": {
            "total": {"value": 2},
            "hits": [
                {
                    "_score": 5.1,
                    "_source": {
                        "uid": 1,
                        "sid": 10,
                        "idno": "IDNO-1",
                        "name": "hhid",
                        "label": "Household id",
                        "question": None,
                        "title": "Study one",
                        "nation": "Kenya",
                        "dataset_type": "survey",
                    },
                }
            ],
        },
    }
    page = asyncio.run(search_variables(_job(client=client)))
    assert page.found == 2
    assert page.took_ms == 4
    assert len(page.hits) == 1
    hit = page.hits[0]
    assert (hit.uid, hit.sid, hit.idno, hit.name, hit.label, hit.score) == (
        1,
        10,
        "IDNO-1",
        "hhid",
        "Household id",
        5.1,
    )


def test_search_variables_raises_index_not_ready_when_the_index_is_missing() -> None:
    client = AsyncMock()
    client.search.side_effect = NotFoundError(404, "index_not_found_exception", {})
    with pytest.raises(IndexNotReady):
        asyncio.run(search_variables(_job(client=client)))
