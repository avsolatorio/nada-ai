"""Facet counts of the study search (``facets`` in ``POST /studies/search``; docs/studies-search-facets.md)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from nada_ai.app.main import app, state
from nada_ai.app.studies_schemas import (
    FacetKey,
    FacetRequest,
    SortField,
    SortOrder,
    StudyFilters,
    StudySearchRequest,
    StudySearchResponse,
)
from nada_ai.search.backend.opensearch.studies_search import (
    SearchJob,
    browse,
    exact_idno_match,
    facets_body,
    filter_clauses,
    hybrid,
    lexical,
    lexical_query,
    parse_facets,
    semantic,
)
from nada_ai.search.backend.opensearch.studies_semantic import StudyPolicy
from nada_ai.settings import Settings

PUBLISHED = {"term": {"filter_facets.published": 1}}
COUNTRIES = {"terms": {"filter_facets.countries": [16]}}
TAGS = {"terms": {"filter_facets.tags": ["health"]}}
SURVEYS = {"terms": {"filter_facets.dataset_type": ["survey"]}}


# ---------------------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------------------


def test_empty_facet_request_asks_for_nothing() -> None:
    assert StudySearchRequest.model_validate({"facets": {"keys": [], "user": []}}).facets is None
    request = StudySearchRequest.model_validate({"facets": {"keys": ["tags", "tags"], "user": ["a", "a"]}})
    assert request.facets == FacetRequest(keys=[FacetKey.tags], user=["a"])


def test_unknown_facet_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        StudySearchRequest.model_validate({"facets": {"keys": ["topics"]}})


@pytest.mark.parametrize(
    ("filters", "clause"),
    [
        ({"regions": [4]}, {"terms": {"filter_facets.regions": [4]}}),
        ({"topics": [12]}, {"terms": {"filter_facets.topics": [12]}}),
        ({"databases": ["WB_WDI"]}, {"terms": {"filter_facets.ts_databases": ["WB_WDI"]}}),
    ],
)
def test_new_filters_are_one_clause_each(filters: dict[str, Any], clause: dict[str, Any]) -> None:
    assert filter_clauses(StudyFilters.model_validate(filters)) == [PUBLISHED, clause]


# ---------------------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------------------


def _aggs(filters: StudyFilters, keys: list[str], user: list[str] | None = None, match: Any = None) -> dict:
    body = facets_body(match, filters, FacetRequest(keys=keys, user=user or []), 500)
    assert (body["size"], body["track_total_hits"]) == (0, False)
    return body


def test_each_facet_ignores_its_own_filter_and_keeps_the_others() -> None:
    filters = StudyFilters(countries=[16], tags=["health"], types=["survey"], repository="demo")
    body = _aggs(filters, ["countries", "tags", "regions"])
    # the scope is the query; the facet filters and the tab are per aggregation
    assert body["query"] == {"bool": {"filter": [PUBLISHED, {"term": {"filter_facets.repositories": "demo"}}]}}
    aggs = body["aggs"]
    assert aggs["f_countries"]["filter"] == {"bool": {"filter": [TAGS, SURVEYS]}}
    assert aggs["f_tags"]["filter"] == {"bool": {"filter": [COUNTRIES, SURVEYS]}}
    assert aggs["f_regions"]["filter"] == {"bool": {"filter": [COUNTRIES, TAGS, SURVEYS]}}


def test_a_facet_counts_its_field_and_its_selected_values() -> None:
    aggs = _aggs(StudyFilters(countries=[16, 191]), ["countries", "collections"])["aggs"]
    values = aggs["f_countries"]["aggs"]["values"]["terms"]
    assert (values["field"], values["size"]) == ("filter_facets.countries", 500)
    assert values["order"] == [{"_count": "desc"}, {"_key": "asc"}]
    assert aggs["f_countries"]["aggs"]["selected"] == {
        "filters": {
            "filters": {
                "16": {"term": {"filter_facets.countries": 16}},
                "191": {"term": {"filter_facets.countries": 191}},
            }
        }
    }
    # nothing selected: no selected aggregation; collections count the membership list
    assert "selected" not in aggs["f_collections"]["aggs"]
    assert aggs["f_collections"]["aggs"]["values"]["terms"]["field"] == "filter_facets.repositories"


def test_years_is_a_range_that_ignores_the_year_filter() -> None:
    aggs = _aggs(StudyFilters(year_from=2000, tags=["health"]), ["years"])["aggs"]
    assert aggs["f_years"] == {
        "filter": {"bool": {"filter": [TAGS]}},
        "aggs": {"min": {"min": {"field": "filter_facets.years"}}, "max": {"max": {"field": "filter_facets.years"}}},
    }


def test_user_facets_ignore_their_own_terms_only() -> None:
    filters = StudyFilters(facets={"author": [7], "funding": [9]})
    aggs = _aggs(filters, [], ["author"])["aggs"]
    assert aggs["u_author"]["filter"] == {"bool": {"filter": [{"terms": {"filter_facets.fq_funding": [9]}}]}}
    assert aggs["u_author"]["aggs"]["values"]["terms"]["field"] == "filter_facets.fq_author"
    assert set(aggs["u_author"]["aggs"]["selected"]["filters"]["filters"]) == {"7"}


def test_topics_always_apply() -> None:
    aggs = _aggs(StudyFilters(topics=[3]), ["countries"])["aggs"]
    assert aggs["f_countries"]["filter"] == {"bool": {"filter": [{"terms": {"filter_facets.topics": [3]}}]}}


def test_the_match_selects_the_result_set() -> None:
    match = {"term": {"idno": "X"}}
    assert _aggs(StudyFilters(), ["tags"], match=match)["query"] == {"bool": {"filter": [PUBLISHED, match]}}


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------


def _bucket_agg(values: dict[Any, int], other: int = 0, selected: dict[str, int] | None = None) -> dict:
    agg: dict[str, Any] = {
        "values": {
            "sum_other_doc_count": other,
            "buckets": [{"key": k, "doc_count": v} for k, v in values.items()],
        }
    }
    if selected is not None:
        agg["selected"] = {"buckets": {k: {"doc_count": v} for k, v in selected.items()}}
    return agg


def test_parse_lists_values_as_strings_and_adds_selected_ones_that_match() -> None:
    request = FacetRequest(keys=[FacetKey.countries, FacetKey.tags, FacetKey.years], user=["author"])
    response = {
        "aggregations": {
            "f_countries": _bucket_agg({16: 40, 35: 7}, other=3, selected={"16": 40, "77": 2, "99": 0}),
            "f_tags": _bucket_agg({"health": 2}),
            "f_years": {"min": {"value": 1990.0}, "max": {"value": 2022.0}},
            "u_author": _bucket_agg({7: 1}),
        }
    }
    facets = parse_facets(response, request)
    countries = facets.values[FacetKey.countries]
    assert [(b.value, b.count) for b in countries.buckets] == [
        ("16", 40),
        ("35", 7),
        ("77", 2),
    ]  # a value with no studies is never listed
    assert countries.more is True
    assert facets.values[FacetKey.tags].more is False
    assert (facets.years.min, facets.years.max) == (1990, 2022)
    assert [(b.value, b.count) for b in facets.user["author"].buckets] == [("7", 1)]


def test_parse_years_of_an_empty_set() -> None:
    request = FacetRequest(keys=[FacetKey.years])
    facets = parse_facets({"aggregations": {"f_years": {"min": {"value": None}, "max": {"value": None}}}}, request)
    assert (facets.years.min, facets.years.max) == (None, None)


# ---------------------------------------------------------------------------------------
# Executors: which result set each mode counts
# ---------------------------------------------------------------------------------------


FACET_RESPONSE = {
    "took": 2,
    "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
    "aggregations": {"f_tags": _bucket_agg({"health": 5})},
}


def _page_response(rows: list[tuple[int, str]]) -> dict:
    return {
        "took": 3,
        "hits": {
            "total": {"value": len(rows), "relation": "eq"},
            "hits": [
                {
                    "_source": {"sid": sid, "idno": idno, "title": idno, "filter_facets": {"dataset_type": ["survey"]}},
                    "_score": 1.0,
                    "sort": [None, None, sid],
                }
                for sid, idno in rows
            ],
        },
        "aggregations": {"by_type": {"buckets": [{"key": "survey", "doc_count": len(rows)}]}},
    }


def _knn_response(sids: list[int]) -> dict:
    return {
        "took": 5,
        "hits": {
            "total": {"value": len(sids), "relation": "eq"},
            "hits": [
                {
                    "_score": 0.9,
                    "_source": {"metadata": {"sid": sid}},
                    "fields": {"metadata.sid": [sid]},
                    "inner_hits": {"best": {"hits": {"hits": [{"_score": 0.9, "_source": {"metadata": {}}}]}}},
                }
                for sid in sids
            ],
        },
    }


def _client(rows: list[tuple[int, str]]) -> tuple[MagicMock, list[dict]]:
    """A fake OpenSearch: a facet request gets the facet counts, every other request the same page."""
    facet_bodies: list[dict] = []

    async def search(index: str, body: dict, request_timeout: float) -> dict:
        if any(name.startswith(("f_", "u_")) for name in body.get("aggs", {})):
            facet_bodies.append(body)
            return FACET_RESPONSE
        return _page_response(rows)

    client = MagicMock()
    client.search = AsyncMock(side_effect=search)
    client.count = AsyncMock(return_value={"count": 3})
    return client, facet_bodies


def _job(client: Any, query: str | None, **kw: Any) -> SearchJob:
    return SearchJob(
        client=client,
        index="studies",
        chunk_index="chunks",
        query=query,
        filters=StudyFilters(),
        sort_by=SortField.relevance if query else SortField.title,
        sort_order=SortOrder.desc if query else SortOrder.asc,
        limit=10,
        offset=0,
        policy=StudyPolicy.from_settings(Settings()),
        facets=FacetRequest(keys=[FacetKey.tags]),
        **kw,
    )


def test_no_facets_asked_means_no_facet_request() -> None:
    client, facet_bodies = _client([(1, "A")])
    page = asyncio.run(browse(replace(_job(client, None), facets=None)))
    assert page.facets is None and facet_bodies == []


def test_browse_counts_every_study_of_the_filters() -> None:
    client, facet_bodies = _client([(1, "A")])
    page = asyncio.run(browse(_job(client, None)))
    assert page.facets is not None and page.facets.values[FacetKey.tags].buckets[0].count == 5
    assert facet_bodies[0]["query"] == {"bool": {"filter": [PUBLISHED]}}
    assert facet_bodies[0] in page.request_bodies


def test_lexical_counts_the_keyword_matches() -> None:
    client, facet_bodies = _client([(1, "A")])
    page = asyncio.run(lexical(_job(client, "child labour")))
    assert page.facets is not None
    assert facet_bodies[0]["query"]["bool"]["filter"] == [PUBLISHED, lexical_query("child labour")]


def test_exact_idno_counts_the_idno_match() -> None:
    client, facet_bodies = _client([(1, "ABC")])
    page = asyncio.run(exact_idno_match(_job(client, "ABC")))
    assert page is not None and page.facets is not None
    assert facet_bodies[0]["query"]["bool"]["filter"][-1] == {"term": {"idno": "ABC"}}


async def _embed(_: str) -> list[float]:
    return [0.1, 0.2]


def _semantic_client(sids: list[int]) -> tuple[MagicMock, list[dict]]:
    client, facet_bodies = _client([(sid, f"S{sid}") for sid in sids])
    page_search = client.search.side_effect

    async def search(index: str, body: dict, request_timeout: float) -> dict:
        if index == "chunks":
            return _knn_response(sids)
        return await page_search(index, body, request_timeout)

    client.search = AsyncMock(side_effect=search)
    return client, facet_bodies


def test_semantic_counts_the_semantic_studies() -> None:
    client, facet_bodies = _semantic_client([4, 9])
    page = asyncio.run(semantic(_job(client, "malaria", embed=_embed)))
    assert page.facets is not None
    assert facet_bodies[0]["query"]["bool"]["filter"][-1] == {"terms": {"sid": [4, 9]}}


def test_hybrid_counts_the_keyword_matches_and_the_semantic_studies() -> None:
    client, facet_bodies = _semantic_client([4, 9])
    page = asyncio.run(hybrid(_job(client, "malaria", embed=_embed)))
    assert page.facets is not None
    assert len(facet_bodies) == 1
    assert facet_bodies[0]["query"]["bool"]["filter"][-1] == {
        "bool": {"should": [lexical_query("malaria"), {"terms": {"sid": [4, 9]}}], "minimum_should_match": 1}
    }


def test_hybrid_with_another_sort_counts_the_same_set() -> None:
    client, facet_bodies = _semantic_client([4, 9])
    job = replace(_job(client, "malaria", embed=_embed), sort_by=SortField.year)
    page = asyncio.run(hybrid(job))
    assert page.facets is not None and len(facet_bodies) == 1
    assert facet_bodies[0]["query"]["bool"]["filter"][-1]["bool"]["should"][1] == {"terms": {"sid": [4, 9]}}


# ---------------------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------------------


@contextmanager
def _running(monkeypatch: pytest.MonkeyPatch, client_mock: Any, backend: str = "opensearch") -> Iterator[TestClient]:
    monkeypatch.setenv("NADA_SEARCH_BACKEND", backend)
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")
    with TestClient(app) as client:
        previous = state.client
        state.client = client_mock
        try:
            yield client
        finally:
            state.client = previous


def test_the_response_carries_the_facets(monkeypatch: pytest.MonkeyPatch) -> None:
    client_mock, _ = _client([(1, "A")])
    with _running(monkeypatch, client_mock) as client:
        response = client.post("/studies/search", json={"facets": {"keys": ["tags"]}})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert body.invariant_violations() == []
    assert response.json()["facets"] == {
        "values": {"tags": {"buckets": [{"value": "health", "count": 5}], "more": False}},
        "user": {},
        "years": None,
    }


def test_no_facets_key_means_no_facets(monkeypatch: pytest.MonkeyPatch) -> None:
    client_mock, facet_bodies = _client([(1, "A")])
    with _running(monkeypatch, client_mock) as client:
        response = client.post("/studies/search", json={})
    assert response.json()["facets"] is None and facet_bodies == []


def test_facets_need_the_facets_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, MagicMock(), backend="qdrant") as client:
        response = client.post("/studies/search", json={"facets": {"keys": ["tags"]}})
    # qdrant has no study search at all: the first capability it lacks is the one reported
    assert response.status_code == 501
    assert response.json()["error"]["details"]["capability"] == "studies_search"
