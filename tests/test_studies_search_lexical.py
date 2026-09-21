"""POST /studies/search, lexical mode: the keyword query, every match paged with exact totals, and the endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opensearchpy.exceptions import NotFoundError
from starlette.testclient import TestClient

from nada_ai.app.main import app, state
from nada_ai.app.studies_schemas import (
    MAX_OFFSET,
    EffectiveMode,
    ErrorResponse,
    SortField,
    SortOrder,
    StudyFilters,
    StudySearchResponse,
)
from nada_ai.search.backend.opensearch.mapping import studies_index_body
from nada_ai.search.backend.opensearch.studies_search import (
    EXECUTORS,
    LEXICAL_FIELDS,
    IndexNotReady,
    SearchJob,
    keyword_body,
    lexical,
    lexical_query,
    relevance_sort,
)
from nada_ai.search.backend.opensearch.studies_semantic import StudyPolicy
from nada_ai.settings import Settings

# ---------------------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------------------


def test_lexical_fields_and_boosts_are_the_ones_nada_search_has_always_used() -> None:
    assert LEXICAL_FIELDS == (
        "idno.text^60",
        "title^40",
        "nation^30",
        "authoring_entity^10",
        "keywords^10",
        "abstract",
        "methodology",
        "var_keywords^15",
    )


def test_every_lexical_field_exists_in_the_study_mapping() -> None:
    props = studies_index_body()["mappings"]["properties"]
    for field in LEXICAL_FIELDS:
        name = field.split("^")[0]
        root, _, sub = name.partition(".")
        assert root in props, name
        assert props[root]["type"] == "text" or sub in props[root].get("fields", {}), name


def test_idno_is_searchable_as_text_and_still_an_exact_keyword() -> None:
    idno = studies_index_body()["mappings"]["properties"]["idno"]
    assert idno["type"] == "keyword"
    assert idno["fields"]["text"] == {"type": "text", "analyzer": "nada_text"}


def test_query_uses_fuzzy_most_fields_with_minimum_should_match() -> None:
    match = lexical_query("consumer price index")["multi_match"]
    assert match["query"] == "consumer price index"
    assert match["type"] == "most_fields"
    assert match["minimum_should_match"] == "2<75%"
    assert match["fuzziness"] == "AUTO:5,9"
    assert match["prefix_length"] == 2
    assert match["fields"] == list(LEXICAL_FIELDS)


def test_query_text_is_data_not_syntax() -> None:
    """A multi_match never interprets operators, so user text cannot change the query structure."""
    text = 'title:"x" OR (a AND NOT b) *'
    assert lexical_query(text)["multi_match"]["query"] == text


def test_the_keyword_query_has_no_score_cutoff() -> None:
    """Every study the match rules accept is a keyword match; the best simply score highest."""
    body = keyword_body("poverty", StudyFilters(), sort=relevance_sort(SortOrder.desc), limit=15, offset=0)
    assert "min_score" not in body
    assert "min_score" not in str(body["query"])


def test_relevance_sort_orders_by_score_then_sid() -> None:
    assert relevance_sort(SortOrder.desc) == [{"_score": {"order": "desc"}}, {"sid": {"order": "asc"}}]
    assert relevance_sort(SortOrder.asc) == [{"_score": {"order": "asc"}}, {"sid": {"order": "asc"}}]


def test_body_pages_in_opensearch_and_counts_every_match() -> None:
    body = keyword_body(
        "poverty",
        StudyFilters(types=["survey"], countries=[16]),
        sort=relevance_sort(SortOrder.desc),
        limit=15,
        offset=30,
    )
    assert (body["from"], body["size"], body["track_total_hits"]) == (30, 15, True)
    assert body["query"]["bool"]["must"] == [lexical_query("poverty")]
    assert {"terms": {"filter_facets.countries": [16]}} in body["query"]["bool"]["filter"]
    # `types` narrows the hits and the total but not the per-type counts, which are an aggregation over all matches
    assert "dataset_type" not in str(body["query"])
    assert body["post_filter"] == {"terms": {"filter_facets.dataset_type": ["survey"]}}
    assert body["aggs"]["by_type"]["terms"]["field"] == "filter_facets.dataset_type"


def test_body_can_leave_studies_out() -> None:
    body = keyword_body(
        "x", StudyFilters(), sort=relevance_sort(SortOrder.desc), limit=15, offset=0, exclude_sids=[4, 2]
    )
    assert body["query"]["bool"]["must_not"] == [{"terms": {"sid": [4, 2]}}]
    assert "must_not" not in keyword_body("x", StudyFilters(), sort=[], limit=1, offset=0)["query"]["bool"]


def test_body_can_add_studies_that_match_without_the_keyword() -> None:
    body = keyword_body("x", StudyFilters(), sort=[], limit=15, offset=0, union_sids=[4, 2])
    assert body["query"]["bool"]["must"] == [
        {"bool": {"should": [lexical_query("x"), {"terms": {"sid": [4, 2]}}], "minimum_should_match": 1}}
    ]


# ---------------------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------------------


def _hit(sid: int, score: float | None, dataset_type: str = "survey") -> dict[str, Any]:
    return {
        "_score": score,
        "_source": {"sid": sid, "idno": f"IDNO_{sid}", "filter_facets": {"dataset_type": [dataset_type]}},
    }


def _response(
    hits: list[dict[str, Any]], total: int | None = None, counts: dict[str, int] | None = None, took: int = 3
) -> dict[str, Any]:
    """A keyword search response: the page's hits, the total of all matches, and the per-type counts."""
    by_type = counts if counts is not None else _count_types(hits)
    return {
        "took": took,
        "hits": {"total": {"value": len(hits) if total is None else total}, "hits": hits},
        "aggregations": {"by_type": {"buckets": [{"key": k, "doc_count": v} for k, v in by_type.items()]}},
    }


def _count_types(hits: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for h in hits:
        kind = h["_source"]["filter_facets"]["dataset_type"][0]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _job(client: Any, **overrides: Any) -> SearchJob:
    fields: dict[str, Any] = {
        "client": client,
        "index": "studies",
        "chunk_index": "chunks",
        "policy": StudyPolicy.from_settings(Settings()),
        "query": "poverty",
        "filters": StudyFilters(),
        "sort_by": SortField.relevance,
        "sort_order": SortOrder.desc,
        "limit": 15,
        "offset": 0,
    }
    fields.update(overrides)
    return SearchJob(**fields)


def _client(*responses: dict[str, Any], indexed: int = 10) -> MagicMock:
    client = MagicMock()
    client.search = AsyncMock(side_effect=list(responses))
    client.count = AsyncMock(return_value={"count": indexed})
    return client


def _run(job: SearchJob):
    return asyncio.run(lexical(job))


PAGE = [_hit(4, 9.5, "survey"), _hit(2, 7.25, "document"), _hit(9, 7.25, "survey"), _hit(1, 3.0, "table")]


def test_relevance_order_is_the_score_order_with_scores_and_matched_by() -> None:
    page = _run(_job(_client(_response(PAGE))))
    assert [h["sid"] for h in page.hits] == [4, 2, 9, 1]
    assert [h["score"] for h in page.hits] == [9.5, 7.25, 7.25, 3.0]
    assert all(h["matched_by"] == ["lexical"] for h in page.hits)
    assert (page.found, page.counts_by_type) == (4, {"survey": 2, "document": 1, "table": 1})
    assert len(page.request_bodies) == 1


def test_found_and_the_counts_cover_every_match_not_just_the_page() -> None:
    """Two hundred and seventy-one studies match: the page shows three, and nothing else is cut."""
    response = _response(PAGE[:3], total=271, counts={"survey": 200, "document": 60, "table": 11})
    client = _client(response)
    page = _run(_job(client, limit=3, offset=0))
    assert (page.found, page.counts_by_type) == (271, {"survey": 200, "document": 60, "table": 11})
    assert [h["sid"] for h in page.hits] == [4, 2, 9]


def test_paging_is_done_by_opensearch() -> None:
    client = _client(_response(PAGE[1:3], total=271))
    page = _run(_job(client, limit=2, offset=1))
    body = client.search.call_args.kwargs["body"]
    assert (body["from"], body["size"]) == (1, 2)
    assert [h["sid"] for h in page.hits] == [2, 9]
    assert page.found == 271


def test_ascending_relevance_asks_for_the_lowest_scores_first() -> None:
    client = _client(_response(PAGE[::-1]))
    page = _run(_job(client, sort_order=SortOrder.asc))
    assert client.search.call_args.kwargs["body"]["sort"] == relevance_sort(SortOrder.asc)
    assert [h["sid"] for h in page.hits] == [1, 9, 2, 4]


def test_types_filter_narrows_found_and_hits_but_not_the_tab_counts() -> None:
    surveys = [PAGE[0], PAGE[2]]
    response = _response(surveys, total=2, counts={"survey": 2, "document": 1, "table": 1})
    client = _client(response)
    page = _run(_job(client, filters=StudyFilters(types=["survey"])))
    assert client.search.call_args.kwargs["body"]["post_filter"] == {
        "terms": {"filter_facets.dataset_type": ["survey"]}
    }
    assert [h["sid"] for h in page.hits] == [4, 9]
    assert page.found == 2
    assert page.counts_by_type == {"survey": 2, "document": 1, "table": 1}


def test_another_sort_orders_all_the_matches_by_that_sort_in_one_request() -> None:
    rows = [_hit(2, None, "document"), _hit(4, None, "survey")]
    client = _client(_response(rows, total=40))
    page = _run(_job(client, sort_by=SortField.title, sort_order=SortOrder.asc, limit=2))
    body = client.search.call_args.kwargs["body"]
    assert body["sort"][0] == {"title_sort": {"order": "asc", "missing": "_last"}}
    assert client.search.call_count == 1  # no second query: the matches are the whole set
    assert [h["sid"] for h in page.hits] == [2, 4]
    assert [h["score"] for h in page.hits] == [None, None]  # ordered by title, so there is no relevance score
    assert page.found == 40


def test_no_match_returns_nothing_and_asks_nothing_more() -> None:
    client = _client(_response([], counts={}), indexed=5)
    page = _run(_job(client, sort_by=SortField.title, sort_order=SortOrder.asc))
    assert (page.found, page.hits, page.counts_by_type) == (0, [], {})
    assert client.search.call_count == 1


def test_an_empty_index_is_not_ready_but_no_match_in_a_populated_index_is_fine() -> None:
    with pytest.raises(IndexNotReady):
        _run(_job(_client(_response([], counts={}), indexed=0)))
    assert _run(_job(_client(_response([], counts={}), indexed=5))).found == 0


def test_a_missing_index_is_not_ready() -> None:
    client = MagicMock()
    client.search = AsyncMock(side_effect=NotFoundError(404, "index_not_found_exception", {}))
    with pytest.raises(IndexNotReady):
        _run(_job(client))


def test_lexical_is_registered() -> None:
    assert EXECUTORS[EffectiveMode.lexical] is lexical


# ---------------------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------------------


@contextmanager
def _running(monkeypatch: pytest.MonkeyPatch, client_mock: Any) -> Iterator[TestClient]:
    monkeypatch.setenv("NADA_SEARCH_BACKEND", "opensearch")
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        previous = state.client
        state.client = client_mock
        try:
            yield client
        finally:
            state.client = previous


def _post(client: TestClient, body: dict[str, Any]):
    return client.post("/studies/search", json=body)


def test_a_query_runs_lexical_and_returns_scored_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    os_client = _client(_response(PAGE[:3], total=4, counts={"survey": 2, "document": 1, "table": 1}))
    with _running(monkeypatch, os_client) as client:
        response = _post(client, {"query": "  poverty  ", "limit": 3, "mode": "lexical"})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert body.invariant_violations() == []
    assert body.applied.mode is EffectiveMode.lexical
    assert body.applied.query == "poverty"  # trimmed
    assert (body.applied.sort.by, body.applied.sort.order) == (SortField.relevance, SortOrder.desc)
    assert [h.sid for h in body.hits] == [4, 2, 9]
    assert [h.rank for h in body.hits] == [1, 2, 3]
    assert all(h.matched_by[0].value == "lexical" and h.score is not None for h in body.hits)
    assert (body.found, body.truncated) == (4, False)
    assert body.search_counts_by_type == {"survey": 2, "document": 1, "table": 1}
    assert os_client.search.call_args.kwargs["body"]["size"] == 3


def test_more_matches_than_can_be_paged_are_reported_as_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_hit(i, 10.0 - i * 0.01) for i in range(1, 4)]
    with _running(
        monkeypatch, _client(_response(rows, total=MAX_OFFSET + 5, counts={"survey": MAX_OFFSET + 5}))
    ) as client:
        body = StudySearchResponse.model_validate(_post(client, {"query": "survey", "mode": "lexical"}).json())
    assert (body.found, body.truncated) == (MAX_OFFSET + 5, True)
    assert body.invariant_violations() == []


def test_a_page_beyond_the_paging_depth_is_out_of_range_for_a_query_too(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _client(_response([]))) as client:
        response = _post(client, {"query": "x", "mode": "lexical", "limit": 100, "offset": MAX_OFFSET - 99})
    assert response.status_code == 422
    assert ErrorResponse.model_validate(response.json()).error.code.value == "offset_out_of_range"


def test_explicit_lexical_and_another_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_hit(2, None, "document"), _hit(4, None)]
    with _running(monkeypatch, _client(_response(rows, total=2))) as client:
        body = StudySearchResponse.model_validate(
            _post(client, {"query": "x", "mode": "lexical", "sort": {"by": "title", "order": "asc"}}).json()
        )
    assert body.applied.mode is EffectiveMode.lexical
    assert [h.sid for h in body.hits] == [2, 4]
    assert body.invariant_violations() == []


def test_types_filter_over_a_query_keeps_the_contract_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response([PAGE[0], PAGE[2]], total=2, counts={"survey": 2, "document": 1, "table": 1})
    with _running(monkeypatch, _client(response)) as client:
        body = StudySearchResponse.model_validate(
            _post(client, {"query": "x", "mode": "lexical", "filters": {"types": ["survey"]}}).json()
        )
    assert body.found == 2
    assert body.search_counts_by_type == {"survey": 2, "document": 1, "table": 1}
    assert body.invariant_violations() == []


def test_a_query_that_matches_nothing_is_an_empty_result_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _client(_response([], counts={}), indexed=11)) as client:
        response = _post(client, {"query": "xyzzy qwerty flurbo", "mode": "lexical"})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert (body.found, body.hits, body.search_counts_by_type) == (0, [], {})
    assert body.invariant_violations() == []


def test_an_empty_index_is_index_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _client(_response([], counts={}), indexed=0)) as client:
        response = _post(client, {"query": "x", "mode": "lexical"})
    assert response.status_code == 503
    assert ErrorResponse.model_validate(response.json()).error.code.value == "index_not_ready"


def test_debug_lists_every_opensearch_request(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _client(_response([_hit(4, 9.5)]))) as client:
        body = _post(client, {"query": "x", "mode": "lexical", "include_debug": True}).json()
    assert len(body["debug"]["opensearch_requests"]) == 1
