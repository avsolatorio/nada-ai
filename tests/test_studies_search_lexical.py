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
    PHRASE_BOOST,
    PHRASE_SLOP,
    IndexNotReady,
    SearchJob,
    _title_is_complete_match,
    exact_idno_match,
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


def test_what_identifies_and_describes_a_study_counts_most() -> None:
    assert LEXICAL_FIELDS == (
        "idno.text^60",
        "title^40",
        "nation^30",
        "authoring_entity^10",
        "abstract^10",
        "keywords",
        "methodology",
        "var_keywords",
    )


def test_the_long_blobs_count_least() -> None:
    """``keywords`` and ``var_keywords`` hold thousands of characters of unrelated text: a word found somewhere in them
    must not outweigh the abstract, or the title."""
    boosts = {f.split("^")[0]: float(f.split("^")[1]) if "^" in f else 1.0 for f in LEXICAL_FIELDS}
    assert boosts["keywords"] == boosts["var_keywords"] == 1.0
    assert boosts["abstract"] > boosts["keywords"] and boosts["title"] > boosts["abstract"]


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
    match = lexical_query("consumer")["multi_match"]
    assert match["query"] == "consumer"
    assert match["type"] == "most_fields"
    assert match["minimum_should_match"] == "2<75%"
    assert match["fuzziness"] == "AUTO:5,9"
    assert match["prefix_length"] == 2
    assert match["fields"] == list(LEXICAL_FIELDS)


def test_query_text_is_data_not_syntax() -> None:
    """A multi_match never interprets operators, so user text cannot change the query structure."""
    text = 'title:"x" OR (a AND NOT b) *'
    query = lexical_query(text)["bool"]
    assert query["must"][0]["multi_match"]["query"] == text
    assert query["should"][0]["multi_match"]["query"] == text
    assert lexical_query("poverty")["multi_match"]["query"] == "poverty"


def test_a_single_word_has_no_phrase_bonus() -> None:
    assert list(lexical_query("poverty")) == ["multi_match"]


def test_two_or_more_words_also_score_as_a_phrase_without_changing_what_matches() -> None:
    query = lexical_query("foreign direct investment")["bool"]
    assert list(query) == ["must", "should"]  # the phrase is optional: it adds to the score of studies already matched
    (match,) = query["must"]
    (phrase,) = query["should"]
    assert match["multi_match"]["type"] == "most_fields" and match["multi_match"]["minimum_should_match"] == "2<75%"
    assert phrase["multi_match"] == {
        "query": "foreign direct investment",
        "type": "phrase",
        "fields": list(LEXICAL_FIELDS),
        "slop": PHRASE_SLOP,
        "boost": PHRASE_BOOST,
    }
    assert PHRASE_BOOST > 1


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


def _is_idno_probe(body: dict[str, Any]) -> bool:
    """The exact-idno check every relevance search makes first: a filter-only query with a ``term`` on ``idno``,
    no ``must`` and no ``aggs``. Recognized so it never consumes a response meant for the search under test."""
    clauses = body["query"]["bool"]
    return "must" not in clauses and any("term" in f and "idno" in f["term"] for f in clauses.get("filter", []))


def _client(*responses: dict[str, Any], indexed: int = 10, idno_match: dict[str, Any] | None = None) -> MagicMock:
    """``idno_match``, if given, is what the exact-idno probe sees; by default it finds nothing, so every test below
    keeps answering ``responses`` in order for the search itself."""
    client = MagicMock()
    queue = list(responses)

    async def search(index: str, body: dict[str, Any]) -> dict[str, Any]:
        if _is_idno_probe(body):
            return idno_match if idno_match is not None else _response([], counts={})
        return queue.pop(0)

    client.search = AsyncMock(side_effect=search)
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
# Exact idno match
# ---------------------------------------------------------------------------------------


def _idno_hit(sid: int, idno: str, dataset_type: str = "survey") -> dict[str, Any]:
    return {"_source": {"sid": sid, "idno": idno, "filter_facets": {"dataset_type": [dataset_type]}}}


def _idno_response(hits: list[dict[str, Any]]) -> dict[str, Any]:
    return {"took": 1, "hits": {"total": {"value": len(hits)}, "hits": hits}}


def test_title_completeness_ignores_stopwords_and_word_order() -> None:
    assert _title_is_complete_match("high resolution angola", "High Resolution Poverty Map, Angola, 2020")
    assert _title_is_complete_match("angola high resolution", "High Resolution Poverty Map, Angola, 2020")
    assert _title_is_complete_match("district census handbook", "Census of India 2011 - District Census Handbook")


def test_title_completeness_requires_every_real_word() -> None:
    assert not _title_is_complete_match("high resolution angola", "High Resolution Imagery")  # missing "angola"
    assert not _title_is_complete_match("", "Anything")  # nothing to require: never a match
    assert not _title_is_complete_match("the of a", "Anything")  # only stopwords: never a match


def test_title_completeness_is_case_and_punctuation_insensitive() -> None:
    assert _title_is_complete_match("HIGH-RESOLUTION angola!", "high resolution, Angola.")


def test_a_matching_idno_is_a_term_query_on_the_idno_field() -> None:
    client = _client(idno_match=_idno_response([_idno_hit(234, "AGO_2020_HRPM_GEO_v01_M", "geospatial")]))
    page = asyncio.run(exact_idno_match(_job(client, query="AGO_2020_HRPM_GEO_v01_M")))
    assert page is not None
    assert (page.found, page.counts_by_type) == (1, {"geospatial": 1})
    assert page.hits == [{"sid": 234, "idno": "AGO_2020_HRPM_GEO_v01_M", "score": None, "matched_by": ["idno"]}]
    probe = client.search.call_args.kwargs["body"]
    assert probe["query"]["bool"]["filter"][-1] == {"term": {"idno": "AGO_2020_HRPM_GEO_v01_M"}}
    assert "must" not in probe["query"]["bool"]


def test_the_match_is_case_and_accent_insensitive_because_the_field_is_normalized() -> None:
    """OpenSearch does the folding (the ``idno`` field's ``nada_sort`` normalizer): the query here is sent as
    typed, unchanged, and it is the index side that makes ago_2020... and AGO_2020... compare equal."""
    client = _client(idno_match=_idno_response([_idno_hit(234, "AGO_2020_HRPM_GEO_v01_M")]))
    page = asyncio.run(exact_idno_match(_job(client, query="ago_2020_hrpm_geo_v01_m")))
    assert page is not None and page.found == 1
    probe = client.search.call_args.kwargs["body"]
    assert probe["query"]["bool"]["filter"][-1] == {"term": {"idno": "ago_2020_hrpm_geo_v01_m"}}


def test_no_match_is_none_not_an_empty_page() -> None:
    client = _client(idno_match=_idno_response([]))
    assert asyncio.run(exact_idno_match(_job(client, query="does-not-exist"))) is None


def test_a_multi_word_query_is_never_checked_as_an_idno() -> None:
    client = _client()  # no idno_match configured; a probe call would raise (queue is empty)
    assert asyncio.run(exact_idno_match(_job(client, query="poverty in rwanda"))) is None
    client.search.assert_not_called()


def test_the_sidebar_filters_apply_to_the_idno_match_too() -> None:
    client = _client(idno_match=_idno_response([_idno_hit(234, "AGO_2020_HRPM_GEO_v01_M", "geospatial")]))
    asyncio.run(exact_idno_match(_job(client, query="AGO_2020_HRPM_GEO_v01_M", filters=StudyFilters(countries=[16]))))
    probe = client.search.call_args.kwargs["body"]
    assert {"terms": {"filter_facets.countries": [16]}} in probe["query"]["bool"]["filter"]


def test_the_dataset_type_tab_applies_to_the_idno_match_too() -> None:
    client = _client(idno_match=_idno_response([]))
    asyncio.run(exact_idno_match(_job(client, query="AGO_2020_HRPM_GEO_v01_M", filters=StudyFilters(types=["survey"]))))
    probe = client.search.call_args.kwargs["body"]
    assert {"terms": {"filter_facets.dataset_type": ["survey"]}} in probe["query"]["bool"]["filter"]


def test_matches_are_paged_like_any_other_result() -> None:
    hits = [_idno_hit(i, f"IDNO_{i}") for i in (1, 2, 3)]
    client = _client(idno_match=_idno_response(hits))
    page = asyncio.run(exact_idno_match(_job(client, query="IDNO", limit=2, offset=1)))
    assert page is not None
    assert (page.found, [h["sid"] for h in page.hits]) == (3, [2, 3])


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
        response = _post(client, {"query": "poverty", "mode": "lexical", "limit": 100, "offset": MAX_OFFSET - 99})
    assert response.status_code == 422
    assert ErrorResponse.model_validate(response.json()).error.code.value == "offset_out_of_range"


def test_explicit_lexical_and_another_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_hit(2, None, "document"), _hit(4, None)]
    with _running(monkeypatch, _client(_response(rows, total=2))) as client:
        body = StudySearchResponse.model_validate(
            _post(client, {"query": "poverty", "mode": "lexical", "sort": {"by": "title", "order": "asc"}}).json()
        )
    assert body.applied.mode is EffectiveMode.lexical
    assert [h.sid for h in body.hits] == [2, 4]
    assert body.invariant_violations() == []


def test_types_filter_over_a_query_keeps_the_contract_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response([PAGE[0], PAGE[2]], total=2, counts={"survey": 2, "document": 1, "table": 1})
    with _running(monkeypatch, _client(response)) as client:
        body = StudySearchResponse.model_validate(
            _post(client, {"query": "poverty", "mode": "lexical", "filters": {"types": ["survey"]}}).json()
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
        response = _post(client, {"query": "poverty", "mode": "lexical"})
    assert response.status_code == 503
    assert ErrorResponse.model_validate(response.json()).error.code.value == "index_not_ready"


def test_debug_lists_every_opensearch_request(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _client(_response([_hit(4, 9.5)]))) as client:
        body = _post(client, {"query": "poverty", "mode": "lexical", "include_debug": True}).json()
    assert len(body["debug"]["opensearch_requests"]) == 1


def test_the_route_answers_an_exact_idno_without_running_the_search_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idno check runs before dispatch, for every mode, so a keyword-search response left in the queue is
    never touched."""
    os_client = _client(idno_match=_idno_response([_idno_hit(234, "AGO_2020_HRPM_GEO_v01_M", "geospatial")]))
    with _running(monkeypatch, os_client) as client:
        response = _post(client, {"query": "AGO_2020_HRPM_GEO_v01_M", "mode": "lexical"})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert body.invariant_violations() == []
    assert (body.found, body.search_counts_by_type) == (1, {"geospatial": 1})
    assert [h.sid for h in body.hits] == [234]
    assert body.hits[0].matched_by[0].value == "idno" and body.hits[0].score is None
    os_client.search.assert_called_once()  # only the idno probe: the queue behind it was never touched
