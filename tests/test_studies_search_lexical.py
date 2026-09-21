"""POST /studies/search, lexical mode (step 6 of the OpenSearch plan): query building, the cut set, and the endpoint."""

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
    lexical,
    lexical_body,
    lexical_query,
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


def test_body_takes_the_top_cap_across_all_types_ordered_by_score_then_sid() -> None:
    body = lexical_body("poverty", StudyFilters(types=["survey"], countries=[16]), 100)
    assert (body["from"], body["size"], body["track_total_hits"]) == (0, 100, True)
    assert body["sort"] == [{"_score": {"order": "desc"}}, {"sid": {"order": "asc"}}]
    assert body["query"]["bool"]["must"] == [lexical_query("poverty")]
    filters = body["query"]["bool"]["filter"]
    assert {"terms": {"filter_facets.countries": [16]}} in filters
    assert "dataset_type" not in str(filters)  # `types` is applied after the cut, not before
    assert "post_filter" not in body


# ---------------------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------------------


def _hit(sid: int, score: float, dataset_type: str = "survey") -> dict[str, Any]:
    return {
        "_score": score,
        "_source": {"sid": sid, "idno": f"IDNO_{sid}", "filter_facets": {"dataset_type": [dataset_type]}},
    }


def _response(hits: list[dict[str, Any]], total: int | None = None, took: int = 3) -> dict[str, Any]:
    return {"took": took, "hits": {"total": {"value": len(hits) if total is None else total}, "hits": hits}}


def _browse_response(rows: list[tuple[int, str]], found: int, counts: dict[str, int], took: int = 2) -> dict[str, Any]:
    return {
        "took": took,
        "hits": {
            "total": {"value": found},
            "hits": [{"_source": {"sid": sid, "idno": idno}} for sid, idno in rows],
        },
        "aggregations": {"by_type": {"buckets": [{"key": k, "doc_count": v} for k, v in counts.items()]}},
    }


def _job(client: Any, **overrides: Any) -> SearchJob:
    fields: dict[str, Any] = {
        "client": client,
        "index": "studies",
        "chunk_index": "chunks",
        "policy": StudyPolicy.from_settings(Settings(studies_lexical_relative_cutoff=0.0)),
        "query": "poverty",
        "filters": StudyFilters(),
        "sort_by": SortField.relevance,
        "sort_order": SortOrder.desc,
        "limit": 15,
        "offset": 0,
        "result_cap": 100,
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


CUT = [_hit(4, 9.5, "survey"), _hit(2, 7.25, "document"), _hit(9, 7.25, "survey"), _hit(1, 3.0, "table")]


def test_relevance_order_is_the_score_order_with_scores_and_matched_by() -> None:
    page = _run(_job(_client(_response(CUT))))
    assert [h["sid"] for h in page.hits] == [4, 2, 9, 1]
    assert [h["score"] for h in page.hits] == [9.5, 7.25, 7.25, 3.0]
    assert all(h["matched_by"] == ["lexical"] for h in page.hits)
    assert (page.found, page.truncated, page.result_cap) == (4, False, 100)
    assert page.counts_by_type == {"survey": 2, "document": 1, "table": 1}
    assert len(page.request_bodies) == 1


def test_ascending_relevance_reverses_the_cut_set() -> None:
    page = _run(_job(_client(_response(CUT)), sort_order=SortOrder.asc))
    assert [h["sid"] for h in page.hits] == [1, 9, 2, 4]


def test_paging_slices_the_cut_set() -> None:
    client = _client(_response(CUT))
    page = _run(_job(client, limit=2, offset=1))
    assert [h["sid"] for h in page.hits] == [2, 9]
    assert page.found == 4
    beyond = _run(_job(_client(_response(CUT)), limit=2, offset=10))
    assert (beyond.hits, beyond.found) == ([], 4)


def test_types_filter_narrows_found_and_hits_but_not_the_tab_counts() -> None:
    page = _run(_job(_client(_response(CUT)), filters=StudyFilters(types=["survey"])))
    assert [h["sid"] for h in page.hits] == [4, 9]
    assert page.found == 2
    assert page.counts_by_type == {"survey": 2, "document": 1, "table": 1}


def test_more_matches_than_the_cap_are_reported_as_truncated() -> None:
    cut = [_hit(i, 10.0 - i) for i in range(1, 4)]
    page = _run(_job(_client(_response(cut, total=250)), result_cap=3))
    assert (page.found, page.truncated, page.result_cap) == (3, True, 3)
    assert page.counts_by_type == {"survey": 3}


def test_weak_keyword_matches_are_cut_off_by_the_relative_cutoff() -> None:
    policy = StudyPolicy.from_settings(Settings(studies_lexical_relative_cutoff=0.4))
    page = _run(_job(_client(_response(CUT)), policy=policy))  # 3.0 is under 40% of 9.5
    assert [h["sid"] for h in page.hits] == [4, 2, 9]
    assert (page.found, page.counts_by_type) == (3, {"survey": 2, "document": 1})


def test_truncated_means_strong_matches_beyond_the_cap_not_a_cut_tail() -> None:
    policy = StudyPolicy.from_settings(Settings(studies_lexical_relative_cutoff=0.4))
    strong = [_hit(i, 10.0 - i * 0.1) for i in range(1, 4)]
    assert _run(_job(_client(_response(strong, total=250)), result_cap=3, policy=policy)).truncated is True
    with_tail = [*strong[:2], _hit(3, 0.5)]  # the cutoff removed the tail, so nothing strong was left behind
    assert _run(_job(_client(_response(with_tail, total=250)), result_cap=3, policy=policy)).truncated is False


def test_exactly_the_cap_is_not_truncated() -> None:
    cut = [_hit(i, 10.0 - i) for i in range(1, 4)]
    assert _run(_job(_client(_response(cut, total=3)), result_cap=3)).truncated is False


def test_another_sort_re_sorts_the_same_set_inside_opensearch() -> None:
    cut = [_hit(4, 9.5, "survey"), _hit(2, 7.25, "document"), _hit(9, 7.25, "survey")]
    resorted = _browse_response(
        [(2, "IDNO_2"), (4, "IDNO_4"), (9, "IDNO_9")], found=2, counts={"survey": 2, "document": 1}
    )
    client = _client(_response(cut), resorted)
    page = _run(
        _job(client, sort_by=SortField.title, sort_order=SortOrder.asc, filters=StudyFilters(types=["survey"]), limit=3)
    )

    first, second = (call.kwargs["body"] for call in client.search.call_args_list)
    assert second["query"]["bool"]["filter"][-1] == {"terms": {"sid": [4, 2, 9]}}  # pins the cut set
    assert second["post_filter"] == {"terms": {"filter_facets.dataset_type": ["survey"]}}  # types narrows found only
    assert second["sort"][0] == {"title_sort": {"order": "asc", "missing": "_last"}}
    assert "must" not in second["query"]["bool"]  # no second relevance query
    # the order and totals come from the re-sort; each hit keeps its relevance score
    assert [h["sid"] for h in page.hits] == [2, 4, 9]
    assert [h["score"] for h in page.hits] == [7.25, 9.5, 7.25]
    assert (page.found, page.counts_by_type) == (2, {"survey": 2, "document": 1})
    assert len(page.request_bodies) == 2 and first["size"] == 100


def test_no_match_returns_nothing_and_asks_nothing_more() -> None:
    client = _client(_response([]))
    page = _run(_job(client, sort_by=SortField.title, sort_order=SortOrder.asc))
    assert (page.found, page.hits, page.counts_by_type, page.truncated) == (0, [], {}, False)
    assert client.search.call_count == 1


def test_an_empty_index_is_not_ready_but_no_match_in_a_populated_index_is_fine() -> None:
    with pytest.raises(IndexNotReady):
        _run(_job(_client(_response([]), indexed=0)))
    assert _run(_job(_client(_response([]), indexed=5))).found == 0


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
    monkeypatch.setenv("NADA_STUDIES_LEXICAL_RELATIVE_CUTOFF", "0")  # the fixtures score a weak tail on purpose
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
    os_client = _client(_response(CUT))
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
    assert (body.found, body.truncated, body.result_cap) == (4, False, Settings().studies_result_cap)
    assert body.search_counts_by_type == {"survey": 2, "document": 1, "table": 1}
    assert os_client.search.call_args.kwargs["body"]["size"] == Settings().studies_result_cap


def test_the_result_cap_and_truncation_reach_the_response(monkeypatch: pytest.MonkeyPatch) -> None:
    cut = [_hit(i, 10.0 - i) for i in range(1, 4)]
    monkeypatch.setenv("NADA_STUDIES_RESULT_CAP", "3")
    with _running(monkeypatch, _client(_response(cut, total=50))) as client:
        body = StudySearchResponse.model_validate(_post(client, {"query": "survey", "mode": "lexical"}).json())
    assert (body.found, body.truncated, body.result_cap) == (3, True, 3)
    assert body.invariant_violations() == []


def test_explicit_lexical_and_another_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    cut = [_hit(4, 9.5), _hit(2, 7.25, "document")]
    resorted = _browse_response([(2, "IDNO_2"), (4, "IDNO_4")], found=2, counts={"survey": 1, "document": 1})
    with _running(monkeypatch, _client(_response(cut), resorted)) as client:
        body = StudySearchResponse.model_validate(
            _post(client, {"query": "x", "mode": "lexical", "sort": {"by": "title", "order": "asc"}}).json()
        )
    assert body.applied.mode is EffectiveMode.lexical
    assert [h.sid for h in body.hits] == [2, 4]
    assert body.invariant_violations() == []


def test_types_filter_over_a_query_keeps_the_contract_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _client(_response(CUT))) as client:
        body = StudySearchResponse.model_validate(
            _post(client, {"query": "x", "mode": "lexical", "filters": {"types": ["survey"]}}).json()
        )
    assert body.found == 2
    assert body.search_counts_by_type == {"survey": 2, "document": 1, "table": 1}
    assert body.invariant_violations() == []


def test_a_query_that_matches_nothing_is_an_empty_result_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _client(_response([]), indexed=11)) as client:
        response = _post(client, {"query": "xyzzy qwerty flurbo", "mode": "lexical"})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert (body.found, body.hits, body.search_counts_by_type) == (0, [], {})
    assert body.invariant_violations() == []


def test_an_empty_index_is_index_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _client(_response([]), indexed=0)) as client:
        response = _post(client, {"query": "x", "mode": "lexical"})
    assert response.status_code == 503
    assert ErrorResponse.model_validate(response.json()).error.code.value == "index_not_ready"


def test_debug_lists_every_opensearch_request(monkeypatch: pytest.MonkeyPatch) -> None:
    cut = [_hit(4, 9.5)]
    resorted = _browse_response([(4, "IDNO_4")], found=1, counts={"survey": 1})
    with _running(monkeypatch, _client(_response(cut), resorted)) as client:
        body = _post(
            client, {"query": "x", "mode": "lexical", "sort": {"by": "year", "order": "desc"}, "include_debug": True}
        ).json()
    assert len(body["debug"]["opensearch_requests"]) == 2
