"""POST /studies/search, browse mode (step 5 of the OpenSearch plan): query building, mode/sort rules, the endpoint."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import NotFoundError
from starlette.testclient import TestClient

from nada_ai.app import info as info_module
from nada_ai.app.main import app, state
from nada_ai.app.studies_errors import StudiesApiError
from nada_ai.app.studies_schemas import (
    EffectiveMode,
    Engine,
    ErrorResponse,
    SearchMode,
    SortField,
    SortOrder,
    StudyFilters,
    StudySearchRequest,
    StudySearchResponse,
    StudySort,
)
from nada_ai.app.studies_search import resolve_mode, resolve_sort
from nada_ai.search.backend.opensearch.studies_search import (
    browse_body,
    filter_clauses,
    sort_clause,
    types_post_filter,
)
from nada_ai.settings import Settings

PUBLISHED = {"term": {"filter_facets.published": 1}}


# ---------------------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------------------


def test_no_filters_means_published_studies_only() -> None:
    assert filter_clauses(StudyFilters()) == [PUBLISHED]


@pytest.mark.parametrize(
    ("filters", "clause"),
    [
        ({"countries": [16, 191]}, {"terms": {"filter_facets.countries": [16, 191]}}),
        ({"year_from": 2010}, {"range": {"filter_facets.years": {"gte": 2010}}}),
        ({"year_to": 2020}, {"range": {"filter_facets.years": {"lte": 2020}}}),
        ({"year_from": 2010, "year_to": 2020}, {"range": {"filter_facets.years": {"gte": 2010, "lte": 2020}}}),
        ({"repository": "demo"}, {"term": {"filter_facets.repositories": "demo"}}),
        ({"collections": ["a", "b"]}, {"terms": {"filter_facets.repositories": ["a", "b"]}}),
        ({"form_ids": [1, 2]}, {"terms": {"filter_facets.formid": [1, 2]}}),
        ({"data_class_ids": [3]}, {"terms": {"filter_facets.data_class_id": [3]}}),
        ({"tags": ["health"]}, {"terms": {"filter_facets.tags": ["health"]}}),
        ({"facets": {"author": [7, 8]}}, {"terms": {"filter_facets.fq_author": [7, 8]}}),
        ({"sids": [2, 3]}, {"terms": {"sid": [2, 3]}}),
        ({"created_from": 10, "created_to": 20}, {"range": {"created": {"gte": 10, "lte": 20}}}),
        ({"created_to": 20}, {"range": {"created": {"lte": 20}}}),
    ],
)
def test_each_filter_becomes_one_clause(filters: dict[str, Any], clause: dict[str, Any]) -> None:
    assert filter_clauses(StudyFilters.model_validate(filters)) == [PUBLISHED, clause]


def test_repository_and_collections_are_separate_constraints() -> None:
    clauses = filter_clauses(StudyFilters(repository="demo", collections=["a"]))
    assert {"term": {"filter_facets.repositories": "demo"}} in clauses
    assert {"terms": {"filter_facets.repositories": ["a"]}} in clauses


def test_each_user_facet_is_its_own_and_clause() -> None:
    clauses = filter_clauses(StudyFilters(facets={"author": [1], "funding": [2]}))
    assert {"terms": {"filter_facets.fq_author": [1]}} in clauses
    assert {"terms": {"filter_facets.fq_funding": [2]}} in clauses


def test_types_is_a_post_filter_so_tab_counts_ignore_it() -> None:
    filters = StudyFilters(types=["document", "survey"], countries=[16])
    assert all("dataset_type" not in str(clause) for clause in filter_clauses(filters))
    assert types_post_filter(filters) == {"terms": {"filter_facets.dataset_type": ["document", "survey"]}}
    body = browse_body(filters, SortField.title, SortOrder.asc, 15, 0)
    assert body["post_filter"] == types_post_filter(filters)
    assert "post_filter" not in browse_body(StudyFilters(), SortField.title, SortOrder.asc, 15, 0)


def test_browse_body_pages_counts_exactly_and_aggregates_types() -> None:
    body = browse_body(StudyFilters(), SortField.year, SortOrder.desc, 20, 40)
    assert (body["from"], body["size"], body["track_total_hits"]) == (40, 20, True)
    assert body["_source"] == ["sid", "idno"]
    assert body["aggs"]["by_type"]["terms"]["field"] == "filter_facets.dataset_type"


@pytest.mark.parametrize(
    ("by", "field"),
    [
        (SortField.title, "title_sort"),
        (SortField.nation, "nation_sort"),
        (SortField.year, "year_start"),
        (SortField.popularity, "total_views"),
        (SortField.created, "created"),
        (SortField.changed, "changed"),
    ],
)
def test_sort_fields_map_to_index_fields(by: SortField, field: str) -> None:
    clause = sort_clause(by, SortOrder.desc)
    assert next(iter(clause[0])) == field
    assert clause[0][field] == {"order": "desc", "missing": "_last"}


def test_sort_is_deterministic_and_never_repeats_a_field() -> None:
    def fields(by: SortField) -> list[str]:
        return [next(iter(item)) for item in sort_clause(by, SortOrder.asc)]

    assert fields(SortField.popularity) == ["total_views", "year_start", "title_sort", "sid"]
    assert fields(SortField.title) == ["title_sort", "year_start", "sid"]
    assert fields(SortField.year) == ["year_start", "title_sort", "sid"]
    assert fields(SortField.title)[-1] == "sid"


# ---------------------------------------------------------------------------------------
# Mode and sort resolution
# ---------------------------------------------------------------------------------------

BROWSE = frozenset({"browse"})


def _request(**fields: Any) -> StudySearchRequest:
    return StudySearchRequest.model_validate(fields)


def test_no_query_is_browse_whatever_the_requested_mode() -> None:
    for mode in ("auto", "lexical", "semantic", "hybrid"):
        assert resolve_mode(_request(mode=mode), Engine.opensearch, BROWSE) is EffectiveMode.browse


def test_browse_needs_the_browse_capability() -> None:
    with pytest.raises(StudiesApiError) as raised:
        resolve_mode(_request(), Engine.opensearch, frozenset())
    assert raised.value.details == {"capability": "browse", "engine": "opensearch"}


def test_a_query_is_refused_while_no_query_mode_exists() -> None:
    with pytest.raises(StudiesApiError) as raised:
        resolve_mode(_request(query="poverty"), Engine.opensearch, BROWSE)
    assert raised.value.code.value == "unsupported_capability"
    assert raised.value.details["capability"] == "lexical"

    with pytest.raises(StudiesApiError) as explicit:
        resolve_mode(_request(query="poverty", mode="semantic"), Engine.opensearch, BROWSE)
    assert explicit.value.details["capability"] == "semantic"


def test_auto_uses_the_best_mode_the_engine_serves() -> None:
    both = frozenset({"browse", "lexical", "hybrid"})
    assert resolve_mode(_request(query="x"), Engine.opensearch, both) is EffectiveMode.hybrid
    assert resolve_mode(_request(query="x"), Engine.opensearch, frozenset({"lexical"})) is EffectiveMode.lexical
    assert resolve_mode(_request(query="x", mode="lexical"), Engine.opensearch, both) is EffectiveMode.lexical
    assert _request(query="x").mode is SearchMode.auto


def test_browse_sort_defaults_to_title_ascending() -> None:
    sort, warnings = resolve_sort(_request(), EffectiveMode.browse)
    assert (sort.by, sort.order, warnings) == (SortField.title, SortOrder.asc, [])


def test_browse_sort_is_kept_when_it_is_a_real_field() -> None:
    sort, warnings = resolve_sort(_request(sort={"by": "year", "order": "desc"}), EffectiveMode.browse)
    assert (sort.by, sort.order, warnings) == (SortField.year, SortOrder.desc, [])


def test_relevance_without_a_query_is_adjusted_with_a_warning() -> None:
    sort, warnings = resolve_sort(_request(sort={"by": "relevance", "order": "desc"}), EffectiveMode.browse)
    assert (sort.by, sort.order) == (SortField.title, SortOrder.asc)
    assert [w.code.value for w in warnings] == ["sort_adjusted"]


def test_a_query_defaults_to_relevance_descending() -> None:
    sort, _ = resolve_sort(_request(query="x"), EffectiveMode.hybrid)
    assert (sort.by, sort.order) == (SortField.relevance, SortOrder.desc)
    assert StudySort(by=SortField.title).order is SortOrder.asc


# ---------------------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------------------


@contextmanager
def _running(monkeypatch: pytest.MonkeyPatch, client_mock: Any, backend: str = "opensearch") -> Iterator[TestClient]:
    monkeypatch.setenv("NADA_SEARCH_BACKEND", backend)
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        previous = state.client
        state.client = client_mock
        try:
            yield client
        finally:
            state.client = previous


def _os(
    *, found: int = 3, rows: list[tuple[int, str]] | None = None, counts: dict[str, int] | None = None
) -> MagicMock:
    rows = rows if rows is not None else [(2, "EGY_2014_DHS_v01_M"), (4, "PC11_A02-28-v22"), (21, "document-unique-id")]
    counts = counts if counts is not None else {"survey": 2, "document": 1}
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "took": 4,
            "hits": {
                "total": {"value": found, "relation": "eq"},
                "hits": [{"_source": {"sid": sid, "idno": idno}} for sid, idno in rows],
            },
            "aggregations": {"by_type": {"buckets": [{"key": k, "doc_count": v} for k, v in counts.items()]}},
        }
    )
    client.count = AsyncMock(return_value={"count": 3})
    return client


def _post(client: TestClient, body: dict[str, Any] | None = None, **kwargs: Any):
    return client.post("/studies/search", json=body if body is not None else {}, **kwargs)


def _error(response: Any) -> tuple[str, dict[str, Any]]:
    error = ErrorResponse.model_validate(response.json()).error
    return error.code.value, error.model_dump(mode="json")


def test_browse_returns_ids_counts_and_the_effective_search(monkeypatch: pytest.MonkeyPatch) -> None:
    os_client = _os(found=3)
    with _running(monkeypatch, os_client) as client:
        response = _post(client, {"filters": {"countries": [16]}, "limit": 3, "offset": 0})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert body.invariant_violations() == []
    assert (body.engine.value, body.found, body.truncated, body.result_cap) == ("opensearch", 3, False, None)
    assert [(h.sid, h.idno, h.rank) for h in body.hits] == [
        (2, "EGY_2014_DHS_v01_M", 1),
        (4, "PC11_A02-28-v22", 2),
        (21, "document-unique-id", 3),
    ]
    assert all(h.score is None and h.matched_by == [] for h in body.hits)
    assert body.search_counts_by_type == {"survey": 2, "document": 1}
    assert body.applied.mode is EffectiveMode.browse
    assert body.applied.filters == {"countries": [16]}
    assert (body.applied.sort.by, body.applied.sort.order) == (SortField.title, SortOrder.asc)
    assert body.warnings == []
    assert set(body.timing_ms or {}) == {"total", "engine"}
    assert body.debug is None

    request = os_client.search.call_args.kwargs
    assert request["index"] == Settings().studies_index
    assert request["body"]["from"] == 0 and request["body"]["size"] == 3


def test_ranks_continue_from_the_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _os(found=30, rows=[(5, "E"), (6, "F")], counts={"survey": 30})) as client:
        body = StudySearchResponse.model_validate(_post(client, {"limit": 2, "offset": 10}).json())
    assert [h.rank for h in body.hits] == [11, 12]
    assert body.invariant_violations() == []


def test_types_filter_narrows_found_but_not_the_tab_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    os_client = _os(found=1, rows=[(21, "document-unique-id")], counts={"survey": 2, "document": 1})
    with _running(monkeypatch, os_client) as client:
        body = StudySearchResponse.model_validate(_post(client, {"filters": {"types": ["document"]}}).json())
    assert body.found == 1
    assert body.search_counts_by_type == {"survey": 2, "document": 1}
    assert body.invariant_violations() == []
    assert os_client.search.call_args.kwargs["body"]["post_filter"] == {
        "terms": {"filter_facets.dataset_type": ["document"]}
    }


def test_relevance_sort_without_a_query_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _os()) as client:
        body = StudySearchResponse.model_validate(_post(client, {"sort": {"by": "relevance", "order": "desc"}}).json())
    assert [w.code.value for w in body.warnings] == ["sort_adjusted"]
    assert (body.applied.sort.by, body.applied.sort.order) == (SortField.title, SortOrder.asc)


def test_semantic_and_hybrid_need_the_local_embedding_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NADA_EMBEDDING_BACKEND", "opensearch_ml")
    monkeypatch.setenv("NADA_OPENSEARCH_ML_MODEL_ID", "model")
    monkeypatch.setenv("NADA_OPENSEARCH_ML_EMBEDDING_DIMENSION", "384")
    with _running(monkeypatch, _os()) as client:
        for mode in ("semantic", "hybrid"):
            response = _post(client, {"query": "poverty", "mode": mode})
            assert response.status_code == 501
            code, error = _error(response)
            assert code == "unsupported_capability"
            assert error["details"] == {"capability": mode, "engine": "opensearch"}


def test_facets_are_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _os()) as client:
        response = _post(client, {"include_facets": True})
    assert response.status_code == 501
    assert _error(response)[1]["details"]["capability"] == "facets"


def test_debug_output_needs_the_admin_role(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _os()) as client:
        # unconfigured dev instances treat every caller as admin, so debug is available...
        requests = _post(client, {"include_debug": True}).json()["debug"]["opensearch_requests"]
        assert requests[0]["track_total_hits"] is True


def _guard_dependency() -> Any:
    """The callable behind the route's ``principal`` dependency (the access guard)."""
    route = next(r for r in app.routes if getattr(r, "path", "") == "/studies/search")
    return next(d.call for d in route.dependant.dependencies if d.name == "principal")


def test_debug_is_refused_for_a_non_admin_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from nada_ai.app.auth import Principal
    from nada_ai.app.keys_store import Role

    async def reader() -> Principal:
        return Principal(id="k", name="reader", role=Role.read, source="key")

    with _running(monkeypatch, _os()) as client:
        app.dependency_overrides[_guard_dependency()] = reader
        try:
            refused = _post(client, {"include_debug": True})
            plain = _post(client, {})
        finally:
            app.dependency_overrides.clear()
    assert refused.status_code == 403
    assert _error(refused)[0] == "forbidden"
    assert plain.status_code == 200


# ---------------------------------------------------------------------------------------
# Validation errors map to the contract codes
# ---------------------------------------------------------------------------------------


def test_unknown_filter_keys_are_named_with_the_supported_list(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _os()) as client:
        response = _post(client, {"filters": {"region": [1], "author": ["x"], "countries": [1]}})
    assert response.status_code == 422
    code, error = _error(response)
    assert code == "unknown_filter"
    assert error["details"]["filters"] == ["author", "region"]
    assert "countries" in error["details"]["supported"]


@pytest.mark.parametrize(
    ("filters", "filter_name"),
    [
        ({"year_from": 2020, "year_to": 2010}, None),
        ({"countries": ["Brazil"]}, "countries"),
        ({"year_from": 0}, "year_from"),
        ({"facets": {"bad name!": [1]}}, "facets"),
    ],
)
def test_malformed_filter_values_are_invalid_filter_value(
    monkeypatch: pytest.MonkeyPatch, filters: dict[str, Any], filter_name: str | None
) -> None:
    with _running(monkeypatch, _os()) as client:
        response = _post(client, {"filters": filters})
    assert response.status_code == 422
    code, error = _error(response)
    assert code == "invalid_filter_value"
    assert (error["details"] or {}).get("filter") == filter_name


@pytest.mark.parametrize("body", [{"limit": 0}, {"limit": 101}, {"mode": "fuzzy"}, {"query": "x" * 501}, {"page": 2}])
def test_other_bad_requests_are_invalid_request(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> None:
    with _running(monkeypatch, _os()) as client:
        response = _post(client, body)
    assert response.status_code == 422
    assert _error(response)[0] == "invalid_request"


def test_malformed_json_is_invalid_request(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _os()) as client:
        response = client.post("/studies/search", content=b"{nope", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert _error(response)[0] == "invalid_request"


def test_offset_beyond_the_window_is_offset_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _os()) as client:
        too_far = _post(client, {"offset": 10_001})
        window = _post(client, {"offset": 9_990, "limit": 20})
        edge = _post(client, {"offset": 9_985, "limit": 15})
    assert too_far.status_code == window.status_code == 422
    assert _error(too_far)[0] == _error(window)[0] == "offset_out_of_range"
    assert edge.status_code == 200


def test_other_routes_keep_the_framework_validation_body(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _os()) as client:
        response = client.post("/search", json={"mode": "fuzzy"})
    assert response.status_code == 422
    assert "detail" in response.json() and "error" not in response.json()


# ---------------------------------------------------------------------------------------
# Index and engine states
# ---------------------------------------------------------------------------------------


def test_a_missing_index_is_index_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    os_client = _os()
    os_client.search = AsyncMock(side_effect=NotFoundError(404, "index_not_found_exception", {}))
    with _running(monkeypatch, os_client) as client:
        response = _post(client)
    assert response.status_code == 503
    assert _error(response)[0] == "index_not_ready"


def test_an_empty_index_is_index_not_ready_but_an_empty_match_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = _os(found=0, rows=[], counts={})
    empty.count = AsyncMock(return_value={"count": 0})
    with _running(monkeypatch, empty) as client:
        nothing_indexed = _post(client)
    assert nothing_indexed.status_code == 503
    assert _error(nothing_indexed)[0] == "index_not_ready"

    no_match = _os(found=0, rows=[], counts={})
    no_match.count = AsyncMock(return_value={"count": 11})
    with _running(monkeypatch, no_match) as client:
        response = _post(client, {"filters": {"countries": [999]}})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert (body.found, body.hits, body.search_counts_by_type) == (0, [], {})
    assert body.invariant_violations() == []


def test_unreachable_opensearch_is_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    os_client = _os()
    os_client.search = AsyncMock(side_effect=OpenSearchConnectionError("N/A", "refused", Exception("refused")))
    with _running(monkeypatch, os_client) as client:
        response = _post(client)
    assert response.status_code == 503
    assert _error(response)[0] == "backend_unavailable"


def test_qdrant_does_not_support_study_search(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, None, backend="qdrant") as client:
        response = _post(client)
    assert response.status_code == 501
    code, error = _error(response)
    assert code == "unsupported_capability"
    assert error["details"] == {"capability": "studies_search", "engine": "qdrant"}


def test_no_implemented_modes_means_study_search_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(info_module.IMPLEMENTED_STUDY_MODES, Engine.opensearch, frozenset())
    with _running(monkeypatch, _os()) as client:
        assert _post(client).status_code == 501


def test_access_errors_come_before_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _os()) as client:
        monkeypatch.setenv("NADA_ADMIN_API_KEY", "secret")
        response = _post(client, {"limit": 0})
        ok = _post(client, {}, headers={"X-NADA-Admin-Key": "secret"})
    assert response.status_code == 401
    assert _error(response)[0] == "unauthorized"
    assert ok.status_code == 200
