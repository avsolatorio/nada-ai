"""Contract tests for the standard study search (models and sample payloads only; no endpoints yet)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nada_ai.app.studies_schemas import (
    CANONICAL_FILTERS,
    ERROR_HTTP_STATUS,
    MAX_LIMIT,
    MAX_OFFSET,
    Engine,
    ErrorCode,
    ErrorResponse,
    InfoResponse,
    StudyFilters,
    StudySearchRequest,
    StudySearchResponse,
)

FIXTURES = Path(__file__).parent / "fixtures" / "studies_search"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _names(prefix: str) -> list[str]:
    return sorted(p.name for p in FIXTURES.glob(f"{prefix}*.json"))


# ---------------------------------------------------------------------------------------
# Fixtures parse against the models
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _names("info_"))
def test_info_fixtures_parse(name: str) -> None:
    InfoResponse.model_validate(_load(name))


@pytest.mark.parametrize("name", _names("request_"))
def test_request_fixtures_parse(name: str) -> None:
    StudySearchRequest.model_validate(_load(name))


@pytest.mark.parametrize("name", _names("response_"))
def test_response_fixtures_parse_and_hold_invariants(name: str) -> None:
    response = StudySearchResponse.model_validate(_load(name))
    assert response.invariant_violations() == []


@pytest.mark.parametrize("name", _names("error_"))
def test_error_fixtures_parse(name: str) -> None:
    error = ErrorResponse.model_validate(_load(name)).error
    assert error.code in ERROR_HTTP_STATUS


def test_every_error_code_has_a_status() -> None:
    assert set(ERROR_HTTP_STATUS) == set(ErrorCode)


# ---------------------------------------------------------------------------------------
# Info contract
# ---------------------------------------------------------------------------------------


def test_info_opensearch_advertises_the_filter_registry() -> None:
    info = InfoResponse.model_validate(_load("info_opensearch.json"))
    assert info.engine is Engine.opensearch
    assert info.capabilities.studies_search is True
    assert info.filters == list(CANONICAL_FILTERS)
    assert info.id_key == "sid"


def test_info_qdrant_reports_studies_search_unsupported() -> None:
    info = InfoResponse.model_validate(_load("info_qdrant.json"))
    assert info.engine is Engine.qdrant
    assert info.capabilities.studies_search is False
    assert info.filters == []


def test_filter_registry_matches_the_request_model() -> None:
    assert [spec.key for spec in CANONICAL_FILTERS] == list(StudyFilters.model_fields)


# ---------------------------------------------------------------------------------------
# Request validation: nothing is dropped silently
# ---------------------------------------------------------------------------------------


def test_unknown_filter_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StudyFilters.model_validate({"region": [1]})


def test_unknown_request_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StudySearchRequest.model_validate({"query": "x", "page": 2})


@pytest.mark.parametrize(
    "filters",
    [
        {"year_from": 2020, "year_to": 2010},
        {"created_from": 20, "created_to": 10},
        {"year_from": 0},
        {"year_to": 100_000_000},
        {"countries": ["Brazil"]},
        {"countries": [0]},
        {"facets": {"bad name!": [1]}},
        {"facets": {"funding": ["x"]}},
        {"types": [""]},
    ],
)
def test_invalid_filter_values_are_rejected(filters: dict) -> None:
    with pytest.raises(ValidationError):
        StudyFilters.model_validate(filters)


def test_year_range_bounds_prevent_huge_ranges() -> None:
    # The old semantic driver expanded year ranges into lists; the contract accepts bounds only.
    StudyFilters.model_validate({"year_from": 1, "year_to": 9999})
    with pytest.raises(ValidationError):
        StudyFilters.model_validate({"year_from": 1, "year_to": 5_000_000})


def test_empty_values_mean_no_constraint_and_lists_are_deduplicated() -> None:
    filters = StudyFilters.model_validate(
        {"types": [], "countries": [16, 16, 191], "tags": None, "facets": {"funding": [], "author": [5, 5]}}
    )
    assert filters.active() == {"countries": [16, 191], "facets": {"author": [5]}}


def test_only_active_filters_are_echoed() -> None:
    assert StudyFilters().active() == {}


def test_query_is_trimmed_and_blank_means_browse() -> None:
    assert StudySearchRequest.model_validate({"query": "  poverty  "}).query == "poverty"
    assert StudySearchRequest.model_validate({"query": "   "}).query is None


@pytest.mark.parametrize(
    "body",
    [
        {"limit": 0},
        {"limit": MAX_LIMIT + 1},
        {"offset": -1},
        {"offset": MAX_OFFSET + 1},
        {"query": "x" * 501},
        {"mode": "fuzzy"},
        {"sort": {"by": "rank"}},
    ],
)
def test_out_of_range_request_values_are_rejected(body: dict) -> None:
    with pytest.raises(ValidationError):
        StudySearchRequest.model_validate(body)


def test_request_defaults() -> None:
    request = StudySearchRequest.model_validate({})
    assert request.query is None
    assert request.mode.value == "auto"
    assert (request.limit, request.offset) == (15, 0)
    assert request.include_facets is False


# ---------------------------------------------------------------------------------------
# Response invariants catch broken responses
# ---------------------------------------------------------------------------------------


def _hybrid() -> dict:
    return _load("response_hybrid.json")


def test_invariants_flag_non_contiguous_ranks() -> None:
    data = _hybrid()
    data["hits"][1]["rank"] = 5
    assert StudySearchResponse.model_validate(data).invariant_violations()


def test_invariants_flag_counts_that_do_not_add_up() -> None:
    data = _hybrid()
    data["search_counts_by_type"]["survey"] = 9
    assert StudySearchResponse.model_validate(data).invariant_violations()


def test_invariants_flag_found_above_the_cap() -> None:
    data = _hybrid()
    data["found"] = 101
    data["result_cap"] = 100
    assert StudySearchResponse.model_validate(data).invariant_violations()


def test_invariants_flag_truncated_without_a_cap() -> None:
    data = _load("response_browse.json")
    data["truncated"] = True
    assert StudySearchResponse.model_validate(data).invariant_violations()


def test_invariants_flag_relevance_hits_without_matched_by() -> None:
    data = _hybrid()
    data["hits"][0]["matched_by"] = []
    assert StudySearchResponse.model_validate(data).invariant_violations()


def test_invariants_flag_duplicate_sids() -> None:
    data = _hybrid()
    data["hits"][1]["sid"] = data["hits"][0]["sid"]
    assert StudySearchResponse.model_validate(data).invariant_violations()


def test_types_filter_found_equals_the_selected_types_count() -> None:
    response = StudySearchResponse.model_validate(_load("response_types_filter.json"))
    assert response.found == response.search_counts_by_type["document"]
    # tab counts ignore the types filter, so they still add up to more than found
    assert sum(response.search_counts_by_type.values()) > response.found


def test_response_rejects_unknown_fields() -> None:
    data = _hybrid()
    data["total"] = 4
    with pytest.raises(ValidationError):
        StudySearchResponse.model_validate(data)
