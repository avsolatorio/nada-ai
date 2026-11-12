"""``POST /citations/search`` route: request validation, dispatch and the contract error envelope."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opensearchpy.exceptions import NotFoundError, RequestError
from starlette.testclient import TestClient

from nada_ai.app.main import app, state


@contextmanager
def _running(monkeypatch: pytest.MonkeyPatch, backend: str, **overrides: Any) -> Iterator[TestClient]:
    monkeypatch.setenv("NADA_SEARCH_BACKEND", backend)
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")
    with TestClient(app) as client:
        previous = {name: getattr(state, name) for name in overrides}
        for name, value in overrides.items():
            setattr(state, name, value)
        try:
            yield client
        finally:
            for name, value in previous.items():
                setattr(state, name, value)


def _client(result: dict[str, Any] | None = None, *, side_effect: Exception | None = None) -> MagicMock:
    client = MagicMock()
    client.count = AsyncMock(return_value={"count": 1})  # a populated index: zero hits means nothing matched
    if side_effect is not None:
        client.search = AsyncMock(side_effect=side_effect)
    else:
        client.search = AsyncMock(return_value=result or {"took": 2, "hits": {"total": {"value": 0}, "hits": []}})
    return client


def test_a_query_is_run_and_the_response_matches_the_contract_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = {
        "_score": 7.5,
        "_source": {"citation_id": 12, "uuid": "u-12"},
        "sort": [7.5, 12],
    }
    client = _client({"took": 2, "hits": {"total": {"value": 1}, "hits": [hit]}})
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/citations/search", json={"query": "poverty", "limit": 5, "offset": 10})

    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "opensearch" and body["found"] == 1
    assert body["offset"] == 10 and body["max_offset"] == 10_000 and body["next_cursor"] is None
    # ids only: the caller reads the citation itself from its database
    assert body["hits"] == [{"rank": 11, "citation_id": 12, "uuid": "u-12", "score": 7.5}]
    assert body["applied"] == {
        "query": "poverty",
        "filters": {},
        "sort": "relevance",
        "order": "asc",
        "limit": 5,
        "paging": "offset",
    }


def test_filters_and_sort_reach_the_engine_query(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    payload = {
        "query": "poverty",
        "filters": {"ctypes": ["book", "book", "article"], "year_from": 2000, "year_to": 2010},
        "sort": "year",
        "order": "desc",
    }
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/citations/search", json=payload)

    assert response.status_code == 200
    body = client.search.call_args.kwargs["body"]
    assert {"terms": {"ctype": ["book", "article"]}} in body["query"]["bool"]["filter"]
    assert {"range": {"pub_year": {"gte": 2000, "lte": 2010}}} in body["query"]["bool"]["filter"]
    assert body["sort"][0] == {"pub_year": {"order": "desc", "missing": "_last"}}
    assert response.json()["applied"]["filters"] == {"ctypes": ["book", "article"], "year_from": 2000, "year_to": 2010}


def test_an_unknown_filter_is_rejected_with_the_supported_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_client()) as c:
        response = c.post("/citations/search", json={"query": "x", "filters": {"flag": [1]}})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "unknown_filter"
    assert error["details"] == {"filters": ["flag"], "supported": ["ctypes", "year_from", "year_to"]}


def test_a_backwards_year_range_is_an_invalid_filter_value(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_client()) as c:
        response = c.post("/citations/search", json={"query": "x", "filters": {"year_from": 2010, "year_to": 2000}})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_filter_value"


def test_offset_plus_limit_over_the_max_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_client()) as c:
        response = c.post("/citations/search", json={"query": "x", "offset": 9995, "limit": 15})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "offset_out_of_range" and error["details"] == {"max_offset": 10_000}


def _hits(ids: list[int], total: int | None = None) -> dict[str, Any]:
    hits: dict[str, Any] = {"hits": [{"_score": 1.0, "_source": {"citation_id": i}, "sort": [1.0, i]} for i in ids]}
    if total is not None:
        hits["total"] = {"value": total}
    return {"took": 1, "hits": hits}


def test_a_next_cursor_pages_past_the_offset_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_hits([9999, 10_000], total=14_200))
    with _running(monkeypatch, "opensearch", client=client) as c:
        first = c.post("/citations/search", json={"query": "x", "offset": 9998, "limit": 2}).json()
        assert first["offset"] == 9998 and first["next_cursor"]
        client.search.return_value = _hits([10_001, 10_002, 10_003])
        second = c.post("/citations/search", json={"query": "x", "limit": 2, "cursor": first["next_cursor"]}).json()
    assert second["offset"] == 10_000 and second["found"] == 14_200 and second["applied"]["paging"] == "cursor"
    assert [h["citation_id"] for h in second["hits"]] == [10_001, 10_002]
    assert [h["rank"] for h in second["hits"]] == [10_001, 10_002]
    assert second["next_cursor"] and second["prev_cursor"]


def test_offset_and_cursor_together_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_hits([1, 2], total=40))
    with _running(monkeypatch, "opensearch", client=client) as c:
        token = c.post("/citations/search", json={"query": "x", "limit": 2}).json()["next_cursor"]
        response = c.post("/citations/search", json={"query": "x", "limit": 2, "offset": 2, "cursor": token})
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"


def test_a_cursor_of_another_sort_is_cursor_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_hits([1, 2], total=40))
    with _running(monkeypatch, "opensearch", client=client) as c:
        token = c.post("/citations/search", json={"query": "x", "limit": 2}).json()["next_cursor"]  # relevance
        response = c.post("/citations/search", json={"query": "x", "limit": 2, "sort": "year", "cursor": token})
    assert response.status_code == 422 and response.json()["error"]["code"] == "cursor_invalid"


def test_the_offset_limit_is_a_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NADA_CITATIONS_MAX_OFFSET", "100")
    with _running(monkeypatch, "opensearch", client=_client()) as c:
        response = c.post("/citations/search", json={"query": "x", "offset": 90, "limit": 15})
    assert response.status_code == 422 and response.json()["error"]["details"] == {"max_offset": 100}


def test_an_empty_query_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_client()) as c:
        response = c.post("/citations/search", json={"query": " "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_the_qdrant_engine_answers_unsupported_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "qdrant") as c:
        response = c.post("/citations/search", json={"query": "poverty"})
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "unsupported_capability"
    assert response.json()["error"]["details"]["capability"] == "citations_search"


def test_a_query_opensearch_rejects_is_query_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(side_effect=RequestError(400, "search_phase_execution_exception", {}))
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/citations/search", json={"query": "poverty"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "query_rejected"


def test_a_missing_index_is_index_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(side_effect=NotFoundError(404, "index_not_found_exception", {}))
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/citations/search", json={"query": "poverty"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "index_not_ready"


def test_an_empty_index_is_index_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    client.count = AsyncMock(return_value={"count": 0})
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/citations/search", json={"query": "poverty"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "index_not_ready"


def test_a_search_that_takes_too_long_is_search_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from opensearchpy.exceptions import ConnectionTimeout

    client = _client(side_effect=ConnectionTimeout("TIMEOUT", "timed out", Exception("timed out")))
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/citations/search", json={"query": "poverty"})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "search_timeout"


def test_the_search_is_bounded_by_the_search_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NADA_SEARCH_TIMEOUT_SECONDS", "2.5")
    client = _client()
    with _running(monkeypatch, "opensearch", client=client) as c:
        assert c.post("/citations/search", json={"query": "poverty"}).status_code == 200
    assert client.search.call_args.kwargs["request_timeout"] == 2.5
