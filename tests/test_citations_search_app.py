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
    if side_effect is not None:
        client.search = AsyncMock(side_effect=side_effect)
    else:
        client.search = AsyncMock(return_value=result or {"took": 2, "hits": {"total": {"value": 0}, "hits": []}})
    return client


def test_a_query_is_run_and_the_response_matches_the_contract_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = {
        "_score": 7.5,
        "_source": {
            "citation_id": 12,
            "uuid": "u-12",
            "title": "Poverty and health",
            "authors": "Jane Doe",
            "ctype": "book",
            "pub_year": 2019,
            "doi": "10.1/abc",
        },
    }
    client = _client({"took": 2, "hits": {"total": {"value": 1}, "hits": [hit]}})
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/citations/search", json={"query": "poverty", "limit": 5, "offset": 10})

    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "opensearch" and body["found"] == 1 and body["truncated"] is False
    assert body["hits"] == [
        {
            "rank": 11,
            "citation_id": 12,
            "uuid": "u-12",
            "title": "Poverty and health",
            "authors": "Jane Doe",
            "ctype": "book",
            "pub_year": 2019,
            "doi": "10.1/abc",
            "score": 7.5,
        }
    ]
    assert body["applied"] == {
        "query": "poverty",
        "filters": {},
        "sort": "relevance",
        "order": "asc",
        "limit": 5,
        "offset": 10,
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
    assert response.json()["error"]["code"] == "offset_out_of_range"


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
