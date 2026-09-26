"""``POST /variables/search`` route: request validation, dispatch and the contract error envelope.

Mirrors the harness pattern in ``test_info_endpoint.py`` (a real ``TestClient`` over ``app``, with ``state.client``
swapped for a mock) rather than the executor-level mocks in ``test_variables_search_backend.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import RequestError
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


def _opensearch_client(
    search_result: dict[str, Any] | None = None, *, side_effect: Exception | None = None
) -> MagicMock:
    client = MagicMock()
    if side_effect is not None:
        client.search = AsyncMock(side_effect=side_effect)
    else:
        client.search = AsyncMock(
            return_value=search_result
            or {
                "took": 3,
                "hits": {
                    "total": {"value": 0},
                    "hits": [],
                },
            }
        )
    return client


def test_a_query_is_run_and_the_response_matches_the_contract_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = {
        "_score": 9.0,
        "_source": {
            "uid": 1,
            "sid": 10,
            "idno": "IDNO-1",
            "fid": "F1",
            "vid": "V1",
            "name": "hhid",
            "label": "Household id",
            "question": None,
            "title": "Study one",
            "nation": "Kenya",
            "dataset_type": "survey",
            "year_start": 2020,
            "year_end": 2020,
        },
    }
    client = _opensearch_client({"took": 3, "hits": {"total": {"value": 1}, "hits": [hit]}})
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "household"})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "opensearch"
    assert body["found"] == 1
    assert body["hits"] == [
        {
            "rank": 1,
            "uid": 1,
            "sid": 10,
            "idno": "IDNO-1",
            "fid": "F1",
            "vid": "V1",
            "name": "hhid",
            "label": "Household id",
            "question": None,
            "title": "Study one",
            "nation": "Kenya",
            "dataset_type": "survey",
            "year_start": 2020,
            "year_end": 2020,
            "score": 9.0,
        }
    ]
    assert body["applied"]["query"] == "household"
    assert body["applied"]["sort"] == "relevance"


def test_a_name_sort_with_an_order_is_sent_to_the_sortable_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """What NADA sends for a variable list sorted by name, descending."""
    client = _opensearch_client()
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "x", "sort": "name", "order": "desc"})
    assert response.status_code == 200
    assert response.json()["applied"]["sort"] == "name"
    assert response.json()["applied"]["order"] == "desc"
    assert client.search.call_args.kwargs["body"]["sort"][0] == {"name.sort": {"order": "desc", "missing": "_last"}}


def test_a_query_opensearch_rejects_is_query_rejected_not_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """How the name sort on a text field surfaced: a 400 from OpenSearch, which used to be answered 503."""
    client = _opensearch_client(side_effect=RequestError(400, "search_phase_execution_exception", {}))
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "x", "sort": "name"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "query_rejected"


def test_an_unreachable_engine_is_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _opensearch_client(side_effect=OpenSearchConnectionError("N/A", "refused", Exception("refused")))
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "x"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backend_unavailable"


def test_an_unknown_filter_is_rejected_with_the_contract_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_opensearch_client()) as c:
        response = c.post("/variables/search", json={"query": "x", "filters": {"countries": [1]}})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "unknown_filter"
    assert body["error"]["details"]["filters"] == ["countries"]


def test_offset_plus_limit_over_the_max_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_opensearch_client()) as c:
        response = c.post("/variables/search", json={"query": "x", "offset": 9995, "limit": 15})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "offset_out_of_range"


def test_an_empty_query_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_opensearch_client()) as c:
        response = c.post("/variables/search", json={"query": ""})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_the_qdrant_engine_answers_unsupported_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "qdrant") as c:
        response = c.post("/variables/search", json={"query": "household"})
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "unsupported_capability"


def test_a_missing_index_is_index_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    from opensearchpy.exceptions import NotFoundError

    client = _opensearch_client(side_effect=NotFoundError(404, "index_not_found_exception", {}))
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "household"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "index_not_ready"


def test_sids_filter_is_sent_as_a_terms_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _opensearch_client()
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "household", "filters": {"sids": [10, 20]}})
    assert response.status_code == 200
    body = client.search.call_args.kwargs["body"]
    assert {"terms": {"sid": [10, 20]}} in body["query"]["bool"]["filter"]
