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
    search_result: dict[str, Any] | None = None, *, side_effect: Exception | None = None, sids: list[int] | None = None
) -> MagicMock:
    """The study index answers the allowed study ids (``sids``), the variable index ``search_result``."""
    client = MagicMock()
    client.count = AsyncMock(return_value={"count": 1})  # populated indexes: zero hits means nothing matched
    result = search_result or {"took": 3, "hits": {"total": {"value": 0}, "hits": []}}

    async def search(index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        if side_effect is not None:
            raise side_effect
        if index.endswith("-studies"):
            buckets = [{"key": sid, "doc_count": 1} for sid in ([10] if sids is None else sids)]
            return {"aggregations": {"sids": {"buckets": buckets, "sum_other_doc_count": 0}}}
        return result

    client.search = AsyncMock(side_effect=search)
    return client


def _calls(client: MagicMock, studies: bool = False) -> list[Any]:
    return [c for c in client.search.call_args_list if c.kwargs["index"].endswith("-studies") == studies]


def _variable_body(client: MagicMock) -> dict[str, Any]:
    return _calls(client)[-1].kwargs["body"]


def _studies_filter(client: MagicMock) -> list[dict[str, Any]]:
    return _calls(client, studies=True)[-1].kwargs["body"]["query"]["bool"]["filter"]


def test_a_query_is_run_and_the_response_matches_the_contract_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = {"_score": 9.0, "_source": {"uid": 1, "sid": 10}, "sort": [9.0, 1]}
    client = _opensearch_client({"took": 3, "hits": {"total": {"value": 1}, "hits": [hit]}})
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "household"})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "opensearch"
    assert body["found"] == 1
    assert (body["found_is_lower_bound"], body["offset"], body["max_offset"]) == (False, 0, 1000)
    assert body["next_cursor"] is None and body["prev_cursor"] is None
    assert body["applied"]["mode"] == "lexical" and body["applied"]["paging"] == "offset"
    # ids only: the caller reads the variable itself from its database
    assert body["hits"] == [{"rank": 1, "uid": 1, "sid": 10, "score": 9.0}]
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
    assert _variable_body(client)["sort"] == [
        {"name.sort": {"order": "desc", "missing": "_last"}},
        {"uid": {"order": "desc"}},
    ]


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
        response = c.post("/variables/search", json={"query": "x", "filters": {"tags": ["a"]}})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "unknown_filter"
    assert body["error"]["details"]["filters"] == ["tags"]
    assert {"countries", "collections", "repository", "form_ids"} <= set(body["error"]["details"]["supported"])


def test_offset_paging_reaches_exactly_variables_max_offset_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the default 1,000 and 15 a page, page 66 (offset 975) is the last offset page; 67 needs a cursor."""
    with _running(monkeypatch, "opensearch", client=_opensearch_client()) as c:
        last = c.post("/variables/search", json={"query": "x", "offset": 975, "limit": 15})
        beyond = c.post("/variables/search", json={"query": "x", "offset": 990, "limit": 15})
    assert last.status_code == 200
    assert beyond.status_code == 422
    assert beyond.json()["error"]["code"] == "offset_out_of_range"
    assert beyond.json()["error"]["details"] == {"max_offset": 1000}


def test_the_offset_limit_is_a_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NADA_VARIABLES_MAX_OFFSET", "5000")
    with _running(monkeypatch, "opensearch", client=_opensearch_client()) as c:
        response = c.post("/variables/search", json={"query": "x", "offset": 4985, "limit": 15})
    assert response.status_code == 200
    assert response.json()["max_offset"] == 5000


def test_no_query_is_a_browse_sorted_by_study_then_name(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _opensearch_client()
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"filters": {"countries": [4], "year_from": 2000}})
    assert response.status_code == 200
    applied = response.json()["applied"]
    assert (applied["mode"], applied["sort"], applied["query"]) == ("browse", "title", None)
    body = _variable_body(client)
    assert "must" not in body["query"]["bool"]
    assert body["track_total_hits"] is True
    assert {"range": {"filter_facets.years": {"gte": 2000}}} in _studies_filter(client)


def test_a_relevance_sort_without_a_query_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_opensearch_client()) as c:
        response = c.post("/variables/search", json={"sort": "relevance"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_a_year_range_backwards_is_an_invalid_filter_value(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_opensearch_client()) as c:
        response = c.post("/variables/search", json={"filters": {"year_from": 2010, "year_to": 2000}})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_filter_value"


def _page_of(rows: int, total: int) -> dict[str, Any]:
    hits = [{"_score": 10.0 - i, "_source": {"uid": i, "sid": 1}, "sort": [10.0 - i, i]} for i in range(1, rows + 1)]
    return {"took": 3, "hits": {"total": {"value": total, "relation": "eq"}, "hits": hits}}


def test_a_next_cursor_pages_past_the_offset_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _opensearch_client(_page_of(15, 4000))
    with _running(monkeypatch, "opensearch", client=client) as c:
        first = c.post("/variables/search", json={"query": "x", "offset": 975}).json()
        page_two = _opensearch_client(_page_of(16, 0))
        client.search.side_effect = page_two.search.side_effect
        response = c.post("/variables/search", json={"query": "x", "cursor": first["next_cursor"]})
    assert response.status_code == 200
    body = response.json()
    assert (body["offset"], body["found"], body["applied"]["paging"]) == (990, 4000, "cursor")
    assert [h["rank"] for h in body["hits"]][:2] == [991, 992]
    assert body["next_cursor"] and body["prev_cursor"]
    sent = _variable_body(client)
    assert sent["search_after"] == [-5.0, 15] and sent["track_total_hits"] is False


def test_offset_and_cursor_together_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _opensearch_client(_page_of(15, 4000))
    with _running(monkeypatch, "opensearch", client=client) as c:
        cursor = c.post("/variables/search", json={"query": "x"}).json()["next_cursor"]
        response = c.post("/variables/search", json={"query": "x", "cursor": cursor, "offset": 0})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_a_cursor_of_another_sort_is_cursor_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _opensearch_client(_page_of(15, 4000))
    with _running(monkeypatch, "opensearch", client=client) as c:
        cursor = c.post("/variables/search", json={"query": "x"}).json()["next_cursor"]  # [score, uid]
        response = c.post("/variables/search", json={"query": "x", "sort": "title", "cursor": cursor})
        garbage = c.post("/variables/search", json={"query": "x", "cursor": "not-a-cursor"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "cursor_invalid"
    assert garbage.json()["error"]["code"] == "cursor_invalid"


def test_a_search_that_takes_too_long_is_search_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from opensearchpy.exceptions import ConnectionTimeout

    client = _opensearch_client(side_effect=ConnectionTimeout("TIMEOUT", "timed out", Exception("timed out")))
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "household"})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "search_timeout"


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


def test_an_empty_index_is_index_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _opensearch_client()
    client.count = AsyncMock(return_value={"count": 0})
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "household"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "index_not_ready"


def test_sids_filter_is_sent_as_a_terms_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _opensearch_client()
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"query": "household", "filters": {"sids": [10, 20]}})
    assert response.status_code == 200
    assert {"terms": {"sid": [10, 20]}} in _studies_filter(client)


def test_collection_repository_and_access_filters_select_the_studies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Study-level filters go to the study index; the variable query keeps the variables of the studies it gives."""
    client = _opensearch_client(sids=[7, 8])
    filters = {"collections": ["wb"], "repository": "central", "form_ids": [2]}
    with _running(monkeypatch, "opensearch", client=client) as c:
        response = c.post("/variables/search", json={"filters": filters})
    assert response.status_code == 200
    study_filter = _studies_filter(client)
    assert {"terms": {"filter_facets.repositories": ["wb"]}} in study_filter
    assert {"term": {"filter_facets.repositories": "central"}} in study_filter
    assert {"terms": {"filter_facets.formid": [2]}} in study_filter
    assert {"term": {"filter_facets.published": 1}} in study_filter
    assert _variable_body(client)["query"]["bool"]["filter"] == [{"terms": {"sid": [7, 8]}}]
