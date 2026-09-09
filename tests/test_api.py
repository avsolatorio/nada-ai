"""FastAPI smoke tests (no live OpenSearch cluster required for /demo)."""

from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

from nada_ai.app.main import app, state
from nada_ai.search.factory import create_search_backend
from nada_ai.search.ports import SearchOutcome


def test_health_returns_ok(monkeypatch):
    # This test mocks an OpenSearch client/search backend post-lifespan; pin
    # the backend the lifespan itself constructs to match, rather than
    # whatever NADA_SEARCH_BACKEND currently defaults to.
    monkeypatch.setenv("NADA_SEARCH_BACKEND", "opensearch")
    with TestClient(app) as client:
        mock = MagicMock()
        mock.cluster.health = AsyncMock(return_value={"status": "green", "cluster_name": "test"})
        prev_client, prev_search = state.client, state.search
        state.client = mock
        state.search = create_search_backend(state.settings, mock)
        try:
            r = client.get("/health")
        finally:
            state.client = prev_client
            state.search = prev_search
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_search_falls_back_to_real_search_when_idno_fast_path_misses():
    """Regression test: looks_like_catalog_idno() is a broad heuristic (any
    compact, space-free token — including plain single-word queries like
    "disability" that aren't idnos at all). The fast path does an exact idno
    match with no fallback of its own, so before this fix a query like
    "disability" that heuristically LOOKS idno-shaped but ISN'T a real idno
    silently returned zero results forever, never running real search.
    Found via live end-to-end testing against a real catalog — no unit test
    exercised the real heuristic against a real single-word query before."""
    with TestClient(app) as client:
        prev_search = state.search
        mock_search = AsyncMock()
        # First call = the fast-path idno lookup (misses); second = the real
        # keyword-search fallback this fix adds.
        mock_search.search = AsyncMock(
            side_effect=[
                SearchOutcome(total=0, hits=[]),
                SearchOutcome(total=1, hits=[{"_id": "x", "_score": 1.0, "_source": {"metadata": {"idno": "WB_LSMS_001"}}}]),
            ]
        )
        state.search = mock_search
        try:
            r = client.post("/search", json={"query": "disability", "mode": "keyword", "size": 5})
        finally:
            state.search = prev_search

    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["hits"][0]["_source"]["metadata"]["idno"] == "WB_LSMS_001"
    assert mock_search.search.await_count == 2


def test_search_does_not_retry_when_idno_fast_path_hits():
    """A real idno match (fast path total > 0) must NOT trigger the fallback
    retry — that would defeat the point of the fast path."""
    with TestClient(app) as client:
        prev_search = state.search
        mock_search = AsyncMock()
        mock_search.search = AsyncMock(
            return_value=SearchOutcome(
                total=1, hits=[{"_id": "x", "_score": 1.0, "_source": {"metadata": {"idno": "WB_LSMS_001"}}}]
            )
        )
        state.search = mock_search
        try:
            r = client.post("/search", json={"query": "WB_LSMS_001", "mode": "keyword", "size": 5})
        finally:
            state.search = prev_search

    assert r.status_code == 200
    assert r.json()["total"] == 1
    assert mock_search.search.await_count == 1


def test_demo_route_returns_html():
    with TestClient(app) as client:
        r = client.get("/demo")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert len(r.text) > 100
    assert 'role="tablist"' in r.text
    assert "results-region" in r.text
    assert "TAB_DEFS" in r.text
    assert "doc-carousel" in r.text
    assert "doc-carousel-lightbox" in r.text
    assert "splitDocumentVariants" in r.text
    assert "dynamic-filters-list" in r.text
    assert "facet-keys-config" in r.text
