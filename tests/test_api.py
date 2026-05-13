"""FastAPI smoke tests (no live OpenSearch cluster required for /demo)."""

from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

from nada_ai.app.main import app, state
from nada_ai.search.factory import create_search_backend


def test_health_returns_ok():
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


def test_demo_route_returns_html():
    with TestClient(app) as client:
        r = client.get("/demo")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert len(r.text) > 100
