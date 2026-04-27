"""FastAPI smoke tests (no live OpenSearch cluster required for /demo)."""

from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

from nada_ai.app.main import app, state


def test_health_returns_ok():
    with TestClient(app) as client:
        mock = MagicMock()
        mock.cluster.health = AsyncMock(return_value={"status": "green", "cluster_name": "test"})
        prev = state.client
        state.client = mock
        try:
            r = client.get("/health")
        finally:
            state.client = prev
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_demo_route_returns_html():
    with TestClient(app) as client:
        r = client.get("/demo")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert len(r.text) > 100
