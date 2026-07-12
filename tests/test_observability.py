"""Tests for the request-ID middleware and /admin/metrics endpoint."""

from __future__ import annotations

from starlette.testclient import TestClient

from nada_ai.app.jobs import JobRegistry
from nada_ai.app.main import app, state


def _fresh_state() -> None:
    state.jobs = JobRegistry()


def test_request_id_generated_and_returned(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        _fresh_state()
        r1 = client.get("/health")
        r2 = client.get("/health")
    assert r1.headers.get("x-request-id")
    assert r2.headers.get("x-request-id")
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


def test_request_id_propagated_from_caller(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/health", headers={"X-Request-ID": "caller-supplied-id"})
    assert r.headers["x-request-id"] == "caller-supplied-id"


def test_metrics_endpoint_requires_auth(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "secret")
    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/metrics")
    assert r.status_code == 401


def test_metrics_endpoint_records_requests(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "secret")
    headers = {"X-NADA-Admin-Key": "secret"}
    with TestClient(app) as client:
        _fresh_state()
        client.get("/health")
        client.get("/health")
        r = client.get("/admin/metrics", headers=headers)

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "nada_http_requests_total" in r.text
    assert 'route="/health"' in r.text
    assert "nada_http_request_duration_seconds_bucket" in r.text
