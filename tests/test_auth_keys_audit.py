"""Tests for the per-caller API key store, role-based auth, and audit trail.

Uses ``NADA_API_KEYS_PATH`` / ``NADA_AUDIT_LOG_PATH`` pointed at tmp files so
these tests never touch the real ``config/`` directory.
"""

from __future__ import annotations

import asyncio

from starlette.testclient import TestClient

from nada_ai.app import admin as admin_module
from nada_ai.app.jobs import JobRegistry
from nada_ai.app.main import app, state
from nada_ai.app.rate_limit import RateLimiter


def _fresh_state() -> None:
    state.jobs = JobRegistry()


def _isolate_stores(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_API_KEYS_PATH", str(tmp_path / "api_keys.json"))
    monkeypatch.setenv("NADA_AUDIT_LOG_PATH", str(tmp_path / "audit.log"))


def test_unconfigured_server_is_fail_open(monkeypatch, tmp_path):
    """No env key and no stored keys => admin routes remain open (dev default)."""
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    # /admin/index is OpenSearch-only (_require_opensearch) — pin the backend
    # explicitly rather than relying on whatever NADA_SEARCH_BACKEND defaults to.
    monkeypatch.setenv("NADA_SEARCH_BACKEND", "opensearch")
    _isolate_stores(monkeypatch, tmp_path)
    monkeypatch.setattr(admin_module, "create_index_op", lambda settings, recreate=False: {"index": "x"})

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/index", json={"recreate": False})
    assert r.status_code == 202


def test_legacy_env_key_creates_and_scopes_new_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "legacy-secret")
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        headers = {"X-NADA-Admin-Key": "legacy-secret"}

        r = client.post("/admin/keys", json={"name": "ci-bot", "role": "write"}, headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "write"
        assert "key" in body and body["key"].startswith("nada_")
        raw_key = body["key"]
        key_id = body["id"]

        # listing never exposes the raw key
        r = client.get("/admin/keys", headers=headers)
        assert r.status_code == 200
        listed = r.json()["keys"]
        assert len(listed) == 1
        assert "key" not in listed[0]
        assert listed[0]["key_prefix"].startswith("nada_")
        assert raw_key.startswith("nada_")
        assert key_id == listed[0]["id"]


def test_write_role_key_cannot_perform_admin_actions(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "legacy-secret")
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        admin_headers = {"X-NADA-Admin-Key": "legacy-secret"}
        r = client.post("/admin/keys", json={"name": "reader", "role": "read"}, headers=admin_headers)
        raw_key = r.json()["key"]

        # read-role key can read jobs...
        r = client.get("/jobs", headers={"X-NADA-Admin-Key": raw_key})
        assert r.status_code == 200

        # ...but cannot create keys (requires admin role)
        r = client.post(
            "/admin/keys", json={"name": "x", "role": "read"}, headers={"X-NADA-Admin-Key": raw_key}
        )
        assert r.status_code == 403

        # ...and cannot mutate facets (requires write role)
        r = client.post("/admin/facets", json={"keys": ["x"]}, headers={"X-NADA-Admin-Key": raw_key})
        assert r.status_code == 403


def test_invalid_key_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "legacy-secret")
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/jobs", headers={"X-NADA-Admin-Key": "totally-wrong"})
        assert r.status_code == 401
        r = client.get("/jobs")  # missing header entirely
        assert r.status_code == 401


def test_revoked_key_stops_working(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "legacy-secret")
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        admin_headers = {"X-NADA-Admin-Key": "legacy-secret"}
        r = client.post("/admin/keys", json={"name": "temp", "role": "read"}, headers=admin_headers)
        raw_key, key_id = r.json()["key"], r.json()["id"]

        r = client.get("/jobs", headers={"X-NADA-Admin-Key": raw_key})
        assert r.status_code == 200

        r = client.delete(f"/admin/keys/{key_id}", headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["revoked_at"] is not None

        r = client.get("/jobs", headers={"X-NADA-Admin-Key": raw_key})
        assert r.status_code == 401


def test_revoke_unknown_key_404(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "legacy-secret")
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        r = client.delete("/admin/keys/does-not-exist", headers={"X-NADA-Admin-Key": "legacy-secret"})
        assert r.status_code == 404


def test_audit_trail_records_mutations(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "legacy-secret")
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        admin_headers = {"X-NADA-Admin-Key": "legacy-secret"}
        r = client.post("/admin/keys", json={"name": "audited", "role": "write"}, headers=admin_headers)
        assert r.status_code == 200

        r = client.get("/admin/audit", headers=admin_headers)
        assert r.status_code == 200
        entries = r.json()["entries"]
        assert any(e["action"] == "key.create" and e["detail"] == "name=audited role=write" for e in entries)
        assert all(e["principal_name"] == "legacy env admin key" for e in entries)


def test_audit_requires_admin_role(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "legacy-secret")
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        admin_headers = {"X-NADA-Admin-Key": "legacy-secret"}
        r = client.post("/admin/keys", json={"name": "reader", "role": "read"}, headers=admin_headers)
        raw_key = r.json()["key"]

        r = client.get("/admin/audit", headers={"X-NADA-Admin-Key": raw_key})
        assert r.status_code == 403


def test_rate_limiter_blocks_after_limit():
    async def run() -> None:
        limiter = RateLimiter(limit_per_minute=2)
        results = [await limiter.check("1.2.3.4") for _ in range(4)]
        assert results == [True, True, False, False]
        # a different key gets its own bucket
        assert await limiter.check("5.6.7.8") is True

    asyncio.run(run())


def test_rate_limiter_disabled_when_zero():
    async def run() -> None:
        limiter = RateLimiter(limit_per_minute=0)
        assert all([await limiter.check("x") for _ in range(20)])

    asyncio.run(run())


def test_search_rate_limit_enforced(monkeypatch, tmp_path):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        state.search_rate_limiter = RateLimiter(limit_per_minute=1)
        try:
            r1 = client.post("/search", json={"query": "poverty", "mode": "keyword"})
            r2 = client.post("/search", json={"query": "poverty", "mode": "keyword"})
            assert r2.status_code == 429
        finally:
            state.search_rate_limiter = RateLimiter(limit_per_minute=state.settings.rate_limit_search_per_minute)
