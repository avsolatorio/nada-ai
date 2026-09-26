"""Tests for the per-caller API key store, role-based auth, and audit trail.

Uses ``NADA_API_KEYS_PATH`` / ``NADA_AUDIT_LOG_PATH`` pointed at tmp files so
these tests never touch the real ``config/`` directory.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from nada_ai.app import admin as admin_module
from nada_ai.app.jobs import JobRegistry
from nada_ai.app.keys_store import KeyStoreError, Role, create_key
from nada_ai.app.main import app, state
from nada_ai.app.rate_limit import RateLimiter
from nada_ai.search.ports import SearchOutcome
from nada_ai.settings import Settings


def _fresh_state() -> None:
    state.jobs = JobRegistry()


def _isolate_stores(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_API_KEYS_PATH", str(tmp_path / "api_keys.json"))
    monkeypatch.setenv("NADA_AUDIT_LOG_PATH", str(tmp_path / "audit.log"))


def _issue_key(tmp_path, role: Role) -> tuple[str, str]:
    """Write a key straight into the store (no route: issuing one over HTTP needs an admin already)."""
    settings = Settings(api_keys_path=str(tmp_path / "api_keys.json"))
    record, raw_key = asyncio.run(create_key("seed", role, settings, asyncio.Lock()))
    return raw_key, record.id


def test_unconfigured_server_answers_503_not_anonymous_admin(monkeypatch, tmp_path):
    """No env key, no stored key, auth not disabled => protected routes refuse everyone, with the reason."""
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/jobs")
        with_header = client.get("/jobs", headers={"X-NADA-Admin-Key": "anything"})
    assert r.status_code == with_header.status_code == 503
    assert "NADA_ADMIN_API_KEY" in r.json()["detail"]
    assert "NADA_ADMIN_AUTH_DISABLED" in r.json()["detail"]


def test_auth_disabled_lets_every_caller_through(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")
    # /admin/index is OpenSearch-only (_require_opensearch) — pin the backend
    # explicitly rather than relying on whatever NADA_SEARCH_BACKEND defaults to.
    monkeypatch.setenv("NADA_SEARCH_BACKEND", "opensearch")
    _isolate_stores(monkeypatch, tmp_path)
    monkeypatch.setattr(admin_module, "create_index_op", lambda settings, recreate=False: {"index": "x"})

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/index", json={"recreate": False})
    assert r.status_code == 202


def test_startup_refuses_auth_disabled_with_an_env_key(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "secret")
    _isolate_stores(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="NADA_ADMIN_AUTH_DISABLED"), TestClient(app):
        pass


def test_startup_refuses_auth_disabled_with_an_active_stored_key(monkeypatch, tmp_path):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")
    _isolate_stores(monkeypatch, tmp_path)
    _issue_key(tmp_path, Role.read)

    with pytest.raises(RuntimeError, match="NADA_ADMIN_AUTH_DISABLED"), TestClient(app):
        pass


def test_revoking_the_last_stored_key_does_not_open_the_server(monkeypatch, tmp_path):
    _isolate_stores(monkeypatch, tmp_path)
    admin_key, admin_id = _issue_key(tmp_path, Role.admin)

    with TestClient(app) as client:
        _fresh_state()
        headers = {"X-NADA-Admin-Key": admin_key}
        assert client.get("/jobs", headers=headers).status_code == 200
        assert client.delete(f"/admin/keys/{admin_id}", headers=headers).status_code == 200
        after = client.get("/jobs")
    assert after.status_code == 503


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '["a list, not an object"]',
        '{"keys": "not a list"}',
        '{"keys": [{"id": "k1", "name": "no role or hash"}]}',
        '{"keys": [{"id": "k1", "name": "n", "role": "superuser", "key_hash": "h", "created_at": "t"}]}',
    ],
)
def test_an_unreadable_key_store_answers_503_and_is_never_overwritten(monkeypatch, tmp_path, content):
    """Treating it as "no keys" would open the server (before), or lock out every stored key silently; and creating
    a key would overwrite the file with only the new one."""
    _isolate_stores(monkeypatch, tmp_path)
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "legacy-secret")
    keys_file = tmp_path / "api_keys.json"
    keys_file.write_text('{"keys": []}', encoding="utf-8")  # readable at startup; broken while running

    with TestClient(app) as client:
        _fresh_state()
        keys_file.write_text(content, encoding="utf-8")
        stored = client.get("/jobs", headers={"X-NADA-Admin-Key": "nada_some_stored_key"})
        env = client.get("/jobs", headers={"X-NADA-Admin-Key": "legacy-secret"})
        create = client.post(
            "/admin/keys", json={"name": "x", "role": "read"}, headers={"X-NADA-Admin-Key": "legacy-secret"}
        )
    assert stored.status_code == 503
    assert env.status_code == 200  # the env key never reads the store
    assert create.status_code == 503
    assert keys_file.read_text(encoding="utf-8") == content


def test_startup_fails_on_an_unreadable_key_store(monkeypatch, tmp_path):
    """Also with the env key set, which on its own would never read the store."""
    _isolate_stores(monkeypatch, tmp_path)
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "legacy-secret")
    (tmp_path / "api_keys.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(KeyStoreError), TestClient(app):
        pass


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
        r = client.post("/admin/keys", json={"name": "x", "role": "read"}, headers={"X-NADA-Admin-Key": raw_key})
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
    """This must never touch a real search backend: /search's first call has
    to succeed for the rate limiter (not the backend) to be what produces the
    429 on the second call. Un-mocked, this test's outcome — and, worse, its
    process stability — depends on whatever happens to be reachable at
    localhost:6333/:9200 when it runs (e.g. a Qdrant container left running
    from manual testing), which previously caused a real, unmocked keyword
    search to reach a live Qdrant and segfault computing a real FastEmbed
    BM25 sparse embedding — a native-extension crash, not a test assertion
    failure, so it took the whole pytest process down with it."""
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    _isolate_stores(monkeypatch, tmp_path)

    with TestClient(app) as client:
        _fresh_state()
        state.search_rate_limiter = RateLimiter(limit_per_minute=1)
        prev_search = state.search
        state.search = AsyncMock()
        state.search.search = AsyncMock(return_value=SearchOutcome(total=0, hits=[]))
        try:
            r1 = client.post("/search", json={"query": "poverty", "mode": "keyword"})
            r2 = client.post("/search", json={"query": "poverty", "mode": "keyword"})
            assert r1.status_code == 200
            assert r2.status_code == 429
        finally:
            state.search_rate_limiter = RateLimiter(limit_per_minute=state.settings.rate_limit_search_per_minute)
            state.search = prev_search
