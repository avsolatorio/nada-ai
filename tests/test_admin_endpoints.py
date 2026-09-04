"""Smoke tests for admin / jobs endpoints with stubbed ops.

The ingest service functions are monkey-patched to no-op coroutines so we don't
require a live OpenSearch cluster or the embedding model. These tests focus on:

* ``admin_auth`` (gated only when ``NADA_ADMIN_API_KEY`` is set)
* HTTP status codes (``202`` on accept, ``409`` on single-flight collision)
* ``GET /jobs`` and ``GET /jobs/{id}`` reflecting status transitions
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from nada_ai.app import admin as admin_module
from nada_ai.app.jobs import JobRegistry
from nada_ai.app.main import app, state
from nada_ai.search.factory import create_search_backend
from nada_ai.settings import Settings


@pytest.fixture(autouse=True)
def _opensearch_backend_by_default(monkeypatch):
    """This file's tests exercise OpenSearch-specific admin routes throughout
    (only test_put_index_template_501_when_qdrant wants qdrant, and it
    overrides state.settings directly post-lifespan, unaffected by this) —
    pin the backend explicitly rather than relying on whatever
    NADA_SEARCH_BACKEND currently defaults to."""
    monkeypatch.setenv("NADA_SEARCH_BACKEND", "opensearch")


def _fresh_state() -> None:
    """Replace mutable parts of the module-level ``state`` so tests are isolated."""
    state.jobs = JobRegistry()


def test_admin_auth_required_when_env_set(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "secret")
    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/index", json={"recreate": False})
    assert r.status_code == 401


def test_admin_auth_optional_when_env_unset(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    monkeypatch.setattr(
        admin_module, "create_index_op", lambda settings, recreate=False: {"index": "x", "dim": 0}
    )

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/index", json={"recreate": False})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] in {"pending", "running", "succeeded"}
    assert body["kind"] == "create_index"


def test_admin_auth_passes_with_correct_key(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "secret")
    monkeypatch.setattr(
        admin_module, "create_index_op", lambda settings, recreate=False: {"index": "x", "dim": 0}
    )

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/index", json={"recreate": False}, headers={"X-NADA-Admin-Key": "secret"})
    assert r.status_code == 202


def test_put_index_template_returns_json(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    monkeypatch.setattr(
        admin_module,
        "put_index_template_op",
        lambda settings: {"dim": 384, "template": {"template": "nada-ai-nada-metadata-template"}},
    )

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/index/template")
    assert r.status_code == 200
    assert r.json()["dim"] == 384
    assert r.json()["template"]["template"] == "nada-ai-nada-metadata-template"


def test_put_index_template_requires_admin_key_when_configured(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_API_KEY", "secret")
    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/index/template")
    assert r.status_code == 401


def test_put_index_template_501_when_qdrant(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        _fresh_state()
        prev_settings, prev_client = state.settings, state.client
        state.settings = Settings(search_backend="qdrant")
        state.client = None
        try:
            r = client.post("/admin/index/template")
        finally:
            state.settings, state.client = prev_settings, prev_client
    assert r.status_code == 501


def test_create_index_returns_409_when_already_running(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    gate = threading.Event()

    def slow(settings, recreate=False):
        gate.wait(timeout=5)
        return {"index": "x", "dim": 0}

    monkeypatch.setattr(admin_module, "create_index_op", slow)

    with TestClient(app) as client:
        _fresh_state()
        r1 = client.post("/admin/index", json={"recreate": False})
        assert r1.status_code == 202
        r2 = client.post("/admin/index", json={"recreate": False})
        assert r2.status_code == 409
        body = r2.json()
        assert body["job"]["id"] == r1.json()["id"]
        gate.set()


def test_jobs_list_and_get(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    monkeypatch.setattr(
        admin_module,
        "create_index_op",
        lambda settings, recreate=False: {"index": "x", "dim": 7, "recreated": False},
    )

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/index", json={"recreate": False})
        assert r.status_code == 202
        job_id = r.json()["id"]

        for _ in range(50):
            sj = client.get(f"/jobs/{job_id}")
            assert sj.status_code == 200
            if sj.json()["status"] == "succeeded":
                break
            import time as _time

            _time.sleep(0.02)
        assert sj.json()["status"] == "succeeded"
        assert sj.json()["result"] == {"index": "x", "dim": 7, "recreated": False}

        listing = client.get("/jobs")
        assert listing.status_code == 200
        ids = {j["id"] for j in listing.json()["jobs"]}
        assert job_id in ids


def test_jobs_get_404():
    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/jobs/nope")
    assert r.status_code == 404


def test_ingest_by_ids_validates_idnos(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        _fresh_state()
        r = client.post(
            "/admin/ingest/by-ids",
            json={"idnos": ["   ", ""], "metadata_type": "indicator"},
        )
    assert r.status_code == 400


def test_ingest_from_catalog_singleflight(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    gate = threading.Event()

    def slow(settings, catalog_type="timeseries", *args, **kwargs):
        gate.wait(timeout=5)
        return {"indexed": 0, "errors": [], "rows": 0, "catalog_type": catalog_type, "index": "x"}

    monkeypatch.setattr(admin_module, "index_from_catalog_op", slow)

    with TestClient(app) as client:
        _fresh_state()
        r1 = client.post("/admin/ingest/from-catalog", json={"catalog_type": "timeseries"})
        assert r1.status_code == 202
        r2 = client.post("/admin/ingest/from-catalog", json={"catalog_type": "document"})
        assert r2.status_code == 202
        assert r1.json()["id"] != r2.json()["id"]
        r3 = client.post("/admin/ingest/from-catalog", json={"catalog_type": "timeseries"})
        assert r3.status_code == 409
        gate.set()


def test_index_delete_requires_confirm():
    with TestClient(app) as client:
        _fresh_state()
        r = client.delete("/admin/index")
    assert r.status_code == 400


def test_index_stats_passes_through_async_client(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    fake = MagicMock()

    with TestClient(app) as client:
        _fresh_state()
        fake.indices.stats = AsyncMock(
            return_value={
                "indices": {
                    state.settings.index_name: {
                        "primaries": {
                            "docs": {"count": 42},
                            "store": {"size_in_bytes": 1024},
                        }
                    }
                }
            }
        )
        prev_client, prev_search = state.client, state.search
        state.client = fake
        state.search = create_search_backend(state.settings, fake)
        try:
            r = client.get("/admin/index/stats")
        finally:
            state.client = prev_client
            state.search = prev_search
    assert r.status_code == 200
    body = r.json()
    assert body["docs"] == 42
    assert body["size_bytes"] == 1024
    assert body["index"] == state.settings.index_name


def test_admin_doc_get_passes_through(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)

    fake = MagicMock()
    fake.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {"_id": "abc", "_score": 1.0, "_source": {"idno": "WB_X"}},
                ]
            }
        }
    )

    with TestClient(app) as client:
        _fresh_state()
        prev_client, prev_search = state.client, state.search
        state.client = fake
        state.search = create_search_backend(state.settings, fake)
        try:
            r = client.get("/admin/docs/WB_X")
        finally:
            state.client = prev_client
            state.search = prev_search
    assert r.status_code == 200
    body = r.json()
    assert body["idno"] == "WB_X"
    assert body["count"] == 1
    assert body["hits"][0]["_id"] == "abc"


def test_jobs_list_invalid_status_returns_400():
    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/jobs?status=bogus")
    assert r.status_code == 400


def test_cancel_running_job_via_endpoint(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    gate = threading.Event()

    def slow(settings, recreate=False):
        gate.wait(timeout=5)
        return {"index": "x", "dim": 0}

    monkeypatch.setattr(admin_module, "create_index_op", slow)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/index", json={"recreate": False})
        assert r.status_code == 202
        job_id = r.json()["id"]
        # Issue cancel; thread is blocked on gate, so cancel should mark cancelled-or-running.
        rc = client.delete(f"/jobs/{job_id}")
        assert rc.status_code == 200
        gate.set()
        # Eventually terminal (cancelled or succeeded depending on timing of the thread).
        import time as _time

        for _ in range(50):
            s = client.get(f"/jobs/{job_id}").json()
            if s["status"] in {"cancelled", "succeeded", "failed"}:
                break
            _time.sleep(0.02)
        assert s["status"] in {"cancelled", "succeeded", "failed"}


def test_webhook_and_admin_index_dedupe_same_idno(monkeypatch):
    """Regression test: /admin/catalog/{idno}/index and the created/updated
    webhook used to submit under different job-registry keys (index:... vs
    reindex:...) for the exact same (metadata_type, idno) write, so they could
    run concurrently with zero coordination. They now share
    content_sync_job_key() and must single-flight against each other."""
    import nada_ai.app.catalog_admin as catalog_admin_module
    import nada_ai.app.webhooks as webhooks_module

    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    gate = threading.Event()

    def slow_index(settings, idnos, metadata_type, force, embedding=None):
        gate.wait(timeout=5)
        return {"indexed": 1, "errors": []}

    monkeypatch.setattr(catalog_admin_module, "index_ids_op", slow_index)
    monkeypatch.setattr(webhooks_module, "index_ids_op", slow_index)
    monkeypatch.setattr(webhooks_module, "delete_by_idno_op", lambda settings, idno: {"deleted": 0})

    with TestClient(app) as client:
        _fresh_state()
        r1 = client.post(
            "/admin/catalog/SAME_IDNO/index",
            json={"metadata_type": "indicator", "force": False},
        )
        assert r1.status_code == 202

        r2 = client.post(
            "/webhooks/catalog",
            json={"event": "updated", "idno": "SAME_IDNO", "metadata_type": "indicator"},
        )
        assert r2.status_code == 409
        body = r2.json()
        assert body["job"]["id"] == r1.json()["id"]
        gate.set()


# Avoid leaving asyncio mocks around between tests.
def teardown_function(_) -> None:
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(state.jobs.shutdown())
        loop.close()
    except Exception:
        pass
