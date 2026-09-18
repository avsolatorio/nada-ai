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


def test_ingest_from_catalog_works_under_qdrant(monkeypatch):
    """index_from_catalog_op dispatches through search.factory.create_ingest_writer,
    which is backend-agnostic — this route must not require an OpenSearch client."""
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    monkeypatch.setattr(
        admin_module,
        "index_from_catalog_op",
        lambda settings, catalog_type="timeseries", *a, **kw: {
            "indexed": 0, "errors": [], "rows": 0, "catalog_type": catalog_type, "index": "x"
        },
    )

    with TestClient(app) as client:
        _fresh_state()
        prev_settings, prev_client = state.settings, state.client
        state.settings = Settings(search_backend="qdrant")
        state.client = None
        try:
            r = client.post("/admin/ingest/from-catalog", json={"catalog_type": "timeseries"})
        finally:
            state.settings, state.client = prev_settings, prev_client
    assert r.status_code == 202


def test_ingest_from_catalog_all_submits_one_job_per_type(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    seen_types: list[str] = []

    def fake(settings, catalog_type="timeseries", *a, **kw):
        seen_types.append(catalog_type)
        return {"indexed": 0, "errors": [], "rows": 0, "catalog_type": catalog_type, "index": "x"}

    monkeypatch.setattr(admin_module, "index_from_catalog_op", fake)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/ingest/from-catalog/all", json={})
    assert r.status_code == 202
    body = r.json()
    assert body["recreated"] is False
    every_type = {
        "document", "timeseries", "survey", "geospatial", "timeseriesdb", "table", "script", "image", "video",
    }
    assert {j["catalog_type"] for j in body["jobs"]} == every_type
    assert all(not j["already_running"] for j in body["jobs"])
    assert len({j["job"]["id"] for j in body["jobs"]}) == len(every_type)  # one distinct job per type


def test_ingest_from_catalog_all_reports_already_running_type(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    gate = threading.Event()

    def slow(settings, catalog_type="timeseries", *a, **kw):
        gate.wait(timeout=5)
        return {"indexed": 0, "errors": [], "rows": 0, "catalog_type": catalog_type, "index": "x"}

    monkeypatch.setattr(admin_module, "index_from_catalog_op", slow)

    with TestClient(app) as client:
        _fresh_state()
        r1 = client.post("/admin/ingest/from-catalog", json={"catalog_type": "document"})
        assert r1.status_code == 202
        in_flight_id = r1.json()["id"]

        r2 = client.post("/admin/ingest/from-catalog/all", json={})
        assert r2.status_code == 202
        by_type = {j["catalog_type"]: j for j in r2.json()["jobs"]}
        assert by_type["document"]["already_running"] is True
        assert by_type["document"]["job"]["id"] == in_flight_id
        assert by_type["timeseries"]["already_running"] is False
        gate.set()


def test_ingest_from_catalog_all_recreates_once_not_per_type(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    recreate_calls: list[bool] = []

    def fake_create_index_op(settings, recreate=False):
        recreate_calls.append(recreate)
        return {"index": "x", "dim": 0, "recreated": recreate}

    monkeypatch.setattr(admin_module, "create_index_op", fake_create_index_op)
    monkeypatch.setattr(
        admin_module,
        "index_from_catalog_op",
        lambda settings, catalog_type="timeseries", *a, **kw: {
            "indexed": 0, "errors": [], "rows": 0, "catalog_type": catalog_type, "index": "x"
        },
    )

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/ingest/from-catalog/all", json={"recreate_index": True})
    assert r.status_code == 202
    assert r.json()["recreated"] is True
    assert recreate_calls == [True]  # exactly one recreate call, not four


def test_ingest_reconcile_triggers_poll_once(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    mock_poll_once = AsyncMock(return_value={"polled": 3})
    monkeypatch.setattr("nada_ai.app.reconcile_scheduler.poll_once", mock_poll_once)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/ingest/reconcile")
    assert r.status_code == 200
    assert r.json() == {"polled": 3}
    mock_poll_once.assert_awaited_once()


def test_search_index_diff_missing_list_returns_items(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.ingest.search_index_sync import DiffItem, DiffPage

    monkeypatch.setattr(
        "nada_ai.ingest.search_index_sync.list_diff_missing",
        lambda settings, **kw: DiffPage(items=[DiffItem(idno="A", type="survey", last_error="boom")], total=1382),
    )

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/diff/missing?object_type=survey&limit=50&offset=100")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1382
    assert body["items"] == [{"idno": "A", "type": "survey", "last_error": "boom"}]


def test_search_index_diff_missing_list_passes_has_error_through(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.ingest.search_index_sync import DiffPage

    mock_list = MagicMock(return_value=DiffPage(items=[], total=0))
    monkeypatch.setattr("nada_ai.ingest.search_index_sync.list_diff_missing", mock_list)

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/diff/missing?object_type=survey&has_error=true")
    assert r.status_code == 200
    assert mock_list.call_args.kwargs["has_error"] is True


def test_search_index_diff_stale_list_returns_items(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.ingest.search_index_sync import DiffItem, DiffPage

    monkeypatch.setattr(
        "nada_ai.ingest.search_index_sync.list_diff_stale",
        lambda settings, **kw: DiffPage(items=[DiffItem(idno="Z", type=None)], total=1),
    )

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/diff/stale?object_type=survey")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"] == [{"idno": "Z", "type": None, "last_error": None}]


def test_search_index_diff_missing_list_400_when_not_configured(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.ingest.search_index_sync import SearchIndexSyncError

    def raises(*a, **kw):
        raise SearchIndexSyncError("not configured")

    monkeypatch.setattr("nada_ai.ingest.search_index_sync.list_diff_missing", raises)

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/diff/missing?object_type=survey")
    assert r.status_code == 400


def test_search_index_diff_summary_returns_counts(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.ingest.search_index_sync import DiffItem, DiffPage, ObjectTypeSummary

    monkeypatch.setattr(
        "nada_ai.ingest.search_index_sync.list_diff_missing",
        lambda settings, **kw: DiffPage(items=[DiffItem(idno="A", type="survey")], total=7),
    )
    monkeypatch.setattr(
        "nada_ai.ingest.search_index_sync.list_diff_stale",
        lambda settings, **kw: DiffPage(items=[], total=2),
    )
    monkeypatch.setattr(
        "nada_ai.ingest.search_index_sync.get_object_type_summary",
        lambda settings, object_type: ObjectTypeSummary(
            object_type="survey",
            catalog_total=9,
            state={"indexed": 2, "pending": 5, "failed": 0, "deleted": 0},
        ),
    )

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/diff-summary?object_type=survey")
    assert r.status_code == 200
    assert r.json() == {
        "object_type": "survey",
        "catalog_total": 9,
        "state": {"indexed": 2, "pending": 5, "failed": 0, "deleted": 0},
        "missing_total": 7,
        "stale_total": 2,
    }


def test_search_index_type_breakdown_returns_items(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.ingest.search_index_sync import TypeBreakdownItem

    monkeypatch.setattr(
        "nada_ai.ingest.search_index_sync.list_type_breakdown",
        lambda settings, object_type: [
            TypeBreakdownItem(data_type="survey", catalog_total=1200, indexed=120, missing=1080, stale=0, errors=5),
            TypeBreakdownItem(data_type="geospatial", catalog_total=40, indexed=38, missing=2, stale=1, errors=2),
        ],
    )

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/type-breakdown?object_type=survey")
    assert r.status_code == 200
    body = r.json()
    assert body["object_type"] == "survey"
    assert body["items"] == [
        {"data_type": "survey", "catalog_total": 1200, "indexed": 120, "missing": 1080, "stale": 0, "errors": 5},
        {"data_type": "geospatial", "catalog_total": 40, "indexed": 38, "missing": 2, "stale": 1, "errors": 2},
    ]


def test_search_index_type_breakdown_400_when_not_configured(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.ingest.search_index_sync import SearchIndexSyncError

    def raises(*a, **kw):
        raise SearchIndexSyncError("not configured")

    monkeypatch.setattr("nada_ai.ingest.search_index_sync.list_type_breakdown", raises)

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/type-breakdown?object_type=survey")
    assert r.status_code == 400


def test_search_index_diff_summary_400_when_not_configured(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.ingest.search_index_sync import SearchIndexSyncError

    def raises(*a, **kw):
        raise SearchIndexSyncError("not configured")

    monkeypatch.setattr("nada_ai.ingest.search_index_sync.list_diff_missing", raises)

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/diff-summary?object_type=survey")
    assert r.status_code == 400


def test_search_index_reconcile_diff_submits_job(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    fake_summary = {"missing_total": 0, "stale_total": 0, "indexed": 0, "deleted": 0, "failed": 0, "skipped": 0}
    monkeypatch.setattr("nada_ai.ingest.search_index_sync.reconcile_diff_once", lambda settings, **kw: fake_summary)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/search-index/reconcile-diff?object_type=survey")
    assert r.status_code == 202
    body = r.json()
    assert body["kind"] == "search_index_reconcile_diff"
    assert body["key"] == "search_index_reconcile_diff:survey:all"


def test_search_index_reconcile_diff_singleflights_per_object_type(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    gate = threading.Event()

    def slow(settings, **kw):
        gate.wait(timeout=5)
        return {"missing_total": 0, "stale_total": 0, "indexed": 0, "deleted": 0, "failed": 0, "skipped": 0}

    monkeypatch.setattr("nada_ai.ingest.search_index_sync.reconcile_diff_once", slow)

    with TestClient(app) as client:
        _fresh_state()
        r1 = client.post("/admin/search-index/reconcile-diff?object_type=survey")
        assert r1.status_code == 202
        r2 = client.post("/admin/search-index/reconcile-diff?object_type=survey")
        assert r2.status_code == 409
        r3 = client.post("/admin/search-index/reconcile-diff?object_type=citation")
        assert r3.status_code == 202
        assert r3.json()["id"] != r1.json()["id"]
        gate.set()


def test_search_index_reconcile_diff_singleflights_per_data_type(monkeypatch):
    """Reconciling one data_type must not collide with, or block, another."""
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    gate = threading.Event()

    def slow(settings, **kw):
        gate.wait(timeout=5)
        return {"missing_total": 0, "stale_total": 0, "indexed": 0, "deleted": 0, "failed": 0, "skipped": 0}

    monkeypatch.setattr("nada_ai.ingest.search_index_sync.reconcile_diff_once", slow)

    with TestClient(app) as client:
        _fresh_state()
        r1 = client.post("/admin/search-index/reconcile-diff?object_type=survey&data_type=geospatial")
        assert r1.status_code == 202
        assert r1.json()["key"] == "search_index_reconcile_diff:survey:geospatial"
        r2 = client.post("/admin/search-index/reconcile-diff?object_type=survey&data_type=geospatial")
        assert r2.status_code == 409
        r3 = client.post("/admin/search-index/reconcile-diff?object_type=survey&data_type=document")
        assert r3.status_code == 202
        assert r3.json()["id"] != r1.json()["id"]
        gate.set()


def test_search_index_reconcile_diff_writes_progress_onto_the_job(monkeypatch):
    """The route must bind JobRegistry.set_progress to the submitted job id
    *before* the worker runs, same as index_from_catalog — otherwise Jobs
    shows '—' for the whole reconcile."""
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    started = threading.Event()
    captured: dict = {}

    def capture(settings, **kw):
        captured["progress_cb"] = kw.get("progress_cb")
        captured["cancel_token"] = kw.get("cancel_token")
        started.set()
        return {"missing_total": 0, "stale_total": 0, "indexed": 0, "deleted": 0, "failed": 0, "skipped": 0}

    monkeypatch.setattr("nada_ai.ingest.search_index_sync.reconcile_diff_once", capture)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/search-index/reconcile-diff?object_type=survey")
        assert r.status_code == 202
        job_id = r.json()["id"]
        assert started.wait(timeout=5)
        assert captured.get("cancel_token") is not None
        assert callable(captured.get("progress_cb"))
        captured["progress_cb"](
            {
                "processed": 1,
                "total": 4,
                "failed": 0,
                "current_idno": "WB_1",
                "percent": 25.0,
                "phase": "missing",
            }
        )
        body = client.get(f"/jobs/{job_id}").json()
        assert body["progress"]["total"] == 4
        assert body["progress"]["processed"] == 1
        assert body["progress"]["current_idno"] == "WB_1"


def test_search_index_status_returns_status(monkeypatch):
    from nada_ai.ingest.search_index_sync import SearchIndexStatus

    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    mock_get_status = MagicMock(
        return_value=SearchIndexStatus(
            status="ok",
            search_provider="nada-ai",
            tracking_enabled=True,
            queue={"pending": 2, "failed": 0},
            state={"indexed": 100},
        )
    )
    monkeypatch.setattr("nada_ai.ingest.search_index_sync.get_status", mock_get_status)

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/status")
    assert r.status_code == 200
    body = r.json()
    assert body["tracking_enabled"] is True
    assert body["queue"] == {"pending": 2, "failed": 0}
    mock_get_status.assert_called_once()


def test_search_index_status_400_when_not_configured(monkeypatch):
    from nada_ai.ingest.search_index_sync import SearchIndexSyncError

    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    mock_get_status = MagicMock(side_effect=SearchIndexSyncError("No search-index base URL configured."))
    monkeypatch.setattr("nada_ai.ingest.search_index_sync.get_status", mock_get_status)

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/status")
    assert r.status_code == 400


def test_search_index_status_503_on_unexpected_error(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    mock_get_status = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr("nada_ai.ingest.search_index_sync.get_status", mock_get_status)

    with TestClient(app) as client:
        _fresh_state()
        r = client.get("/admin/search-index/status")
    assert r.status_code == 503


def test_qdrant_search_backend_exposes_public_client_property():
    """Regression guard: admin.py's admin_qdrant_collection / admin_embedding_drift
    read the client via ``getattr(s.search, "client", None)`` — a public attribute
    that didn't exist (only the private ``_client`` did), so both routes always
    503'd with "Qdrant search backend has no client" regardless of real state."""
    from nada_ai.search.backend.qdrant.search_backend import QdrantSearchBackend

    backend = QdrantSearchBackend(Settings(search_backend="qdrant"))
    assert backend.client is backend._client


def test_qdrant_collection_endpoint_returns_info(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.search.backend.qdrant.search_backend import QdrantSearchBackend

    fake_info = MagicMock()
    fake_info.model_dump.return_value = {"points_count": 5}

    with TestClient(app) as client:
        _fresh_state()
        prev_settings, prev_search = state.settings, state.search
        state.settings = Settings(search_backend="qdrant")
        backend = QdrantSearchBackend(state.settings)
        backend._client = AsyncMock()
        backend._client.get_collection = AsyncMock(return_value=fake_info)
        state.search = backend
        expected_collection = state.settings.qdrant_collection
        try:
            r = client.get("/admin/qdrant/collection")
        finally:
            state.settings, state.search = prev_settings, prev_search
    assert r.status_code == 200
    body = r.json()
    assert body["collection"] == expected_collection
    assert body["info"] == {"points_count": 5}


def test_index_delete_requires_confirm():
    with TestClient(app) as client:
        _fresh_state()
        r = client.delete("/admin/index")
    assert r.status_code == 400


def test_catalog_type_counts_combines_qdrant_facet_and_catalog_totals(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.search.backend.qdrant.search_backend import QdrantSearchBackend

    def fake_search_metadata(params):
        # One "found" per catalog_type, keyed off the type param this route sends.
        found_by_type = {"document": 969, "timeseries": 307, "survey": 460, "geospatial": 30, "timeseriesdb": 4}
        return {"rows": [], "found": found_by_type.get(params["type"], 0)}

    monkeypatch.setattr("ai4data.discovery.catalog.http.search_metadata", fake_search_metadata)

    fake_hit_document = MagicMock(value="document", count=969)
    fake_hit_indicator = MagicMock(value="indicator", count=307)
    fake_hit_microdata = MagicMock(value="microdata", count=460)
    fake_hit_geospatial = MagicMock(value="geospatial", count=37)
    fake_hit_indicator_db = MagicMock(value="indicator-db", count=4)
    fake_facet_resp = MagicMock(
        hits=[fake_hit_document, fake_hit_indicator, fake_hit_microdata, fake_hit_geospatial, fake_hit_indicator_db]
    )

    with TestClient(app) as client:
        _fresh_state()
        prev_settings, prev_search = state.settings, state.search
        state.settings = Settings(search_backend="qdrant")
        backend = QdrantSearchBackend(state.settings)
        backend._client = AsyncMock()
        backend._client.facet = AsyncMock(return_value=fake_facet_resp)
        state.search = backend
        try:
            r = client.get("/admin/catalog/type-counts")
        finally:
            state.settings, state.search = prev_settings, prev_search

    assert r.status_code == 200
    body = r.json()["types"]
    assert body["document"] == {"catalog_total": 969, "indexed_documents": 969}
    assert body["timeseries"] == {"catalog_total": 307, "indexed_documents": 307}
    assert body["survey"] == {"catalog_total": 460, "indexed_documents": 460}
    assert body["geospatial"] == {"catalog_total": 30, "indexed_documents": 37}  # not 1:1 — documents, not idnos
    # NADA's "timeseriesdb" is stored under metadata.type "indicator-db"; types with no indexed docs report 0.
    assert body["timeseriesdb"] == {"catalog_total": 4, "indexed_documents": 4}
    assert body["video"] == {"catalog_total": 0, "indexed_documents": 0}


def test_catalog_type_counts_400_when_not_qdrant_backend(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        _fresh_state()
        prev_settings = state.settings
        state.settings = Settings(search_backend="opensearch")
        try:
            r = client.get("/admin/catalog/type-counts")
        finally:
            state.settings = prev_settings
    assert r.status_code == 400


def test_catalog_type_counts_survives_one_catalog_type_being_unreachable(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.search.backend.qdrant.search_backend import QdrantSearchBackend

    def flaky_search_metadata(params):
        if params["type"] == "geospatial":
            raise RuntimeError("catalog unreachable")
        return {"rows": [], "found": 5}

    monkeypatch.setattr("ai4data.discovery.catalog.http.search_metadata", flaky_search_metadata)
    fake_facet_resp = MagicMock(hits=[])

    with TestClient(app) as client:
        _fresh_state()
        prev_settings, prev_search = state.settings, state.search
        state.settings = Settings(search_backend="qdrant")
        backend = QdrantSearchBackend(state.settings)
        backend._client = AsyncMock()
        backend._client.facet = AsyncMock(return_value=fake_facet_resp)
        state.search = backend
        try:
            r = client.get("/admin/catalog/type-counts")
        finally:
            state.settings, state.search = prev_settings, prev_search

    assert r.status_code == 200
    body = r.json()["types"]
    assert body["geospatial"]["catalog_total"] is None
    assert body["document"]["catalog_total"] == 5


def test_qdrant_collection_delete_requires_confirm(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        _fresh_state()
        prev_settings = state.settings
        state.settings = Settings(search_backend="qdrant")
        try:
            r = client.delete("/admin/qdrant/collection")
        finally:
            state.settings = prev_settings
    assert r.status_code == 400


def test_qdrant_collection_delete_400_when_not_qdrant_backend(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        _fresh_state()
        prev_settings = state.settings
        state.settings = Settings(search_backend="opensearch")
        try:
            r = client.delete("/admin/qdrant/collection?confirm=true")
        finally:
            state.settings = prev_settings
    assert r.status_code == 400


def test_qdrant_collection_delete_drops_collection(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    from nada_ai.search.backend.qdrant.search_backend import QdrantSearchBackend

    with TestClient(app) as client:
        _fresh_state()
        prev_settings, prev_search = state.settings, state.search
        state.settings = Settings(search_backend="qdrant")
        backend = QdrantSearchBackend(state.settings)
        backend._client = AsyncMock()
        backend._client.delete_collection = AsyncMock(return_value=True)
        state.search = backend
        expected_collection = state.settings.qdrant_collection
        try:
            r = client.delete("/admin/qdrant/collection?confirm=true")
        finally:
            state.settings, state.search = prev_settings, prev_search
    assert r.status_code == 200
    body = r.json()
    assert body == {"collection": expected_collection, "deleted": True}
    backend._client.delete_collection.assert_awaited_once_with(collection_name=expected_collection)


def test_ingest_from_catalog_reports_live_progress(monkeypatch):
    """progress_cb passed into index_from_catalog_op must land on the job's
    own progress field, readable via GET /jobs/{id} before the job finishes —
    this is what a dashboard polls for a live progress bar."""
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    gate = threading.Event()

    def fake(settings, catalog_type="timeseries", *args, progress_cb=None, **kwargs):
        if progress_cb is not None:
            progress_cb({"processed": 3, "total": 10, "failed": 0, "current_idno": "X", "percent": 30.0})
        gate.wait(timeout=5)
        return {"indexed": 3, "errors": [], "rows": 10, "catalog_type": catalog_type, "index": "x"}

    monkeypatch.setattr(admin_module, "index_from_catalog_op", fake)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/ingest/from-catalog", json={"catalog_type": "timeseries"})
        assert r.status_code == 202
        job_id = r.json()["id"]

        for _ in range(50):
            sj = client.get(f"/jobs/{job_id}")
            if sj.json()["progress"]:
                break
            import time as _time

            _time.sleep(0.02)
        assert sj.json()["progress"] == {"processed": 3, "total": 10, "failed": 0, "current_idno": "X", "percent": 30.0}
        gate.set()


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
