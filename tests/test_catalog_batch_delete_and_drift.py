"""Tests for POST /admin/catalog/index (batch), POST /admin/catalog/delete
(batch), and GET /admin/embeddings/drift."""

from __future__ import annotations

import threading
import time
from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

from nada_ai.app import catalog_admin as catalog_module
from nada_ai.app.jobs import JobRegistry
from nada_ai.app.main import app, state
from nada_ai.ingest.service import delete_by_sids_op


def _fresh_state() -> None:
    state.jobs = JobRegistry()


def test_batch_index_validates_sids(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")
    with TestClient(app) as client:
        _fresh_state()
        assert client.post("/admin/catalog/index", json={"sids": []}).status_code == 422
        assert client.post("/admin/catalog/index", json={"sids": ["WB_X"]}).status_code == 422
        assert client.post("/admin/catalog/index", json={"sids": [0]}).status_code == 422


def test_batch_index_singleflight_and_passes_all_options(monkeypatch):
    """The batch-by-ids route single-flights on its studies and forwards recreate_index/show_progress_bar/
    buffer_size through to index_sids_op."""
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")
    gate = threading.Event()
    captured: dict = {}

    def slow(settings, sids, force, recreate_index, show_progress_bar, buffer_size, embedding=None):
        captured["args"] = (sids, force, recreate_index, show_progress_bar, buffer_size)
        gate.wait(timeout=5)
        return {"indexed": 0, "errors": [], "requested": len(sids), "index": "x"}

    monkeypatch.setattr(catalog_module, "index_sids_op", slow)

    with TestClient(app) as client:
        _fresh_state()
        r1 = client.post(
            "/admin/catalog/index",
            json={"sids": [2, 1, 2], "recreate_index": True, "show_progress_bar": True, "buffer_size": 250},
        )
        assert r1.status_code == 202
        r2 = client.post("/admin/catalog/index", json={"sids": [1, 2]})
        assert r2.status_code == 409  # the same set of studies, whatever the order
        gate.set()
        for _ in range(200):  # the job warms the embedding model before it calls the op
            if client.get(f"/jobs/{r1.json()['id']}").json()["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.05)

    assert captured["args"] == ([2, 1], False, True, True, 250)


def test_batch_delete_validates_sids(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")
    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/catalog/delete", json={"sids": ["A"]})
    assert r.status_code == 422


def test_batch_delete_calls_op_with_all_sids(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")
    captured: dict = {}

    def fake_op(settings, sids):
        captured["sids"] = sids
        return {"backend": "qdrant", "collection": "x", "sids": sids, "operation": "completed"}

    monkeypatch.setattr(catalog_module, "delete_by_sids_op", fake_op)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/catalog/delete", json={"sids": [3, 1, 3, 2]})

    assert r.status_code == 200
    assert captured["sids"] == [3, 1, 2]
    assert r.json()["operation"] == "completed"


def test_batch_delete_error_path_returns_503(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")

    def failing_op(settings, sids):
        raise RuntimeError("backend down")

    monkeypatch.setattr(catalog_module, "delete_by_sids_op", failing_op)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/catalog/delete", json={"sids": [1]})

    assert r.status_code == 503


def test_delete_by_sids_op_dispatches_by_backend(monkeypatch):
    """delete_by_sids_op must route to the qdrant or opensearch helper by settings.search_backend."""
    import nada_ai.ingest.service as service_module

    calls = []
    monkeypatch.setattr(service_module, "_delete_qdrant_by_sids", lambda s, sids: calls.append(("qdrant", sids)))
    monkeypatch.setattr(
        service_module, "_delete_opensearch_by_sids", lambda s, sids: calls.append(("opensearch", sids))
    )

    class FakeSettings:
        search_backend = "qdrant"
        report_search_index_state_enabled = False

    delete_by_sids_op(FakeSettings(), [1, 2])
    assert calls == [("qdrant", [1, 2])]

    calls.clear()

    class FakeSettingsOS:
        search_backend = "opensearch"
        report_search_index_state_enabled = False

    delete_by_sids_op(FakeSettingsOS(), [3])
    assert calls == [("opensearch", [3])]


def test_embedding_drift_no_model_loaded(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")

    fake_vectors = MagicMock()
    fake_vectors.size = 384
    fake_info = MagicMock()
    fake_info.config.params.vectors = fake_vectors

    with TestClient(app) as client:
        _fresh_state()
        prev_settings, prev_search, prev_embedding = state.settings, state.search, state.embedding
        fake_search = MagicMock()
        fake_search.client = AsyncMock()
        fake_search.client.get_collection = AsyncMock(return_value=fake_info)
        state.settings = prev_settings.model_copy(update={"search_backend": "qdrant"})
        state.search = fake_search
        state.embedding = None
        try:
            r = client.get("/admin/embeddings/drift")
        finally:
            state.settings, state.search, state.embedding = prev_settings, prev_search, prev_embedding

    assert r.status_code == 200
    body = r.json()
    assert body["configured_dimension"] is None
    assert body["stored_dimension"] == 384
    assert body["dimension_match"] is None
    assert "warning" not in body


def test_embedding_drift_mismatch_warns(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")

    fake_vectors = MagicMock()
    fake_vectors.size = 384
    fake_info = MagicMock()
    fake_info.config.params.vectors = fake_vectors

    fake_embedding = MagicMock()
    fake_embedding.embedding_dimension.return_value = 768

    with TestClient(app) as client:
        _fresh_state()
        prev_settings, prev_search, prev_embedding = state.settings, state.search, state.embedding
        fake_search = MagicMock()
        fake_search.client = AsyncMock()
        fake_search.client.get_collection = AsyncMock(return_value=fake_info)
        state.settings = prev_settings.model_copy(update={"search_backend": "qdrant"})
        state.search = fake_search
        state.embedding = fake_embedding
        try:
            r = client.get("/admin/embeddings/drift")
        finally:
            state.settings, state.search, state.embedding = prev_settings, prev_search, prev_embedding

    assert r.status_code == 200
    body = r.json()
    assert body["configured_dimension"] == 768
    assert body["stored_dimension"] == 384
    assert body["dimension_match"] is False
    assert "warning" in body


def test_embedding_drift_matching_dimensions(monkeypatch):
    monkeypatch.setenv("NADA_ADMIN_AUTH_DISABLED", "true")

    fake_vectors = MagicMock()
    fake_vectors.size = 384
    fake_info = MagicMock()
    fake_info.config.params.vectors = fake_vectors

    fake_embedding = MagicMock()
    fake_embedding.embedding_dimension.return_value = 384

    with TestClient(app) as client:
        _fresh_state()
        prev_settings, prev_search, prev_embedding = state.settings, state.search, state.embedding
        fake_search = MagicMock()
        fake_search.client = AsyncMock()
        fake_search.client.get_collection = AsyncMock(return_value=fake_info)
        state.settings = prev_settings.model_copy(update={"search_backend": "qdrant"})
        state.search = fake_search
        state.embedding = fake_embedding
        try:
            r = client.get("/admin/embeddings/drift")
        finally:
            state.settings, state.search, state.embedding = prev_settings, prev_search, prev_embedding

    assert r.status_code == 200
    body = r.json()
    assert body["dimension_match"] is True
    assert "warning" not in body
