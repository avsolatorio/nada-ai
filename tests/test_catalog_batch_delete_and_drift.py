"""Tests for POST /admin/catalog/delete (batch) and GET /admin/embeddings/drift."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

from nada_ai.app import catalog_admin as catalog_module
from nada_ai.app.jobs import JobRegistry
from nada_ai.app.main import app, state
from nada_ai.ingest.service import delete_by_idnos_op


def _fresh_state() -> None:
    state.jobs = JobRegistry()


def test_batch_delete_validates_idnos(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/catalog/delete", json={"idnos": ["   ", ""]})
    assert r.status_code == 400


def test_batch_delete_calls_op_with_all_idnos(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    captured: dict = {}

    def fake_op(settings, idnos):
        captured["idnos"] = idnos
        return {"backend": "qdrant", "collection": "x", "idnos": idnos, "operation": "completed"}

    monkeypatch.setattr(catalog_module, "delete_by_idnos_op", fake_op)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/catalog/delete", json={"idnos": ["A", " B ", "C", ""]})

    assert r.status_code == 200
    assert captured["idnos"] == ["A", "B", "C"]
    assert r.json()["operation"] == "completed"


def test_batch_delete_error_path_returns_503(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)

    def failing_op(settings, idnos):
        raise RuntimeError("backend down")

    monkeypatch.setattr(catalog_module, "delete_by_idnos_op", failing_op)

    with TestClient(app) as client:
        _fresh_state()
        r = client.post("/admin/catalog/delete", json={"idnos": ["A"]})

    assert r.status_code == 503


def test_delete_by_idnos_op_dispatches_by_backend(monkeypatch):
    """delete_by_idnos_op must route to the qdrant or opensearch batch helper by settings.search_backend."""
    import nada_ai.ingest.service as service_module

    calls = []
    monkeypatch.setattr(service_module, "_delete_qdrant_batch", lambda s, idnos: calls.append(("qdrant", idnos)))
    monkeypatch.setattr(
        service_module, "_delete_opensearch_batch", lambda s, idnos: calls.append(("opensearch", idnos))
    )

    class FakeSettings:
        search_backend = "qdrant"

    delete_by_idnos_op(FakeSettings(), ["X", "Y"])
    assert calls == [("qdrant", ["X", "Y"])]

    calls.clear()

    class FakeSettingsOS:
        search_backend = "opensearch"

    delete_by_idnos_op(FakeSettingsOS(), ["Z"])
    assert calls == [("opensearch", ["Z"])]


def test_embedding_drift_no_model_loaded(monkeypatch):
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)

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
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)

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
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)

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
