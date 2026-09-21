"""GET /info: engine, capabilities and index summary (step 4 of the OpenSearch plan)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import NotFoundError
from qdrant_client.http.exceptions import ResponseHandlingException
from starlette.testclient import TestClient

from nada_ai.app import info as info_module
from nada_ai.app.info import capabilities_for, limits_for
from nada_ai.app.main import app, state
from nada_ai.app.rate_limit import RateLimiter
from nada_ai.app.studies_schemas import Engine, ErrorResponse, InfoResponse
from nada_ai.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures" / "studies_search"
ALL_MODES = frozenset({"browse", "lexical", "semantic", "hybrid"})


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@contextmanager
def _running(monkeypatch: pytest.MonkeyPatch, backend: str, **overrides: Any) -> Iterator[TestClient]:
    """The app with the given backend, and ``overrides`` swapped onto the shared state for the duration."""
    monkeypatch.setenv("NADA_SEARCH_BACKEND", backend)
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        previous = {name: getattr(state, name) for name in overrides}
        for name, value in overrides.items():
            setattr(state, name, value)
        try:
            yield client
        finally:
            for name, value in previous.items():
                setattr(state, name, value)


def _opensearch(*, studies: int | None = 1198, missing: tuple[str, ...] = (), down: bool = False) -> MagicMock:
    settings = Settings()
    client = MagicMock()
    if down:
        client.info = AsyncMock(side_effect=OpenSearchConnectionError("N/A", "refused", Exception("refused")))
    else:
        client.info = AsyncMock(return_value={"version": {"number": "3.6.0"}})

    metas = {
        settings.studies_index: {"generation": "20260920T150000Z-abc12345"},
        settings.index_name: {
            "generation": "20260920T150000Z-def67890",
            "embedding_model": "microsoft/harrier-oss-v1-270m",
            "embedding_dim": 640,
        },
    }

    async def get_mapping(index: str) -> dict[str, Any]:
        if index in missing:
            raise NotFoundError(404, "index_not_found_exception", {})
        return {index: {"mappings": {"_meta": metas[index]}}}

    client.indices.get_mapping = get_mapping
    client.count = AsyncMock(return_value={"count": studies})
    return client


def _qdrant(*, down: bool = False, collection_exists: bool = True) -> SimpleNamespace:
    client = MagicMock()
    if down:
        client.info = AsyncMock(side_effect=ResponseHandlingException(Exception("refused")))
    else:
        client.info = AsyncMock(return_value=SimpleNamespace(version="1.18.0"))
    client.collection_exists = AsyncMock(return_value=collection_exists)
    vectors = SimpleNamespace(size=640)
    client.get_collection = AsyncMock(
        return_value=SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))
    )
    return SimpleNamespace(client=client)


# ---------------------------------------------------------------------------------------
# OpenSearch
# ---------------------------------------------------------------------------------------


def test_opensearch_info_reports_engine_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    with _running(monkeypatch, "opensearch", client=_opensearch()) as client:
        response = client.get("/info")
    assert response.status_code == 200
    info = InfoResponse.model_validate(response.json())
    assert info.engine is Engine.opensearch
    assert info.engine_version == "3.6.0"
    assert info.id_key == "sid"
    assert info.index.name == settings.studies_index
    assert info.index.generation == "20260920T150000Z-abc12345"
    assert info.index.studies == 1198
    assert (info.index.embedding_model, info.index.embedding_dim) == ("microsoft/harrier-oss-v1-270m", 640)


def test_capabilities_reflect_the_modes_that_are_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only modes with an executor are advertised; the rest stay off until they are built."""
    with _running(monkeypatch, "opensearch", client=_opensearch()) as client:
        info = InfoResponse.model_validate(client.get("/info").json())
    assert info.capabilities.model_dump() == {
        "studies_search": True,
        "browse": True,
        "lexical": True,
        "semantic": True,
        "hybrid": True,
        "facets": False,
        "variables_search": False,
        "citations_search": False,
    }
    assert [spec.key for spec in info.filters][:2] == ["types", "countries"]
    assert info.limits is not None and info.limits.semantic_window == Settings().studies_semantic_window


def test_nothing_is_advertised_when_no_mode_is_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(info_module.IMPLEMENTED_STUDY_MODES, Engine.opensearch, frozenset())
    with _running(monkeypatch, "opensearch", client=_opensearch()) as client:
        info = InfoResponse.model_validate(client.get("/info").json())
    assert not info.capabilities.studies_search
    assert info.filters == [] and info.sort_fields == [] and info.limits is None


def test_the_registry_follows_the_executors() -> None:
    from nada_ai.search.backend.opensearch.studies_search import EXECUTORS

    assert info_module.IMPLEMENTED_STUDY_MODES[Engine.opensearch] == {mode.value for mode in EXECUTORS}


def test_full_capabilities_match_the_contract_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(info_module.IMPLEMENTED_STUDY_MODES, Engine.opensearch, ALL_MODES)
    with _running(monkeypatch, "opensearch", client=_opensearch()) as client:
        body = client.get("/info").json()
    fixture = _fixture("info_opensearch.json")
    assert set(body) == set(fixture)
    for key in ("capabilities", "id_key", "filters", "sort_fields", "limits"):
        assert body[key] == fixture[key], key
    assert set(body["index"]) == set(fixture["index"])


def test_a_missing_index_leaves_its_fields_empty_but_still_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    missing = (settings.studies_index, settings.index_name)
    with _running(monkeypatch, "opensearch", client=_opensearch(missing=missing)) as client:
        response = client.get("/info")
    assert response.status_code == 200
    index = InfoResponse.model_validate(response.json()).index
    assert (index.generation, index.studies, index.embedding_model, index.embedding_dim) == (None, None, None, None)
    assert index.name == settings.studies_index


def test_unreachable_opensearch_is_a_contract_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_opensearch(down=True)) as client:
        response = client.get("/info")
    assert response.status_code == 503
    assert ErrorResponse.model_validate(response.json()).error.code.value == "backend_unavailable"


# ---------------------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------------------


def test_qdrant_reports_study_search_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    with _running(monkeypatch, "qdrant", search=_qdrant()) as client:
        body = client.get("/info").json()
    fixture = _fixture("info_qdrant.json")
    assert body["capabilities"] == fixture["capabilities"]
    assert (body["filters"], body["sort_fields"], body["limits"]) == ([], [], None)
    assert (body["engine"], body["engine_version"]) == ("qdrant", "1.18.0")
    assert body["index"] == {
        "name": settings.qdrant_collection,
        "generation": None,
        "studies": None,
        "embedding_model": settings.embedding_model_id,
        "embedding_dim": 640,
    }


def test_qdrant_without_a_collection_has_no_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "qdrant", search=_qdrant(collection_exists=False)) as client:
        assert client.get("/info").json()["index"]["embedding_dim"] is None


def test_unreachable_qdrant_is_a_contract_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "qdrant", search=_qdrant(down=True)) as client:
        response = client.get("/info")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backend_unavailable"


# ---------------------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------------------


def test_authentication_errors_use_the_contract_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_opensearch()) as client:
        monkeypatch.setenv("NADA_ADMIN_API_KEY", "secret")
        missing = client.get("/info")
        wrong = client.get("/info", headers={"X-NADA-Admin-Key": "nope"})
        ok = client.get("/info", headers={"X-NADA-Admin-Key": "secret"})
    assert missing.status_code == wrong.status_code == 401
    assert ErrorResponse.model_validate(missing.json()).error.code.value == "unauthorized"
    assert ok.status_code == 200


def test_rate_limit_errors_use_the_contract_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, "opensearch", client=_opensearch(), search_rate_limiter=RateLimiter(1)) as client:
        first = client.get("/info")
        second = client.get("/info")
    assert first.status_code == 200
    assert second.status_code == 429
    assert ErrorResponse.model_validate(second.json()).error.code.value == "rate_limited"


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------


def test_unknown_engines_have_no_capabilities() -> None:
    assert not capabilities_for(Engine.solr, Settings()).studies_search


def test_modes_that_need_a_local_query_embedding_are_off_for_other_embedding_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NADA_OPENSEARCH_ML_MODEL_ID", "model")
    monkeypatch.setenv("NADA_OPENSEARCH_ML_EMBEDDING_DIMENSION", "384")
    settings = Settings(search_backend="opensearch", embedding_backend="opensearch_ml")
    capabilities = capabilities_for(Engine.opensearch, settings)
    assert (capabilities.browse, capabilities.lexical) == (True, True)
    assert (capabilities.semantic, capabilities.hybrid) == (False, False)
    assert capabilities_for(Engine.opensearch, Settings(embedding_backend="local")).hybrid


def test_limits_come_from_settings() -> None:
    limits = limits_for(Settings(studies_semantic_window=20))
    assert (limits.max_limit, limits.max_offset, limits.semantic_window, limits.max_query_length) == (
        100,
        10_000,
        20,
        500,
    )
