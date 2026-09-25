"""Unit tests for OpenSearch composable index template + cluster auto-create helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nada_ai.ingest.service import put_index_template_op
from nada_ai.search.backend.opensearch.index_template import (
    _normalize_auto_create_index,
    composable_index_template_body,
    composable_index_template_name,
    put_cluster_auto_create_index,
    put_composable_index_template,
    studies_index_template_body,
    studies_index_template_name,
)
from nada_ai.settings import Settings


def test_normalize_auto_create_index() -> None:
    assert _normalize_auto_create_index("false") is False
    assert _normalize_auto_create_index("FALSE") is False
    assert _normalize_auto_create_index(" true ") is True
    assert _normalize_auto_create_index("+nada-metadata*,-*") == "+nada-metadata*,-*"


def test_composable_index_template_name_sanitizes_slash() -> None:
    s = Settings(index_name="a/b")
    assert composable_index_template_name(s) == "nada-ai-a-b-template"


def test_composable_index_template_body_patterns_and_knn() -> None:
    s = Settings(index_name="nada-metadata", opensearch_index_template_priority=100)
    body = composable_index_template_body(s, 384)
    # exactly the chunk index: a `-*` pattern would also match (and mis-map) the study index
    assert body["index_patterns"] == ["nada-metadata"]
    assert body["priority"] == 100
    emb = body["template"]["mappings"]["properties"]["embedding"]
    assert emb["type"] == "knn_vector"
    assert emb["dimension"] == 384
    assert body["template"]["settings"]["index"]["knn"] is True


def test_studies_index_template_matches_only_the_study_index() -> None:
    s = Settings(index_name="nada-metadata")
    body = studies_index_template_body(s)
    assert body["index_patterns"] == ["nada-metadata-studies"]
    assert "embedding" not in body["template"]["mappings"]["properties"]
    assert studies_index_template_name(s) == "nada-ai-nada-metadata-studies-template"
    assert studies_index_template_name(s) != composable_index_template_name(s)


def test_studies_index_name_can_be_overridden() -> None:
    assert Settings(index_name="a").studies_index == "a-studies"
    assert Settings(index_name="a", studies_index_name="other").studies_index == "other"


def test_put_composable_index_template_installs_every_index_template() -> None:
    client = MagicMock()
    s = Settings(index_name="idx-one")
    out = put_composable_index_template(client, s, 256)
    assert set(out["templates"]) == {
        "nada-ai-idx-one-template",
        "nada-ai-idx-one-studies-template",
        "nada-ai-idx-one-variables-template",
        "nada-ai-idx-one-citations-template",
    }
    assert out["templates"]["nada-ai-idx-one-template"]["index_patterns"] == ["idx-one"]
    assert out["templates"]["nada-ai-idx-one-studies-template"]["index_patterns"] == ["idx-one-studies"]
    assert out["templates"]["nada-ai-idx-one-variables-template"]["index_patterns"] == ["idx-one-variables"]
    assert out["templates"]["nada-ai-idx-one-citations-template"]["index_patterns"] == ["idx-one-citations"]
    assert client.indices.put_index_template.call_count == 4
    names = {c.kwargs["name"] for c in client.indices.put_index_template.call_args_list}
    assert names == {
        "nada-ai-idx-one-template",
        "nada-ai-idx-one-studies-template",
        "nada-ai-idx-one-variables-template",
        "nada-ai-idx-one-citations-template",
    }


def test_put_cluster_auto_create_index() -> None:
    client = MagicMock()
    client.cluster.put_settings.return_value = {"acknowledged": True}
    out = put_cluster_auto_create_index(client, " false ")
    assert out["acknowledged"] is True
    assert out["action.auto_create_index"] is False
    client.cluster.put_settings.assert_called_once_with(body={"persistent": {"action.auto_create_index": False}})


def test_put_index_template_op_skips_when_qdrant() -> None:
    s = Settings(search_backend="qdrant")
    out = put_index_template_op(s)
    assert out.get("skipped") is True


@pytest.mark.parametrize(
    "flag,expect_put",
    [
        (True, True),
        (False, False),
    ],
)
def test_put_index_template_op_respects_template_flag(
    monkeypatch: pytest.MonkeyPatch, flag: bool, expect_put: bool
) -> None:
    s = Settings(
        search_backend="opensearch",
        embedding_backend="opensearch_ml",
        opensearch_ml_model_id="m",
        opensearch_ml_embedding_dimension=512,
        opensearch_put_composable_index_template=flag,
        opensearch_cluster_auto_create_index=None,
    )
    client = MagicMock()
    monkeypatch.setattr("nada_ai.ingest.service.build_client", lambda _settings: client)
    out = put_index_template_op(s)
    assert out["dim"] == 512
    if expect_put:
        assert client.indices.put_index_template.call_count == 4
        assert "template" in out and "skipped" not in out.get("template", {})
    else:
        client.indices.put_index_template.assert_not_called()
        assert out["template"]["skipped"] is True


def test_put_index_template_op_loads_no_model_when_embeddings_are_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """embedding_backend=none must never instantiate EmbeddingService (which eagerly loads the real model) just
    to compute a dimension nothing will use."""
    s = Settings(search_backend="opensearch", embedding_backend="none")
    client = MagicMock()
    monkeypatch.setattr("nada_ai.ingest.service.build_client", lambda _settings: client)

    def _boom(_settings):
        raise AssertionError("EmbeddingService must not be instantiated when embedding_backend=none")

    monkeypatch.setattr("nada_ai.ingest.service.EmbeddingService", _boom)

    out = put_index_template_op(s)
    assert out["dim"] is None
