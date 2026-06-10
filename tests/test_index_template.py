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
    assert body["index_patterns"] == ["nada-metadata", "nada-metadata-*"]
    assert body["priority"] == 100
    emb = body["template"]["mappings"]["properties"]["embedding"]
    assert emb["type"] == "knn_vector"
    assert emb["dimension"] == 384
    assert body["template"]["settings"]["index"]["knn"] is True


def test_put_composable_index_template_calls_client() -> None:
    client = MagicMock()
    s = Settings(index_name="idx-one")
    out = put_composable_index_template(client, s, 256)
    assert out["template"] == "nada-ai-idx-one-template"
    assert out["index_patterns"] == ["idx-one", "idx-one-*"]
    client.indices.put_index_template.assert_called_once()
    call_kw = client.indices.put_index_template.call_args.kwargs
    assert call_kw["name"] == "nada-ai-idx-one-template"
    assert call_kw["body"]["index_patterns"] == ["idx-one", "idx-one-*"]


def test_put_cluster_auto_create_index() -> None:
    client = MagicMock()
    client.cluster.put_settings.return_value = {"acknowledged": True}
    out = put_cluster_auto_create_index(client, " false ")
    assert out["acknowledged"] is True
    assert out["action.auto_create_index"] is False
    client.cluster.put_settings.assert_called_once_with(
        body={"persistent": {"action.auto_create_index": False}}
    )


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
def test_put_index_template_op_respects_template_flag(monkeypatch: pytest.MonkeyPatch, flag: bool, expect_put: bool) -> None:
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
        client.indices.put_index_template.assert_called_once()
        assert "template" in out and "skipped" not in out.get("template", {})
    else:
        client.indices.put_index_template.assert_not_called()
        assert out["template"]["skipped"] is True
