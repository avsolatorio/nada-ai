"""Tests for the embedding-dimension mismatch guards added to both ingest
writers — without these, changing NADA_EMBEDDING_MODEL_ID and ingesting
against an existing index/collection (without recreate) used to fail deep
inside a per-point/per-doc bulk write instead of failing fast up front."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nada_ai.ingest.pipeline import _assert_dense_dim_matches as os_assert_dim
from nada_ai.ingest.pipeline import ensure_index
from nada_ai.ingest.qdrant_writer import _assert_dense_dim_matches as qdrant_assert_dim


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def _qdrant_client_with_dim(size: int) -> MagicMock:
    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=size)))
    )
    return client


def test_qdrant_dim_guard_passes_when_matching():
    client = _qdrant_client_with_dim(384)
    qdrant_assert_dim(client, "coll", 384, "some-model")  # must not raise


def test_qdrant_dim_guard_raises_on_mismatch():
    client = _qdrant_client_with_dim(384)
    with pytest.raises(ValueError, match="384.*512|512.*384"):
        qdrant_assert_dim(client, "coll", 512, "new-model")


def test_qdrant_dim_guard_handles_named_vector_dict():
    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors={"default": SimpleNamespace(size=768)})
        )
    )
    with pytest.raises(ValueError):
        qdrant_assert_dim(client, "coll", 384, "some-model")


# ---------------------------------------------------------------------------
# OpenSearch
# ---------------------------------------------------------------------------

def _os_client_with_dim(dim: int) -> MagicMock:
    client = MagicMock()
    client.indices.get_mapping.return_value = {
        "nada-metadata": {"mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": dim}}}}
    }
    return client


def test_opensearch_dim_guard_passes_when_matching():
    client = _os_client_with_dim(384)
    os_assert_dim(client, "nada-metadata", 384, "some-model")  # must not raise


def test_opensearch_dim_guard_raises_on_mismatch():
    client = _os_client_with_dim(384)
    with pytest.raises(ValueError, match="384.*512|512.*384"):
        os_assert_dim(client, "nada-metadata", 512, "new-model")


def test_ensure_index_checks_dim_when_index_already_exists():
    client = _os_client_with_dim(384)
    client.indices.exists.return_value = True
    settings = SimpleNamespace(index_name="nada-metadata", embedding_model_id="new-model")

    with pytest.raises(ValueError):
        ensure_index(client, settings, 512)
    client.indices.create.assert_not_called()


def test_ensure_index_creates_when_missing_no_dim_check():
    client = MagicMock()
    client.indices.exists.return_value = False
    settings = SimpleNamespace(index_name="nada-metadata", embedding_model_id="some-model")

    ensure_index(client, settings, 384)

    client.indices.get_mapping.assert_not_called()
    client.indices.create.assert_called_once()
