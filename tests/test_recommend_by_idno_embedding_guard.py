"""``recommend_by_idno`` needs a real stored vector in ``_source``; two embedding backends never have one."""

from __future__ import annotations

import asyncio

import pytest

from nada_ai.search.backend.opensearch.search_backend import OpenSearchSearchBackend
from nada_ai.search.ports import RecommendParams
from nada_ai.settings import Settings


def _params(idno: str) -> RecommendParams:
    return RecommendParams(
        idno=idno,
        size=10,
        filters=None,
        exclude_idno=True,
        vector_strategy="mean",
        knn_k=50,
        include_facets=False,
        facet_fields=None,
    )


def test_raises_when_embeddings_are_disabled() -> None:
    settings = Settings(search_backend="opensearch", embedding_backend="none")
    backend = OpenSearchSearchBackend(client=None, settings=settings)
    with pytest.raises(ValueError, match="embedding_backend=none"):
        asyncio.run(backend.recommend_by_idno(_params("WB_LSMS_001")))


def test_raises_for_opensearch_ml_too() -> None:
    settings = Settings(
        search_backend="opensearch",
        embedding_backend="opensearch_ml",
        opensearch_ml_model_id="m",
        opensearch_ml_embedding_dimension=384,
    )
    backend = OpenSearchSearchBackend(client=None, settings=settings)
    with pytest.raises(ValueError, match="opensearch_ml"):
        asyncio.run(backend.recommend_by_idno(_params("WB_LSMS_001")))
