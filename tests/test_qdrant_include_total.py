"""``include_total`` on POST /search: callers that never read ``total`` skip the neighbor count."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from nada_ai.app.schemas import SearchRequest, SearchResponse
from nada_ai.search.backend.qdrant.search_backend import QdrantSearchBackend
from nada_ai.settings import Settings


def _backend() -> tuple[QdrantSearchBackend, MagicMock]:
    backend = QdrantSearchBackend(Settings(search_backend="qdrant"))
    client = MagicMock()
    client.count = AsyncMock(return_value=MagicMock(count=42))
    client.query_points = AsyncMock(return_value=MagicMock(points=[MagicMock(), MagicMock()]))
    backend._client = client
    return backend, client


def test_the_total_is_counted_by_default() -> None:
    backend, client = _backend()
    total, basis, capped = asyncio.run(backend._vector_total_for_response("c", [0.1], None, None))
    assert (total, basis, capped) == (42, "metadata_filters_only", False)
    client.count.assert_awaited_once()


def test_a_threshold_makes_the_default_total_a_neighbor_scan() -> None:
    backend, client = _backend()
    total, basis, _ = asyncio.run(backend._vector_total_for_response("c", [0.1], None, 0.3))
    assert (total, basis) == (2, "vector_similarity_above_threshold")
    client.query_points.assert_awaited()


def test_include_total_false_counts_nothing() -> None:
    backend, client = _backend()
    for threshold in (None, 0.3):
        result = asyncio.run(backend._vector_total_for_response("c", [0.1], None, threshold, include_total=False))
        assert result == (0, "not_computed", False)
    client.count.assert_not_awaited()
    client.query_points.assert_not_awaited()


def test_the_request_counts_by_default_and_the_response_may_omit_the_total() -> None:
    assert SearchRequest(query="x").include_total is True
    assert SearchRequest(query="x", include_total=False).include_total is False
    assert SearchResponse(total=None, hits=[]).total is None
