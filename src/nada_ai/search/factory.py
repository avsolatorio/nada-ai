"""Factory for search backends (OpenSearch, Qdrant)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nada_ai.search.ports import SearchBackendPort
from nada_ai.settings import Settings

if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


def create_search_backend(settings: Settings, opensearch_client: AsyncOpenSearch | None) -> SearchBackendPort:
    if settings.search_backend == "opensearch":
        if opensearch_client is None:
            raise ValueError("OpenSearch client is required when search_backend=opensearch")
        from nada_ai.search.backend.opensearch.search_backend import OpenSearchSearchBackend

        return OpenSearchSearchBackend(opensearch_client, settings)
    from nada_ai.search.backend.qdrant.search_backend import QdrantSearchBackend

    return QdrantSearchBackend(settings)
