"""Ensure Qdrant payload indexes and OpenSearch mappings for dynamic filters."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from nada_ai.search.dynamic_filters import dynamic_facet_qdrant_key, load_dynamic_facet_keys
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def _create_keyword_index(
    client: QdrantClient,
    collection: str,
    field_name: str,
    *,
    strict: bool = False,
) -> str:
    keyword = qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD)
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=keyword,
            wait=True,
        )
        return "created"
    except Exception as e:
        msg = str(e).lower()
        if "already exists" in msg or "already indexed" in msg:
            return "exists"
        logger.warning("Failed to create Qdrant payload index %s: %s", field_name, e)
        if strict:
            raise RuntimeError(f"Failed to create Qdrant payload index {field_name}: {e}") from e
        return f"error: {e}"


def qdrant_filter_facets_index_paths(settings: Settings | None = None) -> tuple[str, ...]:
    return tuple(dynamic_facet_qdrant_key(key) for key in sorted(load_dynamic_facet_keys(settings)))


def qdrant_dynamic_facet_indexes_ready(client: QdrantClient, collection: str, settings: Settings | None = None) -> bool:
    """Return True when ``metadata.filter_facets.<key>`` indexes exist for all facetable keys."""
    info = client.get_collection(collection_name=collection)
    schema = info.payload_schema or {}
    return all(path in schema for path in qdrant_filter_facets_index_paths(settings))


def ensure_qdrant_filter_field_indexes(
    client: QdrantClient,
    collection: str,
    *,
    strict: bool = False,
    settings: Settings | None = None,
) -> dict[str, str]:
    """Create keyword payload indexes on ``metadata.filter_facets.<key>`` for filter + facet use."""
    results: dict[str, str] = {}
    for path in qdrant_filter_facets_index_paths(settings):
        results[path] = _create_keyword_index(client, collection, path, strict=strict)

    if not qdrant_dynamic_facet_indexes_ready(client, collection, settings):
        detail = (
            f"Missing Qdrant payload indexes for filter_facets on {collection!r}. "
            f"Got {results}."
        )
        if strict:
            raise RuntimeError(detail)
        logger.warning(detail)

    return results


def ensure_opensearch_filter_fields_mapping(client: Any, index_name: str) -> dict[str, Any]:
    """Add nested ``metadata.filter_fields`` mapping to an existing OpenSearch index."""
    body = {
        "properties": {
            "metadata": {
                "properties": {
                    "filter_fields": {
                        "type": "nested",
                        "properties": {
                            "key": {"type": "keyword"},
                            "value": {"type": "keyword"},
                        },
                    }
                }
            }
        }
    }
    resp = client.indices.put_mapping(index=index_name, body=body)
    return {"index": index_name, "mapping": body, "raw": resp}
