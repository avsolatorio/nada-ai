"""Ensure Qdrant payload indexes and OpenSearch mappings for dynamic filter_fields."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from nada_ai.search.dynamic_filters import FILTER_FIELDS_PATH

logger = logging.getLogger(__name__)

FILTER_FIELDS_KEY_INDEX = f"{FILTER_FIELDS_PATH}[].key"
FILTER_FIELDS_VALUE_INDEX = f"{FILTER_FIELDS_PATH}[].value"

REQUIRED_QDRANT_FILTER_FIELD_INDEXES = (
    FILTER_FIELDS_KEY_INDEX,
    FILTER_FIELDS_VALUE_INDEX,
)


def qdrant_dynamic_facet_indexes_ready(client: QdrantClient, collection: str) -> bool:
    """Return True when both filter_fields facet indexes exist on the collection."""
    info = client.get_collection(collection_name=collection)
    schema = info.payload_schema or {}
    return all(name in schema for name in REQUIRED_QDRANT_FILTER_FIELD_INDEXES)


def ensure_qdrant_filter_field_indexes(
    client: QdrantClient,
    collection: str,
    *,
    strict: bool = False,
) -> dict[str, str]:
    """Create keyword payload indexes for ``metadata.filter_fields[].{key,value}``."""
    keyword = qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD)
    results: dict[str, str] = {}

    for name in REQUIRED_QDRANT_FILTER_FIELD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=name,
                field_schema=keyword,
                wait=True,
            )
            results[name] = "created"
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "already indexed" in msg:
                results[name] = "exists"
            else:
                results[name] = f"error: {e}"
                logger.warning("Failed to create Qdrant payload index %s: %s", name, e)
                if strict:
                    raise RuntimeError(f"Failed to create Qdrant payload index {name}: {e}") from e

    if not qdrant_dynamic_facet_indexes_ready(client, collection):
        detail = (
            f"Missing Qdrant payload indexes for dynamic facets on {collection!r}. "
            f"Expected {list(REQUIRED_QDRANT_FILTER_FIELD_INDEXES)}. Got {results}."
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
