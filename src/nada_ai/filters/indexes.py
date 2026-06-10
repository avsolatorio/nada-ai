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


def _try_create_payload_index(client: QdrantClient, collection: str, name: str, schema: Any) -> None:
    try:
        client.create_payload_index(collection_name=collection, field_name=name, field_schema=schema, wait=True)
    except Exception as e:
        logger.debug("Payload index %s: %s", name, e)


def ensure_qdrant_filter_field_indexes(client: QdrantClient, collection: str) -> dict[str, str]:
    """Create keyword payload indexes for ``metadata.filter_fields[].{key,value}``."""
    keyword = qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD)
    _try_create_payload_index(client, collection, FILTER_FIELDS_KEY_INDEX, keyword)
    _try_create_payload_index(client, collection, FILTER_FIELDS_VALUE_INDEX, keyword)
    return {
        "key_index": FILTER_FIELDS_KEY_INDEX,
        "value_index": FILTER_FIELDS_VALUE_INDEX,
    }


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
