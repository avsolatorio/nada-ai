"""Shared field-path and facet conventions for OpenSearch and Qdrant.

API filter keys stay logical (``type``, ``idno``, …). Stored paths prefix facet fields with
``metadata.``; root-only fields are ``page_content`` and ``embedding``.
"""

from __future__ import annotations

from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD, TEXT_FIELD, metadata_field
from nada_ai.search.backend.opensearch.queries import FACET_FIELD_WHITELIST

_ROOT_ONLY = frozenset({TEXT_FIELD, EMBEDDING_FIELD})


def facet_field_whitelist() -> frozenset[str]:
    """Facet-safe fields (same whitelist as OpenSearch ``terms`` aggs)."""
    return FACET_FIELD_WHITELIST


def stored_filter_field_name(field: str) -> str:
    """Return the path used in stored JSON / Qdrant payload for ``field``."""
    if field in _ROOT_ONLY:
        return field
    return metadata_field(field)
