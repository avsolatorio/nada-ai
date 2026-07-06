"""Facets config management service.

Provides read/write operations on the ``dynamic_filter_facets.json`` config
that governs which dynamic filter keys are exposed as searchable facets in
search results.

All write helpers follow a load → mutate → atomic-save pattern and return a
uniform ``FacetsConfigState`` dict so API callers always see the full picture
after an operation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nada_ai.search.dynamic_filters import (
    FIXED_FILTER_KEYS,
    _DEFAULT_FACETABLE,
    _resolve_facets_path,
    load_dynamic_facet_keys,
    save_dynamic_facet_keys,
)
from nada_ai.settings import Settings


def _config_state(settings: Settings, keys: frozenset[str]) -> dict[str, Any]:
    """Build the canonical response dict describing current facets config."""
    path, from_file = _resolve_facets_path(settings)
    source: str
    if from_file and path.is_file():
        source = "file"
    elif from_file:
        # Path is configured but file doesn't exist yet — will be created on first write.
        source = "file_pending"
    else:
        source = "default"
    return {
        "keys": sorted(keys),
        "count": len(keys),
        "source": source,
        "path": str(path),
        "writable": True,
        "fixed_filter_keys": sorted(FIXED_FILTER_KEYS),
    }


def get_facets_config(settings: Settings) -> dict[str, Any]:
    """Return the current facets config state."""
    keys = load_dynamic_facet_keys(settings)
    return _config_state(settings, keys)


def set_facets_config(settings: Settings, keys: list[str]) -> dict[str, Any]:
    """Replace the full facetable key list.  Returns the new config state."""
    normalized = frozenset(str(k).strip() for k in keys if str(k).strip())
    overlapping = sorted(normalized & FIXED_FILTER_KEYS)
    path = save_dynamic_facet_keys(normalized, settings)
    state = _config_state(settings, normalized)
    state["path"] = str(path)
    if overlapping:
        state["warning"] = (
            f"Keys {overlapping} overlap with fixed filter keys and will be ignored during "
            "dynamic facet aggregation — they are handled via the static metadata path."
        )
    return state


def add_facet_keys(settings: Settings, new_keys: list[str]) -> dict[str, Any]:
    """Add one or more keys to the facetable list.

    Idempotent: already-present keys are reported but not duplicated.
    Returns the new config state plus ``added`` / ``already_present`` lists.
    """
    current = load_dynamic_facet_keys(settings)
    incoming = frozenset(str(k).strip() for k in new_keys if str(k).strip())
    added = sorted(incoming - current)
    already_present = sorted(incoming & current)
    merged = current | incoming
    save_dynamic_facet_keys(merged, settings)
    state = _config_state(settings, merged)
    state["added"] = added
    state["already_present"] = already_present
    overlapping = sorted(incoming & FIXED_FILTER_KEYS)
    if overlapping:
        state["warning"] = (
            f"Keys {overlapping} overlap with fixed filter keys and will be ignored during "
            "dynamic facet aggregation — they are handled via the static metadata path."
        )
    return state


def remove_facet_keys(settings: Settings, keys_to_remove: list[str]) -> dict[str, Any]:
    """Remove one or more keys from the facetable list.

    Returns the new config state plus ``removed`` / ``not_found`` lists.
    """
    current = load_dynamic_facet_keys(settings)
    targets = frozenset(str(k).strip() for k in keys_to_remove if str(k).strip())
    removed = sorted(targets & current)
    not_found = sorted(targets - current)
    remaining = current - targets
    save_dynamic_facet_keys(remaining, settings)
    state = _config_state(settings, remaining)
    state["removed"] = removed
    state["not_found"] = not_found
    return state


def remove_facet_key(settings: Settings, key: str) -> dict[str, Any]:
    """Remove a single key from the facetable list."""
    return remove_facet_keys(settings, [key])


def backfill_facets_op(settings: Settings, *, show_progress_bar: bool = False) -> dict[str, Any]:
    """Backfill ``metadata.filter_facets`` from existing ``filter_fields`` on all indexed points.

    Qdrant only — OpenSearch uses nested ``filter_fields`` queries directly and
    does not store the flat ``filter_facets`` map.

    Returns a summary dict suitable for a job result.
    """
    if settings.search_backend != "qdrant":
        return {
            "skipped": True,
            "detail": "backfill_filter_facets applies to the qdrant backend only.",
            "backend": settings.search_backend,
        }
    from nada_ai.filters.sync import backfill_filter_facets

    result = backfill_filter_facets(settings, show_progress_bar=show_progress_bar)
    return {
        "backend": settings.search_backend,
        "collection": settings.qdrant_collection,
        **result,
    }
