"""Reusable filter sync operations for CLI and admin API."""

from __future__ import annotations

from typing import Any

from nada_ai.filters.sync import (
    backfill_filter_facets,
    ensure_filter_indexes_op,
    get_filter_fields_for_idno,
    sync_filters_batch,
    sync_filters_for_idno,
)
from nada_ai.settings import Settings


def sync_filters_op(
    settings: Settings,
    records: list[dict[str, Any]],
    *,
    show_progress_bar: bool = False,
) -> dict[str, Any]:
    results = sync_filters_batch(settings, records, show_progress_bar=show_progress_bar)
    return {
        "backend": settings.search_backend,
        "synced": len(results),
        "results": results,
    }


def sync_filter_for_idno_op(settings: Settings, idno: str, filters: dict[str, Any]) -> dict[str, Any]:
    result = sync_filters_for_idno(settings, idno, filters)
    return {"backend": settings.search_backend, **result}


def get_filters_op(settings: Settings, idno: str) -> dict[str, Any]:
    out = get_filter_fields_for_idno(settings, idno)
    out["backend"] = settings.search_backend
    return out


def ensure_filter_indexes_op_service(settings: Settings) -> dict[str, Any]:
    return ensure_filter_indexes_op(settings)


def backfill_filter_facets_op(settings: Settings, *, show_progress_bar: bool = False) -> dict[str, Any]:
    result = backfill_filter_facets(settings, show_progress_bar=show_progress_bar)
    return {"backend": settings.search_backend, **result}
