"""Admin endpoints for managing the dynamic facets config.

The facets config is a JSON file (``config/dynamic_filter_facets.json`` by
default, overridable via ``NADA_DYNAMIC_FILTER_FACETS_PATH``) that controls
which dynamic filter keys are returned as searchable facets in search results.

Changes take effect immediately on the next search request — no server restart
required.

Auth: same ``X-NADA-Admin-Key`` header as all other admin routes.

Routes
------
GET    /admin/facets             — current key list + metadata
PUT    /admin/facets             — replace the full list
POST   /admin/facets             — add one or more keys (idempotent)
DELETE /admin/facets/{key}       — remove a single key
POST   /admin/facets/bulk-remove — remove a set of keys in one call
POST   /admin/facets/backfill    — trigger background job to backfill
                                   filter_facets on all Qdrant points
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from nada_ai.app.admin import admin_auth, _submit_or_409
from nada_ai.app.state import AppState, get_state
from nada_ai.filters.facets_service import (
    add_facet_keys,
    backfill_facets_op,
    get_facets_config,
    remove_facet_key,
    remove_facet_keys,
    set_facets_config,
)

facets_router = APIRouter(prefix="/admin/facets", tags=["facets"])


# ── schemas ──────────────────────────────────────────────────────────────────


class FacetsSetRequest(BaseModel):
    keys: list[str] = Field(..., min_length=0, description="Complete replacement list of facetable keys.")

    @field_validator("keys", mode="before")
    @classmethod
    def _dedupe_and_strip(cls, v: Any) -> list[str]:
        seen: dict[str, None] = {}
        for item in v:
            s = str(item).strip()
            if s:
                seen[s] = None
        return list(seen)


class FacetsAddRequest(BaseModel):
    keys: list[str] = Field(..., min_length=1, description="Keys to add to the facetable list.")

    @field_validator("keys", mode="before")
    @classmethod
    def _dedupe_and_strip(cls, v: Any) -> list[str]:
        seen: dict[str, None] = {}
        for item in v:
            s = str(item).strip()
            if s:
                seen[s] = None
        return list(seen)


class FacetsBulkRemoveRequest(BaseModel):
    keys: list[str] = Field(..., min_length=1, description="Keys to remove from the facetable list.")


# ── endpoints ─────────────────────────────────────────────────────────────────


@facets_router.get(
    "",
    dependencies=[Depends(admin_auth)],
    summary="Get current facetable keys",
    response_description="Config state: key list, count, source, path.",
)
async def facets_get(s: AppState = Depends(get_state)) -> dict[str, Any]:
    """Return the current facetable key list and config metadata.

    ``source`` is one of:
    - ``"file"`` — loaded from an existing config file
    - ``"file_pending"`` — path is configured but file hasn't been written yet
    - ``"default"`` — no config file; using the built-in defaults
    """
    return await asyncio.to_thread(get_facets_config, s.settings)


@facets_router.put(
    "",
    dependencies=[Depends(admin_auth)],
    summary="Replace the full facetable key list",
    response_description="New config state after replacement.",
)
async def facets_put(body: FacetsSetRequest, s: AppState = Depends(get_state)) -> dict[str, Any]:
    """Atomically replace all facetable keys with the supplied list.

    Pass an empty list to clear all dynamic facets.  A ``warning`` field is
    included in the response if any supplied key overlaps with the fixed filter
    key set (those keys are searchable via a separate static path and do not
    benefit from being listed here).
    """
    try:
        return await asyncio.to_thread(set_facets_config, s.settings, body.keys)
    except OSError as e:
        raise HTTPException(status_code=503, detail=f"Could not write facets config: {e}") from e


@facets_router.post(
    "",
    dependencies=[Depends(admin_auth)],
    summary="Add keys to the facetable list (idempotent)",
    response_description="Updated config state plus added / already_present lists.",
)
async def facets_add(body: FacetsAddRequest, s: AppState = Depends(get_state)) -> dict[str, Any]:
    """Add one or more keys to the facetable list.

    Idempotent — keys that are already present are reported in
    ``already_present`` but cause no error.
    """
    try:
        return await asyncio.to_thread(add_facet_keys, s.settings, body.keys)
    except OSError as e:
        raise HTTPException(status_code=503, detail=f"Could not write facets config: {e}") from e


@facets_router.delete(
    "/{key}",
    dependencies=[Depends(admin_auth)],
    summary="Remove a single facetable key",
    response_description="Updated config state. ``removed`` is true when the key existed.",
)
async def facets_delete_key(
    key: str = Path(..., description="The facet key to remove."),
    s: AppState = Depends(get_state),
) -> dict[str, Any]:
    """Remove a single key from the facetable list.

    Idempotent — deleting a key that is not present returns ``"removed": []``
    and ``"not_found": [key]`` without an error.
    """
    try:
        return await asyncio.to_thread(remove_facet_key, s.settings, key)
    except OSError as e:
        raise HTTPException(status_code=503, detail=f"Could not write facets config: {e}") from e


@facets_router.post(
    "/bulk-remove",
    dependencies=[Depends(admin_auth)],
    summary="Remove multiple facetable keys in one call",
    response_description="Updated config state plus removed / not_found lists.",
)
async def facets_bulk_remove(
    body: FacetsBulkRemoveRequest, s: AppState = Depends(get_state)
) -> dict[str, Any]:
    """Remove a set of keys from the facetable list in a single atomic write."""
    try:
        return await asyncio.to_thread(remove_facet_keys, s.settings, body.keys)
    except OSError as e:
        raise HTTPException(status_code=503, detail=f"Could not write facets config: {e}") from e


@facets_router.post(
    "/backfill",
    dependencies=[Depends(admin_auth)],
    summary="Backfill filter_facets from filter_fields (Qdrant only)",
    response_description="202 Accepted with job_id to poll, or immediate result if Qdrant is not in use.",
)
async def facets_backfill(s: AppState = Depends(get_state)) -> JSONResponse:
    """Populate ``metadata.filter_facets`` (flat indexed payload) from the existing
    ``metadata.filter_fields`` arrays on all Qdrant points.

    This is necessary when points were indexed before the ``filter_facets`` field
    was introduced, or after manually setting ``filter_fields`` via an older
    pipeline that did not write the flat map.

    For OpenSearch the request succeeds immediately with ``skipped: true`` —
    OpenSearch uses nested ``filter_fields`` queries directly and does not
    require the flat map.
    """
    if s.settings.search_backend != "qdrant":
        result = await asyncio.to_thread(backfill_facets_op, s.settings)
        return JSONResponse(status_code=200, content=result)

    settings = s.settings

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(backfill_facets_op, settings)

    return await _submit_or_409(
        s,
        kind="backfill_filter_facets",
        key="backfill_filter_facets",
        factory=factory,
        params={"backend": settings.search_backend, "collection": settings.qdrant_collection},
    )
