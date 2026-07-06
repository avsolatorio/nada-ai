"""Catalog-centric management endpoints.

Per-idno index/reindex/delete and filter CRUD — designed for webhook-triggered
updates and programmatic catalog management. All write operations that may take
time are non-blocking (202 Accepted + job_id) so callers can return immediately
and poll /jobs/{job_id} for completion.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from nada_ai.app.admin import admin_auth, _idnos_key, _submit_or_409
from nada_ai.app.admin_schemas import (
    CatalogBatchIndexRequest,
    CatalogFiltersRequest,
    CatalogIndexRequest,
    CatalogStatusResponse,
    SyncFiltersRequest,
)
from nada_ai.app.state import AppState, get_state
from nada_ai.filters.service import get_filters_op, sync_filter_for_idno_op, sync_filters_op
from nada_ai.ingest.service import delete_by_idno_op, index_ids_op

catalog_router = APIRouter(prefix="/admin/catalog", tags=["catalog"])


@catalog_router.get("/{idno}/status", dependencies=[Depends(admin_auth)], response_model=CatalogStatusResponse)
async def catalog_idno_status(idno: str, s: AppState = Depends(get_state)) -> CatalogStatusResponse:
    """Get indexed status and filter fields for a single idno."""
    out = await asyncio.to_thread(get_filters_op, s.settings, idno)
    return CatalogStatusResponse(
        idno=idno,
        backend=s.settings.search_backend,
        indexed=bool(out.get("found")),
        doc_count=int(out.get("point_count") or 0),
        filter_fields=out.get("filter_fields"),
    )


@catalog_router.post("/{idno}/index", dependencies=[Depends(admin_auth)])
async def catalog_idno_index(
    idno: str, body: CatalogIndexRequest, s: AppState = Depends(get_state)
) -> JSONResponse:
    """Queue a non-blocking job to index a single idno."""
    settings = s.settings
    metadata_type = body.metadata_type
    force = body.force

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(index_ids_op, settings, [idno], metadata_type, force)

    return await _submit_or_409(
        s,
        kind="index_by_ids",
        key=f"index:{metadata_type}:{idno}",
        factory=factory,
        params={"idno": idno, "metadata_type": metadata_type, "force": force},
    )


@catalog_router.post("/{idno}/reindex", dependencies=[Depends(admin_auth)])
async def catalog_idno_reindex(
    idno: str, body: CatalogIndexRequest, s: AppState = Depends(get_state)
) -> JSONResponse:
    """Queue a non-blocking job to delete and re-index a single idno."""
    settings = s.settings
    metadata_type = body.metadata_type
    force = body.force

    async def factory() -> dict[str, Any]:
        delete_result = await asyncio.to_thread(delete_by_idno_op, settings, idno)
        index_result = await asyncio.to_thread(index_ids_op, settings, [idno], metadata_type, force)
        return {"delete": delete_result, "index": index_result}

    return await _submit_or_409(
        s,
        kind="reindex",
        key=f"reindex:{metadata_type}:{idno}",
        factory=factory,
        params={"idno": idno, "metadata_type": metadata_type, "force": force},
    )


@catalog_router.delete("/{idno}", dependencies=[Depends(admin_auth)])
async def catalog_idno_delete(idno: str, s: AppState = Depends(get_state)) -> JSONResponse:
    """Delete all indexed documents for an idno (synchronous; typically fast)."""
    try:
        result = await asyncio.to_thread(delete_by_idno_op, s.settings, idno)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return JSONResponse(status_code=200, content=result)


@catalog_router.get("/{idno}/filters", dependencies=[Depends(admin_auth)])
async def catalog_idno_get_filters(idno: str, s: AppState = Depends(get_state)) -> dict[str, Any]:
    """Get filter_fields for an idno."""
    return await asyncio.to_thread(get_filters_op, s.settings, idno)


@catalog_router.put("/{idno}/filters", dependencies=[Depends(admin_auth)])
async def catalog_idno_put_filters(
    idno: str, body: CatalogFiltersRequest, s: AppState = Depends(get_state)
) -> JSONResponse:
    """Replace all filter_fields for an idno (non-blocking job)."""
    settings = s.settings
    filters = body.filters

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(sync_filter_for_idno_op, settings, idno, filters)

    return await _submit_or_409(
        s,
        kind="filters_sync",
        key=f"filters:{idno}",
        factory=factory,
        params={"idno": idno, "filter_count": len(filters)},
    )


@catalog_router.patch("/{idno}/filters", dependencies=[Depends(admin_auth)])
async def catalog_idno_patch_filters(
    idno: str, body: CatalogFiltersRequest, s: AppState = Depends(get_state)
) -> JSONResponse:
    """Merge new filter_fields into existing ones for an idno (non-blocking job)."""
    settings = s.settings
    patch = body.filters

    async def factory() -> dict[str, Any]:
        current = await asyncio.to_thread(get_filters_op, settings, idno)
        merged: dict[str, Any] = {}
        for f in (current.get("filter_fields") or []):
            k, v = f.get("key"), f.get("value")
            if k is not None:
                merged[k] = v
        merged.update(patch)
        return await asyncio.to_thread(sync_filter_for_idno_op, settings, idno, merged)

    return await _submit_or_409(
        s,
        kind="filters_sync",
        key=f"filters:{idno}",
        factory=factory,
        params={"idno": idno, "patch_count": len(patch)},
    )


@catalog_router.delete("/{idno}/filters", dependencies=[Depends(admin_auth)])
async def catalog_idno_delete_filters(idno: str, s: AppState = Depends(get_state)) -> JSONResponse:
    """Clear all filter_fields for an idno (non-blocking job)."""
    settings = s.settings

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(sync_filter_for_idno_op, settings, idno, {})

    return await _submit_or_409(
        s,
        kind="filters_sync",
        key=f"filters:{idno}",
        factory=factory,
        params={"idno": idno, "action": "clear"},
    )


@catalog_router.post("/index", dependencies=[Depends(admin_auth)])
async def catalog_batch_index(body: CatalogBatchIndexRequest, s: AppState = Depends(get_state)) -> JSONResponse:
    """Queue a non-blocking job to index a batch of idnos."""
    settings = s.settings
    idnos = [i.strip() for i in body.idnos if i.strip()]
    if not idnos:
        raise HTTPException(status_code=400, detail="idnos must contain at least one non-empty value")
    metadata_type = body.metadata_type
    force = body.force

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(index_ids_op, settings, idnos, metadata_type, force)

    return await _submit_or_409(
        s,
        kind="index_by_ids",
        key=f"index:{metadata_type}:{_idnos_key(idnos)}",
        factory=factory,
        params={"idnos": idnos, "metadata_type": metadata_type, "force": force},
    )


@catalog_router.post("/filters/sync", dependencies=[Depends(admin_auth)])
async def catalog_batch_filters(body: SyncFiltersRequest, s: AppState = Depends(get_state)) -> JSONResponse:
    """Batch sync filter_fields for multiple idnos (non-blocking job)."""
    records = [{"idno": r.idno.strip(), "filters": r.filters} for r in body.records if r.idno.strip()]
    if not records:
        raise HTTPException(status_code=400, detail="records must contain at least one non-empty idno")
    settings = s.settings

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(sync_filters_op, settings, records)

    return await _submit_or_409(
        s,
        kind="filters_sync",
        key=f"filters_sync:{_idnos_key([r['idno'] for r in records])}",
        factory=factory,
        params={"count": len(records)},
    )
