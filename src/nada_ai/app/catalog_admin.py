"""Catalog-centric management endpoints.

Per-idno index/reindex/delete and filter CRUD — designed for webhook-triggered
updates and programmatic catalog management. All write operations that may take
time are non-blocking (202 Accepted + job_id) so callers can return immediately
and poll /jobs/{job_id} for completion.

Ingest jobs go through :func:`~nada_ai.app._ingest.guarded_ingest` which:
- warms the shared EmbeddingService once (no duplicate model loads)
- acquires the ingest semaphore (bounded by ``settings.max_concurrent_ingest_jobs``)
before dispatching compute to a thread.

Auth: see ``nada_ai.app.auth`` — read-only routes require role ``read``,
mutating routes require role ``write``. Mutating routes write an audit entry.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from nada_ai.app._ingest import guarded_ingest
from nada_ai.app.admin import _idnos_key, _submit_or_409
from nada_ai.app.admin_schemas import (
    CatalogBatchDeleteRequest,
    CatalogBatchIndexRequest,
    CatalogFiltersRequest,
    CatalogIndexRequest,
    CatalogStatusResponse,
    SyncFiltersRequest,
)
from nada_ai.app.audit import audit_log
from nada_ai.app.auth import Principal, require_role
from nada_ai.app.keys_store import Role
from nada_ai.app.state import AppState, get_state
from nada_ai.filters.service import get_filters_op, sync_filter_for_idno_op, sync_filters_op
from nada_ai.ingest.service import delete_by_idno_op, delete_by_idnos_op, index_ids_op

logger = logging.getLogger(__name__)

catalog_router = APIRouter(prefix="/admin/catalog", tags=["catalog"])


@catalog_router.get(
    "/{idno}/status", dependencies=[Depends(require_role(Role.read))], response_model=CatalogStatusResponse
)
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


@catalog_router.post("/{idno}/index")
async def catalog_idno_index(
    idno: str,
    body: CatalogIndexRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Queue a non-blocking job to index a single idno.

    The job waits for the ingest semaphore before loading the embedding model,
    so concurrent requests are serialised on compute rather than multiplying
    memory/GPU usage.
    """
    settings = s.settings
    metadata_type = body.metadata_type
    force = body.force

    async def factory() -> dict[str, Any]:
        return await guarded_ingest(s, index_ids_op, settings, [idno], metadata_type, force)

    return await _submit_or_409(
        s,
        kind="index_by_ids",
        key=f"index:{metadata_type}:{idno}",
        factory=factory,
        params={"idno": idno, "metadata_type": metadata_type, "force": force},
        principal=principal,
    )


@catalog_router.post("/{idno}/reindex")
async def catalog_idno_reindex(
    idno: str,
    body: CatalogIndexRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Queue a non-blocking job to delete and re-index a single idno."""
    settings = s.settings
    metadata_type = body.metadata_type
    force = body.force

    async def factory() -> dict[str, Any]:
        # Delete is fast (no embedding) — do it outside the semaphore.
        delete_result = await asyncio.to_thread(delete_by_idno_op, settings, idno)
        index_result = await guarded_ingest(s, index_ids_op, settings, [idno], metadata_type, force)
        return {"delete": delete_result, "index": index_result}

    return await _submit_or_409(
        s,
        kind="reindex",
        key=f"reindex:{metadata_type}:{idno}",
        factory=factory,
        params={"idno": idno, "metadata_type": metadata_type, "force": force},
        principal=principal,
    )


@catalog_router.delete("/{idno}")
async def catalog_idno_delete(
    idno: str,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Delete all indexed documents for an idno (synchronous; typically fast)."""
    try:
        result = await asyncio.to_thread(delete_by_idno_op, s.settings, idno)
    except Exception as e:
        logger.error("delete failed for %s: %s", idno, e)
        await audit_log(s, principal, action="catalog.delete", target=idno, status="error", detail=str(e))
        raise HTTPException(status_code=503, detail="delete operation failed") from e
    await audit_log(s, principal, action="catalog.delete", target=idno, status="ok")
    return JSONResponse(status_code=200, content=result)


@catalog_router.get("/{idno}/filters", dependencies=[Depends(require_role(Role.read))])
async def catalog_idno_get_filters(idno: str, s: AppState = Depends(get_state)) -> dict[str, Any]:
    """Get filter_fields for an idno."""
    return await asyncio.to_thread(get_filters_op, s.settings, idno)


@catalog_router.put("/{idno}/filters")
async def catalog_idno_put_filters(
    idno: str,
    body: CatalogFiltersRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
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
        principal=principal,
    )


@catalog_router.patch("/{idno}/filters")
async def catalog_idno_patch_filters(
    idno: str,
    body: CatalogFiltersRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
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
        principal=principal,
    )


@catalog_router.delete("/{idno}/filters")
async def catalog_idno_delete_filters(
    idno: str,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
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
        principal=principal,
    )


@catalog_router.post("/index")
async def catalog_batch_index(
    body: CatalogBatchIndexRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Queue a non-blocking job to index a batch of idnos."""
    settings = s.settings
    idnos = [i.strip() for i in body.idnos if i.strip()]
    if not idnos:
        raise HTTPException(status_code=400, detail="idnos must contain at least one non-empty value")
    metadata_type = body.metadata_type
    force = body.force

    async def factory() -> dict[str, Any]:
        return await guarded_ingest(s, index_ids_op, settings, idnos, metadata_type, force)

    return await _submit_or_409(
        s,
        kind="index_by_ids",
        key=f"index:{metadata_type}:{_idnos_key(idnos)}",
        factory=factory,
        params={"idnos": idnos, "metadata_type": metadata_type, "force": force},
        principal=principal,
    )


@catalog_router.post("/delete")
async def catalog_batch_delete(
    body: CatalogBatchDeleteRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Delete all indexed documents for a batch of idnos in one call (synchronous; typically fast)."""
    settings = s.settings
    idnos = [i.strip() for i in body.idnos if i.strip()]
    if not idnos:
        raise HTTPException(status_code=400, detail="idnos must contain at least one non-empty value")
    try:
        result = await asyncio.to_thread(delete_by_idnos_op, settings, idnos)
    except Exception as e:
        logger.error("batch delete failed for %s idnos: %s", len(idnos), e)
        await audit_log(
            s, principal, action="catalog.batch_delete", target=_idnos_key(idnos), status="error", detail=str(e)
        )
        raise HTTPException(status_code=503, detail="batch delete operation failed") from e
    await audit_log(
        s, principal, action="catalog.batch_delete", target=_idnos_key(idnos), status="ok", detail=f"count={len(idnos)}"
    )
    return JSONResponse(status_code=200, content=result)


@catalog_router.post("/filters/sync")
async def catalog_batch_filters(
    body: SyncFiltersRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
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
        principal=principal,
    )
