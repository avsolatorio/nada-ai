"""Catalog-centric management endpoints.

Per-study index/reindex/delete and filter CRUD, by NADA internal id (``sid``) — designed for webhook-triggered
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

from nada_ai.app._ingest import content_sync_job_key, guarded_ingest
from nada_ai.app.admin import Sid, _ids_key, _submit_or_409
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
from nada_ai.filters.service import get_filters_op, sync_filter_for_sid_op, sync_filters_op
from nada_ai.ingest.service import delete_by_sids_op, index_sids_op

logger = logging.getLogger(__name__)

catalog_router = APIRouter(prefix="/admin/catalog", tags=["catalog"])


@catalog_router.get(
    "/{sid}/status", dependencies=[Depends(require_role(Role.read))], response_model=CatalogStatusResponse
)
async def catalog_sid_status(sid: Sid, s: AppState = Depends(get_state)) -> CatalogStatusResponse:
    """Get indexed status and filter fields for a single study."""
    out = await asyncio.to_thread(get_filters_op, s.settings, sid)
    return CatalogStatusResponse(
        sid=sid,
        backend=s.settings.search_backend,
        indexed=bool(out.get("found")),
        doc_count=int(out.get("point_count") or 0),
        filter_fields=out.get("filter_fields"),
    )


@catalog_router.post("/{sid}/index")
async def catalog_sid_index(
    sid: Sid,
    body: CatalogIndexRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Queue a non-blocking job to index a single study (its type comes from its NADA record).

    The job waits for the ingest semaphore before loading the embedding model,
    so concurrent requests are serialised on compute rather than multiplying
    memory/GPU usage.
    """
    settings = s.settings
    force = body.force

    async def factory() -> dict[str, Any]:
        return await guarded_ingest(s, index_sids_op, settings, [sid], force)

    return await _submit_or_409(
        s,
        kind="index_by_ids",
        key=content_sync_job_key(sid),
        factory=factory,
        params={"sid": sid, "force": force},
        principal=principal,
    )


@catalog_router.post("/{sid}/reindex")
async def catalog_sid_reindex(
    sid: Sid,
    body: CatalogIndexRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Queue a non-blocking job to delete and re-index a single study."""
    settings = s.settings
    force = body.force

    async def factory() -> dict[str, Any]:
        # Delete is fast (no embedding) — do it outside the semaphore.
        delete_result = await asyncio.to_thread(delete_by_sids_op, settings, [sid])
        index_result = await guarded_ingest(s, index_sids_op, settings, [sid], force)
        return {"delete": delete_result, "index": index_result}

    return await _submit_or_409(
        s,
        kind="reindex",
        key=content_sync_job_key(sid),
        factory=factory,
        params={"sid": sid, "force": force},
        principal=principal,
    )


@catalog_router.delete("/{sid}")
async def catalog_sid_delete(
    sid: Sid,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Delete all indexed documents of a study (synchronous; typically fast)."""
    try:
        result = await asyncio.to_thread(delete_by_sids_op, s.settings, [sid])
    except Exception as e:
        logger.error("delete failed for study %s: %s", sid, e)
        await audit_log(s, principal, action="catalog.delete", target=str(sid), status="error", detail=str(e))
        raise HTTPException(status_code=503, detail="delete operation failed") from e
    await audit_log(s, principal, action="catalog.delete", target=str(sid), status="ok")
    return JSONResponse(status_code=200, content=result)


@catalog_router.get("/{sid}/filters", dependencies=[Depends(require_role(Role.read))])
async def catalog_sid_get_filters(sid: Sid, s: AppState = Depends(get_state)) -> dict[str, Any]:
    """Get the stored filters of a study."""
    return await asyncio.to_thread(get_filters_op, s.settings, sid)


@catalog_router.put("/{sid}/filters")
async def catalog_sid_put_filters(
    sid: Sid,
    body: CatalogFiltersRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Replace all filters of a study (non-blocking job)."""
    settings = s.settings
    filters = body.filters

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(sync_filter_for_sid_op, settings, sid, filters)

    return await _submit_or_409(
        s,
        kind="filters_sync",
        key=f"filters:{sid}",
        factory=factory,
        params={"sid": sid, "filter_count": len(filters)},
        principal=principal,
    )


@catalog_router.patch("/{sid}/filters")
async def catalog_sid_patch_filters(
    sid: Sid,
    body: CatalogFiltersRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Merge new filters into a study's existing ones (non-blocking job)."""
    settings = s.settings
    patch = body.filters

    async def factory() -> dict[str, Any]:
        current = await asyncio.to_thread(get_filters_op, settings, sid)
        merged: dict[str, Any] = {}
        for f in current.get("filter_fields") or []:
            k, v = f.get("key"), f.get("value")
            if k is not None:
                merged[k] = v
        merged.update(patch)
        return await asyncio.to_thread(sync_filter_for_sid_op, settings, sid, merged)

    return await _submit_or_409(
        s,
        kind="filters_sync",
        key=f"filters:{sid}",
        factory=factory,
        params={"sid": sid, "patch_count": len(patch)},
        principal=principal,
    )


@catalog_router.delete("/{sid}/filters")
async def catalog_sid_delete_filters(
    sid: Sid,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Clear all filters of a study (non-blocking job)."""
    settings = s.settings

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(sync_filter_for_sid_op, settings, sid, {})

    return await _submit_or_409(
        s,
        kind="filters_sync",
        key=f"filters:{sid}",
        factory=factory,
        params={"sid": sid, "action": "clear"},
        principal=principal,
    )


@catalog_router.post("/index")
async def catalog_batch_index(
    body: CatalogBatchIndexRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Queue a non-blocking job to index a batch of studies (by NADA internal id)."""
    settings = s.settings
    sids = list(dict.fromkeys(body.sids))
    force = body.force
    recreate_index = body.recreate_index
    show_progress_bar = body.show_progress_bar
    buffer_size = body.buffer_size

    async def factory() -> dict[str, Any]:
        return await guarded_ingest(
            s,
            index_sids_op,
            settings,
            sids,
            force,
            recreate_index,
            show_progress_bar,
            buffer_size,
        )

    return await _submit_or_409(
        s,
        kind="index_by_ids",
        key=f"index:{_ids_key(sids)}",
        factory=factory,
        params={
            "sids": sids,
            "force": force,
            "recreate_index": recreate_index,
            "buffer_size": buffer_size,
        },
        principal=principal,
    )


@catalog_router.post("/delete")
async def catalog_batch_delete(
    body: CatalogBatchDeleteRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Delete all indexed documents of a batch of studies in one call (synchronous; typically fast)."""
    settings = s.settings
    sids = list(dict.fromkeys(body.sids))
    try:
        result = await asyncio.to_thread(delete_by_sids_op, settings, sids)
    except Exception as e:
        logger.error("batch delete failed for %s studies: %s", len(sids), e)
        await audit_log(
            s, principal, action="catalog.batch_delete", target=_ids_key(sids), status="error", detail=str(e)
        )
        raise HTTPException(status_code=503, detail="batch delete operation failed") from e
    await audit_log(
        s, principal, action="catalog.batch_delete", target=_ids_key(sids), status="ok", detail=f"count={len(sids)}"
    )
    return JSONResponse(status_code=200, content=result)


@catalog_router.post("/filters/sync")
async def catalog_batch_filters(
    body: SyncFiltersRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Batch sync the filters of several studies (non-blocking job)."""
    records = [{"sid": r.sid, "filters": r.filters} for r in body.records]
    settings = s.settings

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(sync_filters_op, settings, records)

    return await _submit_or_409(
        s,
        kind="filters_sync",
        key=f"filters_sync:{_ids_key([r['sid'] for r in records])}",
        factory=factory,
        params={"count": len(records)},
        principal=principal,
    )
