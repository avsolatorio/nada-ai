"""Admin / ingest / job endpoints.

CLI mirrors run as background jobs via :class:`nada_ai.app.jobs.JobRegistry` and
are single-flighted by ``key`` so re-submitting the same operation while it is
in flight returns the existing job (HTTP 409) rather than starting a duplicate.

Auth: write/admin endpoints require ``X-NADA-Admin-Key`` only when the
environment variable ``NADA_ADMIN_API_KEY`` is set. Locally unset = no auth.
``/jobs*`` endpoints are open (no admin auth) so progress can be polled.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from opensearchpy.exceptions import NotFoundError

from nada_ai.app.admin_schemas import (
    CreateIndexRequest,
    DeleteDocsResponse,
    EncodeRequest,
    EncodeResponse,
    GetFiltersResponse,
    IndexByIdsRequest,
    IndexFromCatalogRequest,
    IndexStatsResponse,
    JobListResponse,
    JobResponse,
    SyncFiltersRequest,
    SyncFiltersResponse,
)
from nada_ai.app.jobs import Job, JobStatus
from nada_ai.app.state import AppState, ensure_embedding_initialized, get_state
from nada_ai.ingest.service import (
    create_index_op,
    index_from_catalog_op,
    index_ids_op,
    put_index_template_op,
    setup_ingest_pipeline_op,
)
from nada_ai.filters.service import (
    ensure_filter_indexes_op_service,
    get_filters_op,
    sync_filters_op,
)
from nada_ai.search.backend.opensearch.mapping import metadata_field
from nada_ai.search.backend.opensearch.ml.setup import ingest_pipeline_definition

logger = logging.getLogger(__name__)


def _require_opensearch(s: AppState) -> None:
    if s.client is None:
        raise HTTPException(
            status_code=501,
            detail="This admin route requires OpenSearch. It is unavailable when NADA_SEARCH_BACKEND=qdrant.",
        )


admin_router = APIRouter(tags=["admin"])
jobs_router = APIRouter(tags=["jobs"])


ADMIN_API_KEY_ENV = "NADA_ADMIN_API_KEY"


def admin_auth(x_admin_key: str | None = Header(default=None, alias="X-NADA-Admin-Key")) -> None:
    expected = os.getenv(ADMIN_API_KEY_ENV)
    if expected and x_admin_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-NADA-Admin-Key")


def _job_to_response(job: Job) -> JobResponse:
    return JobResponse(**job.to_dict())


def _job_envelope(job: Job) -> dict[str, Any]:
    return job.to_dict()


def _idnos_key(idnos: list[str]) -> str:
    canonical = ",".join(sorted({i.strip() for i in idnos if i.strip()}))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


async def _submit_or_409(
    s: AppState,
    *,
    kind: str,
    key: str,
    factory,
    params: dict[str, Any],
) -> JSONResponse:
    job = await s.jobs.submit(kind=kind, key=key, factory=factory, params=params)
    payload = _job_envelope(job)
    if job.was_already_running:
        return JSONResponse(
            status_code=409,
            content={"detail": "a job with this key is already running", "job": payload},
        )
    return JSONResponse(status_code=202, content=payload)


@admin_router.post("/admin/index", dependencies=[Depends(admin_auth)])
async def admin_create_index(body: CreateIndexRequest, s: AppState = Depends(get_state)) -> JSONResponse:
    _require_opensearch(s)
    settings = s.settings
    recreate = body.recreate

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(create_index_op, settings, recreate)

    return await _submit_or_409(
        s,
        kind="create_index",
        key="create_index",
        factory=factory,
        params={"recreate": recreate},
    )


@admin_router.post("/admin/index/template", dependencies=[Depends(admin_auth)])
async def admin_put_index_template(s: AppState = Depends(get_state)) -> dict[str, Any]:
    """Install composable index template (knn_vector mapping) for ``index_name``; optional cluster auto-create."""
    _require_opensearch(s)
    return await asyncio.to_thread(put_index_template_op, s.settings)


@admin_router.post("/admin/setup-ingest-pipeline", dependencies=[Depends(admin_auth)])
async def admin_setup_ingest_pipeline(s: AppState = Depends(get_state)) -> JSONResponse:
    settings = s.settings

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(setup_ingest_pipeline_op, settings)

    return await _submit_or_409(
        s,
        kind="setup_ingest_pipeline",
        key="setup_ingest_pipeline",
        factory=factory,
        params={},
    )


@admin_router.post("/admin/ingest/by-ids", dependencies=[Depends(admin_auth)])
async def admin_ingest_by_ids(body: IndexByIdsRequest, s: AppState = Depends(get_state)) -> JSONResponse:
    settings = s.settings
    idnos = [i.strip() for i in body.idnos if i.strip()]
    if not idnos:
        raise HTTPException(status_code=400, detail="idnos must contain at least one non-empty value")
    metadata_type = body.metadata_type
    force = body.force
    recreate_index = body.recreate_index
    show_progress_bar = body.show_progress_bar
    buffer_size = body.buffer_size

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(
            index_ids_op,
            settings,
            idnos,
            metadata_type,
            force,
            recreate_index,
            show_progress_bar,
            buffer_size,
        )

    key = f"index:{metadata_type}:{_idnos_key(idnos)}"
    return await _submit_or_409(
        s,
        kind="index_by_ids",
        key=key,
        factory=factory,
        params={
            "idnos": idnos,
            "metadata_type": metadata_type,
            "force": force,
            "recreate_index": recreate_index,
            "buffer_size": buffer_size,
        },
    )


@admin_router.post("/admin/ingest/from-catalog", dependencies=[Depends(admin_auth)])
async def admin_ingest_from_catalog(
    body: IndexFromCatalogRequest, s: AppState = Depends(get_state)
) -> JSONResponse:
    _require_opensearch(s)
    settings = s.settings
    catalog_type = body.catalog_type
    ps = body.ps
    limit = body.limit
    force = body.force
    recreate_index = body.recreate_index
    show_progress_bar = body.show_progress_bar
    buffer_size = body.buffer_size

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(
            index_from_catalog_op,
            settings,
            catalog_type,
            ps,
            limit,
            force,
            recreate_index,
            show_progress_bar,
            buffer_size,
        )

    return await _submit_or_409(
        s,
        kind="index_from_catalog",
        key=f"index_from_catalog:{catalog_type}",
        factory=factory,
        params={
            "catalog_type": catalog_type,
            "ps": ps,
            "limit": limit,
            "force": force,
            "recreate_index": recreate_index,
            "buffer_size": buffer_size,
        },
    )


@admin_router.get("/admin/index/stats", dependencies=[Depends(admin_auth)], response_model=IndexStatsResponse)
async def admin_index_stats(s: AppState = Depends(get_state)) -> IndexStatsResponse:
    _require_opensearch(s)
    name = s.settings.index_name
    try:
        stats = await s.client.indices.stats(index=name)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=f"index {name} not found") from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    indices = stats.get("indices") or {}
    info = indices.get(name) or {}
    primaries = info.get("primaries") or {}
    docs = (primaries.get("docs") or {}).get("count")
    size = (primaries.get("store") or {}).get("size_in_bytes")
    return IndexStatsResponse(
        index=name,
        docs=int(docs) if docs is not None else None,
        size_bytes=int(size) if size is not None else None,
        primaries=primaries or None,
        raw=info or None,
    )


@admin_router.get("/admin/index/mapping", dependencies=[Depends(admin_auth)])
async def admin_index_mapping(s: AppState = Depends(get_state)) -> dict[str, Any]:
    _require_opensearch(s)
    name = s.settings.index_name
    try:
        return await s.client.indices.get_mapping(index=name)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=f"index {name} not found") from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@admin_router.post("/admin/index/refresh", dependencies=[Depends(admin_auth)])
async def admin_index_refresh(s: AppState = Depends(get_state)) -> dict[str, Any]:
    _require_opensearch(s)
    name = s.settings.index_name
    try:
        resp = await s.client.indices.refresh(index=name)
        return {"index": name, "refreshed": True, "raw": resp}
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=f"index {name} not found") from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@admin_router.delete("/admin/index", dependencies=[Depends(admin_auth)])
async def admin_index_delete(
    confirm: bool = Query(default=False, description="Must be true to actually drop the index."),
    s: AppState = Depends(get_state),
) -> dict[str, Any]:
    _require_opensearch(s)
    if not confirm:
        raise HTTPException(status_code=400, detail="add ?confirm=true to drop the index")
    name = s.settings.index_name
    try:
        resp = await s.client.indices.delete(index=name)
        return {"index": name, "deleted": True, "raw": resp}
    except NotFoundError:
        return {"index": name, "deleted": False, "detail": "index did not exist"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@admin_router.get("/admin/docs/{idno}", dependencies=[Depends(admin_auth)])
async def admin_doc_get(idno: str, s: AppState = Depends(get_state)) -> dict[str, Any]:
    _require_opensearch(s)
    name = s.settings.index_name
    body = {
        "size": 50,
        "query": {"term": {metadata_field("idno"): idno}},
    }
    try:
        resp = await s.client.search(index=name, body=body)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=f"index {name} not found") from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    hits = resp.get("hits", {}).get("hits", []) or []
    return {
        "index": name,
        "idno": idno,
        "count": len(hits),
        "hits": [
            {"_id": h.get("_id"), "_score": h.get("_score"), "_source": h.get("_source", {})} for h in hits
        ],
    }


@admin_router.delete(
    "/admin/docs/{idno}",
    dependencies=[Depends(admin_auth)],
    response_model=DeleteDocsResponse,
)
async def admin_doc_delete(idno: str, s: AppState = Depends(get_state)) -> DeleteDocsResponse:
    _require_opensearch(s)
    name = s.settings.index_name
    body = {"query": {"term": {metadata_field("idno"): idno}}}
    try:
        resp = await s.client.delete_by_query(index=name, body=body, refresh="true")
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=f"index {name} not found") from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return DeleteDocsResponse(
        index=name,
        deleted=int(resp.get("deleted") or 0),
        matched=int(resp.get("total")) if resp.get("total") is not None else None,
        raw=resp,
    )


@admin_router.post(
    "/admin/embeddings/encode",
    dependencies=[Depends(admin_auth)],
    response_model=EncodeResponse,
)
async def admin_embeddings_encode(body: EncodeRequest, s: AppState = Depends(get_state)) -> EncodeResponse:
    if s.settings.embedding_backend != "local":
        raise HTTPException(
            status_code=400,
            detail=f"embedding_backend is {s.settings.embedding_backend!r}; encode is only supported for 'local'",
        )
    try:
        await ensure_embedding_initialized(s)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"EmbeddingService initialization failed: {e}") from e
    if s.embedding is None:
        raise HTTPException(status_code=503, detail="EmbeddingService not initialized")

    if body.as_query:
        vectors = []
        for text in body.texts:
            v = await asyncio.to_thread(s.embedding.encode_query, text)
            vectors.append(v.tolist())
    else:
        arr = await asyncio.to_thread(s.embedding.encode_corpus, list(body.texts))
        vectors = [list(v) for v in arr.tolist()]

    return EncodeResponse(
        model_id=s.settings.embedding_model_id,
        dimension=s.embedding.embedding_dimension(),
        as_query=body.as_query,
        vectors=vectors,
    )


@admin_router.get("/admin/ml/pipeline", dependencies=[Depends(admin_auth)])
async def admin_ml_pipeline(s: AppState = Depends(get_state)) -> dict[str, Any]:
    _require_opensearch(s)
    name = s.settings.opensearch_ml_ingest_pipeline_name
    out: dict[str, Any] = {
        "embedding_backend": s.settings.embedding_backend,
        "pipeline_name": name,
    }
    try:
        defined_name, defined_body = ingest_pipeline_definition(s.settings)
        out["expected_definition"] = {"name": defined_name, "body": defined_body}
    except ValueError as e:
        out["expected_definition_error"] = str(e)

    try:
        resp = await s.client.ingest.get_pipeline(id=name)
        out["installed"] = resp
    except NotFoundError:
        out["installed"] = None
    except Exception as e:
        out["installed_error"] = str(e)
    return out


@admin_router.get("/admin/qdrant/collection", dependencies=[Depends(admin_auth)])
async def admin_qdrant_collection(s: AppState = Depends(get_state)) -> dict[str, Any]:
    """Collection metadata when ``NADA_SEARCH_BACKEND=qdrant`` (no OpenSearch client required)."""
    if s.settings.search_backend != "qdrant":
        raise HTTPException(
            status_code=400,
            detail="This route is only available when NADA_SEARCH_BACKEND=qdrant.",
        )
    client = getattr(s.search, "client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Qdrant search backend has no client")
    coll = s.settings.qdrant_collection
    try:
        info = await client.get_collection(collection_name=coll)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    payload = info.model_dump() if hasattr(info, "model_dump") else {"repr": repr(info)}
    return {"collection": coll, "info": payload}


@admin_router.post(
    "/admin/filters/sync",
    dependencies=[Depends(admin_auth)],
    response_model=SyncFiltersResponse,
)
async def admin_filters_sync(body: SyncFiltersRequest, s: AppState = Depends(get_state)) -> JSONResponse:
    records = [{"idno": r.idno.strip(), "filters": r.filters} for r in body.records if r.idno.strip()]
    if not records:
        raise HTTPException(status_code=400, detail="records must contain at least one non-empty idno")

    async def factory() -> dict[str, Any]:
        return await asyncio.to_thread(sync_filters_op, s.settings, records)

    key = f"filters_sync:{_idnos_key([r['idno'] for r in records])}"
    return await _submit_or_409(
        s,
        kind="filters_sync",
        key=key,
        factory=factory,
        params={"count": len(records)},
    )


@admin_router.post(
    "/admin/filters/ensure-indexes",
    dependencies=[Depends(admin_auth)],
)
async def admin_filters_ensure_indexes(s: AppState = Depends(get_state)) -> dict[str, Any]:
    return await asyncio.to_thread(ensure_filter_indexes_op_service, s.settings)


@admin_router.get(
    "/admin/filters/{idno}",
    dependencies=[Depends(admin_auth)],
    response_model=GetFiltersResponse,
)
async def admin_filters_get(idno: str, s: AppState = Depends(get_state)) -> GetFiltersResponse:
    out = await asyncio.to_thread(get_filters_op, s.settings, idno)
    return GetFiltersResponse(**out)


@jobs_router.get("/jobs", response_model=JobListResponse)
async def jobs_list(
    status: str | None = Query(default=None, description="Filter by status: pending|running|succeeded|failed|cancelled"),
    limit: int = Query(default=50, ge=1, le=500),
    s: AppState = Depends(get_state),
) -> JobListResponse:
    status_enum: JobStatus | None = None
    if status is not None:
        try:
            status_enum = JobStatus(status)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid status: {status}") from e
    jobs = s.jobs.list(status=status_enum, limit=limit)
    return JobListResponse(jobs=[_job_to_response(j) for j in jobs])


@jobs_router.get("/jobs/{job_id}", response_model=JobResponse)
async def job_get(job_id: str, s: AppState = Depends(get_state)) -> JobResponse:
    job = s.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return _job_to_response(job)


@jobs_router.delete("/jobs/{job_id}", response_model=JobResponse)
async def job_cancel(job_id: str, s: AppState = Depends(get_state)) -> JobResponse:
    job = await s.jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return _job_to_response(job)


__all__ = [
    "admin_auth",
    "admin_router",
    "ADMIN_API_KEY_ENV",
    "jobs_router",
    "_idnos_key",
    "_submit_or_409",
]
