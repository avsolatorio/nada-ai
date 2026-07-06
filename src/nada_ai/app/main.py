from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from opensearchpy.exceptions import NotFoundError, RequestError

from nada_ai import __version__
from nada_ai.app.admin import admin_router, jobs_router
from nada_ai.app.catalog_admin import catalog_router
from nada_ai.app.facets_admin import facets_router
from nada_ai.app.webhooks import webhooks_router
from nada_ai.app.demo_preview import render_pdf_page_png, resolve_document_pdf_path
from nada_ai.app.jobs import JobRegistry
from nada_ai.app.schemas import (
    ExplainSearchRequest,
    RecommendRequest,
    SearchRequest,
    SearchResponse,
    coerce_search_facets,
)
from nada_ai.app.state import AppState, ensure_embedding_initialized, get_state, state
from nada_ai.mcp_server import mcp
from nada_ai.search.backend.opensearch.client import build_async_client
from nada_ai.search.dynamic_filters import load_dynamic_facet_keys
from nada_ai.search.factory import create_search_backend
from nada_ai.search.ports import RecommendParams, SearchParams
from nada_ai.search.query_heuristics import looks_like_catalog_idno
from nada_ai.settings import Settings

# NOTE: import to be able to run the server with all definitions loaded
# path="/mcp" means the MCP endpoint lives at /mcp (no trailing slash needed)
mcp_app = mcp.http_app(path="/mcp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.settings = Settings()
    if state.settings.search_backend == "opensearch":
        state.client = build_async_client(state.settings)
    else:
        state.client = None
    state.search = create_search_backend(state.settings, state.client)
    state.embedding = None
    state.embedding_init_lock = asyncio.Lock()
    state.embedding_init_error = None
    state.jobs = JobRegistry()
    state.facets_config_lock = asyncio.Lock()
    state.ingest_semaphore = asyncio.Semaphore(state.settings.max_concurrent_ingest_jobs)

    async with mcp_app.router.lifespan_context(mcp_app):
        yield
    try:
        await state.jobs.shutdown()
    except Exception:
        pass
    try:
        if state.client is not None:
            await state.client.close()
    except Exception:
        pass
    aclose = getattr(state.search, "aclose", None)
    if callable(aclose):
        try:
            await aclose()
        except Exception:
            pass


app = FastAPI(title="NADA AI Search", version=__version__, lifespan=lifespan)
app.include_router(admin_router)
app.include_router(jobs_router)
app.include_router(catalog_router)
app.include_router(facets_router)
app.include_router(webhooks_router)

_STATIC = Path(__file__).resolve().parent / "static"


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo_search_ui(s: AppState = Depends(get_state)) -> HTMLResponse:
    """Minimal browser UI that calls ``POST /search`` (same origin)."""
    path = _STATIC / "demo.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="demo.html not found")
    html = path.read_text(encoding="utf-8")
    facet_keys = sorted(load_dynamic_facet_keys(s.settings))
    config_snippet = f'<script id="facet-keys-config" type="application/json">{json.dumps(facet_keys)}</script>'
    html = html.replace("<!-- FACET_KEYS_CONFIG -->", config_snippet)
    return HTMLResponse(html)


@app.get(
    "/demo/documents/{idno}/pages/{page}.png",
    include_in_schema=False,
    responses={200: {"content": {"image/png": {}}}},
)
async def demo_document_page_preview(
    idno: str,
    page: int,
    dpi: int = Query(default=120, ge=36, le=300, description="Render resolution (72 dpi = 1x)."),
) -> Response:
    """Render one cached PDF page as PNG for the search demo carousel."""
    if page < 0:
        raise HTTPException(status_code=400, detail="page must be non-negative")
    pdf_path = resolve_document_pdf_path(idno)
    if not pdf_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="PDF not cached for this idno; ingest the document type first.",
        )
    try:
        png = await asyncio.to_thread(render_pdf_page_png, pdf_path, page, dpi=dpi)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to render PDF page: {e}") from e
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/health")
async def health(s: AppState = Depends(get_state)) -> dict[str, Any]:
    try:
        if s.settings.search_backend == "qdrant":
            return await s.search.health()
        assert s.client is not None
        ok = await s.client.cluster.health()
        return {"status": "ok", "cluster": ok.get("status"), "index": s.settings.index_name}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _embedding_status_extras(s: AppState) -> dict[str, Any]:
    """Fields common to local-backend embedding health payloads."""
    return {
        "model_id": s.settings.embedding_model_id,
        "query_encoding": s.settings.describe_query_encoding(),
    }


@app.get("/health/embeddings")
async def embedding_health(s: AppState = Depends(get_state)) -> dict[str, Any]:
    if s.settings.embedding_backend != "local":
        return {
            "status": "disabled",
            "backend": s.settings.embedding_backend,
            "detail": "Embedding model is not used with this backend.",
        }

    extras = _embedding_status_extras(s)
    if s.embedding is not None:
        return {
            "status": "ready",
            "backend": s.settings.embedding_backend,
            "dimension": s.embedding.embedding_dimension(),
            **extras,
        }

    if s.embedding_init_lock.locked():
        return {
            "status": "initializing",
            "backend": s.settings.embedding_backend,
            **extras,
        }

    if s.embedding_init_error:
        return {
            "status": "error",
            "backend": s.settings.embedding_backend,
            "detail": s.embedding_init_error,
            **extras,
        }

    return {
        "status": "not_initialized",
        "backend": s.settings.embedding_backend,
        **extras,
    }


@app.post("/health/embeddings/warmup")
async def embedding_warmup(s: AppState = Depends(get_state)) -> dict[str, Any]:
    if s.settings.embedding_backend != "local":
        return {
            "status": "disabled",
            "backend": s.settings.embedding_backend,
            "detail": "Warmup is only applicable for local embedding backend.",
        }

    try:
        await ensure_embedding_initialized(s)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"EmbeddingService initialization failed: {e}") from e

    if s.embedding is None:
        raise HTTPException(status_code=503, detail="EmbeddingService not initialized")

    return {
        "status": "ready",
        "backend": s.settings.embedding_backend,
        "dimension": s.embedding.embedding_dimension(),
        **_embedding_status_extras(s),
    }


@app.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, s: AppState = Depends(get_state)) -> SearchResponse:
    filters = body.filters.as_dict() if body.filters else None
    query_vector = None
    if body.mode in ("vector", "hybrid") and s.settings.embedding_backend == "local":
        try:
            await ensure_embedding_initialized(s)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"EmbeddingService initialization failed: {e}") from e
        if s.embedding is None:
            raise HTTPException(status_code=503, detail="EmbeddingService not initialized")
        try:
            vec = await asyncio.to_thread(
                s.embedding.encode_query,
                body.query,
                query_prompt=body.query_prompt,
                query_prompt_name=body.query_prompt_name,
            )
            query_vector = vec.tolist()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Query embedding failed ({e!r}). Check model load / memory (hybrid and vector modes require local embeddings).",
            ) from e

    inner_hits_spec = body.collapse_inner_hits.model_dump() if body.collapse_inner_hits else None
    use_idno_fast_path = body.mode == "keyword" and looks_like_catalog_idno(body.query)

    v_thr = body.vector_score_threshold
    if v_thr is None:
        v_thr = s.settings.qdrant_vector_score_threshold

    params = SearchParams(
        query=body.query,
        mode=body.mode,
        query_vector=query_vector,
        filters=filters,
        size=body.size,
        from_=body.from_,
        knn_k=body.knn_k,
        collapse_field=body.collapse_field,
        collapse_inner_hits=inner_hits_spec,
        include_embedding=body.include_embedding,
        include_facets=body.include_facets,
        facet_fields=body.facet_fields,
        vector_score_threshold=v_thr,
        use_idno_fast_path=use_idno_fast_path,
    )

    try:
        outcome = await s.search.search(params)
    except Exception as e:
        raise _search_backend_http_exception(e, s) from e

    dbg = outcome.debug_request if body.include_debug_request else None
    return SearchResponse(
        total=outcome.total,
        hits=outcome.hits,
        facets=coerce_search_facets(outcome.facets),
        opensearch_body=dbg,
        debug_request=dbg,
    )


def _search_backend_http_exception(e: Exception, s: AppState) -> HTTPException:
    if isinstance(e, ValueError):
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, NotFoundError):
        tgt = s.settings.qdrant_collection if s.settings.search_backend == "qdrant" else s.settings.index_name
        return HTTPException(
            status_code=404,
            detail=(f"Search target `{tgt}` was not found or is empty. Create and populate it before searching."),
        )
    if isinstance(e, RequestError):
        detail = str(e)
        if "Query vector has invalid dimension" in detail:
            detail = (
                "Embedding dimension mismatch between query model and index mapping. "
                f"Current model `{s.settings.embedding_model_id}` is incompatible with index `{s.settings.index_name}`. "
                "Use the same embedding model used during indexing, or recreate/reindex the index with this model."
            )
            return HTTPException(status_code=400, detail=detail)
        return HTTPException(status_code=400, detail=str(e))
    return HTTPException(status_code=503, detail=str(e))


@app.post("/recommendations", response_model=SearchResponse)
async def recommendations(body: RecommendRequest, s: AppState = Depends(get_state)) -> SearchResponse:
    filters = body.filters.as_dict() if body.filters else None
    r_thr = body.vector_score_threshold
    if r_thr is None:
        r_thr = s.settings.qdrant_vector_score_threshold
    params = RecommendParams(
        idno=body.idno.strip(),
        size=body.size,
        filters=filters,
        exclude_idno=body.exclude_idno,
        vector_strategy=body.vector_strategy,
        knn_k=body.knn_k,
        include_facets=body.include_facets,
        facet_fields=body.facet_fields,
        vector_score_threshold=r_thr,
    )
    try:
        outcome = await s.search.recommend_by_idno(params)
    except Exception as e:
        raise _search_backend_http_exception(e, s) from e
    return SearchResponse(
        total=outcome.total,
        hits=outcome.hits,
        facets=coerce_search_facets(outcome.facets),
        opensearch_body=None,
        debug_request=None,
    )


@app.post("/search/explain")
async def search_explain(body: ExplainSearchRequest, s: AppState = Depends(get_state)) -> dict[str, Any]:
    filters = body.filters.as_dict() if body.filters else None
    try:
        return await s.search.explain_by_idno(body.idno.strip(), filters)
    except Exception as e:
        raise _search_backend_http_exception(e, s) from e


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "demo": "/demo",
        "health": "/health",
        "health/embeddings": "/health/embeddings",
        "health/embeddings/warmup": "/health/embeddings/warmup",
        # "health/mcp": "/health/mcp",
        "recommendations": "/recommendations",
        "search/explain": "/search/explain",
        "search": "/search",
        "mcp": "/mcp",
    }


# Mount MCP app at root — the path="/mcp" in http_app() handles the /mcp route
app.mount("/", mcp_app)
