from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError

from nada_ai import __version__
from nada_ai.app.schemas import SearchRequest, SearchResponse
from nada_ai.search.backend.opensearch.client import build_async_client
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.queries import build_search_query
from nada_ai.settings import Settings


class AppState:
    settings: Settings
    client: AsyncOpenSearch
    embedding: EmbeddingService | None
    embedding_init_lock: asyncio.Lock
    embedding_init_error: str | None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.settings = Settings()
    state.client = build_async_client(state.settings)
    state.embedding = None
    state.embedding_init_lock = asyncio.Lock()
    state.embedding_init_error = None
    yield
    try:
        await state.client.close()
    except Exception:
        pass


app = FastAPI(title="NADA AI Search", version=__version__, lifespan=lifespan)

_STATIC = Path(__file__).resolve().parent / "static"


def get_state() -> AppState:
    return state


async def ensure_embedding_initialized(s: AppState) -> None:
    """Lazily initialize local embedding backend once, on demand."""
    if s.settings.embedding_backend != "local":
        return
    if s.embedding is not None:
        return

    async with s.embedding_init_lock:
        if s.embedding is not None:
            return
        s.embedding_init_error = None
        try:
            s.embedding = await asyncio.to_thread(EmbeddingService, s.settings)
        except Exception as e:
            s.embedding_init_error = str(e)
            raise


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo_search_ui() -> HTMLResponse:
    """Minimal browser UI that calls ``POST /search`` (same origin)."""
    path = _STATIC / "demo.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="demo.html not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/health")
async def health(s: AppState = Depends(get_state)) -> dict[str, Any]:
    try:
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
        vec = await asyncio.to_thread(s.embedding.encode_query, body.query)
        query_vector = vec.tolist()

    inner_hits_spec = body.collapse_inner_hits.model_dump() if body.collapse_inner_hits else None
    try:
        q = build_search_query(
            s.settings,
            query_text=body.query,
            mode=body.mode,
            query_vector=query_vector,
            filters=filters,
            size=body.size,
            from_=body.from_,
            knn_k=body.knn_k,
            collapse_field=body.collapse_field,
            collapse_inner_hits=inner_hits_spec,
            include_embedding=body.include_embedding,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        resp = await s.client.search(index=s.settings.index_name, body=q)
    except NotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Index `{s.settings.index_name}` was not found. "
                "Create and populate the index before searching."
            ),
        ) from e
    except RequestError as e:
        detail = str(e)
        # Common local failure: query embedding dimension mismatches index knn_vector dimension.
        if "Query vector has invalid dimension" in detail:
            detail = (
                "Embedding dimension mismatch between query model and index mapping. "
                f"Current model `{s.settings.embedding_model_id}` is incompatible with index `{s.settings.index_name}`. "
                "Use the same embedding model used during indexing, or recreate/reindex the index with this model."
            )
            raise HTTPException(status_code=400, detail=detail) from e
        raise HTTPException(status_code=400, detail=str(e)) from e
    total = resp.get("hits", {}).get("total", {})
    if isinstance(total, dict):
        total_val = total.get("value")
    else:
        total_val = total
    hits_out = []
    for h in resp.get("hits", {}).get("hits", []):
        item: dict[str, Any] = {
            "_id": h.get("_id"),
            "_score": h.get("_score"),
            "_source": h.get("_source", {}),
        }
        if h.get("inner_hits"):
            item["inner_hits"] = h["inner_hits"]
        hits_out.append(item)
    return SearchResponse(
        total=total_val,
        hits=hits_out,
        opensearch_body=q if body.include_opensearch_body else None,
    )
