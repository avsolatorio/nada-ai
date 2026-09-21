"""``GET /info``: which engine is running, what the standard study search supports, and a summary of its index.

Clients use this for feature detection instead of an API version (see ``docs/studies-search-contract.md``).
Capabilities are never hard-coded as "supported": they come from ``IMPLEMENTED_STUDY_MODES``, which lists the study
search modes each engine actually implements, so ``/info`` cannot advertise something that is not there.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from opensearchpy.exceptions import NotFoundError, TransportError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from nada_ai.app.state import AppState, get_state
from nada_ai.app.studies_errors import StudiesApiError, studies_guard
from nada_ai.app.studies_schemas import (
    CANONICAL_FILTERS,
    MAX_LIMIT,
    MAX_OFFSET,
    MAX_QUERY_LENGTH,
    Capabilities,
    Engine,
    ErrorCode,
    IndexInfo,
    InfoResponse,
    Limits,
    SortField,
)
from nada_ai.search.backend.opensearch.studies_search import OPENSEARCH_STUDY_MODES
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

info_router = APIRouter()

#: Study search modes each engine implements. OpenSearch's come from its executors (``studies_search.EXECUTORS``), so
#: a mode is advertised exactly when it is served; steps 6-7 of ``docs/opensearch-standard-search-plan.md`` add
#: ``lexical``, ``semantic`` and ``hybrid`` there. Qdrant keeps the legacy ``POST /search`` path and stays empty.
IMPLEMENTED_STUDY_MODES: dict[Engine, frozenset[str]] = {
    Engine.opensearch: OPENSEARCH_STUDY_MODES,
    Engine.qdrant: frozenset(),
}


#: Modes that need the query embedded in nada-ai (the OpenSearch-ML backend embeds inside the cluster instead, which the
#: study search does not use).
_NEEDS_LOCAL_EMBEDDING = frozenset({"semantic", "hybrid"})


def modes_for(engine: Engine, settings: Settings) -> frozenset[str]:
    """The study search modes this deployment serves: what the engine implements, minus what its config rules out."""
    modes = IMPLEMENTED_STUDY_MODES.get(engine, frozenset())
    if settings.embedding_backend != "local":
        modes = modes - _NEEDS_LOCAL_EMBEDDING
    return modes


def capabilities_for(engine: Engine, settings: Settings) -> Capabilities:
    modes = modes_for(engine, settings)
    return Capabilities(
        studies_search=bool(modes),
        lexical="lexical" in modes,
        semantic="semantic" in modes,
        hybrid="hybrid" in modes,
        browse="browse" in modes,
        facets=False,
        variables_search=False,
        citations_search=False,
    )


def limits_for(settings: Settings) -> Limits:
    return Limits(
        max_limit=MAX_LIMIT,
        max_offset=MAX_OFFSET,
        query_result_cap=settings.studies_result_cap,
        max_query_length=MAX_QUERY_LENGTH,
    )


async def _opensearch_details(s: AppState) -> tuple[str | None, IndexInfo]:
    """Cluster version plus the study index (name, generation, study count) and the chunk index's embedding."""
    client = s.client
    settings = s.settings
    if client is None:
        raise StudiesApiError(ErrorCode.backend_unavailable, "OpenSearch client is not initialised")

    async def meta_and_count(index: str, *, want_count: bool) -> tuple[dict[str, Any], int | None]:
        try:
            mapping = await client.indices.get_mapping(index=index)
        except NotFoundError:
            return {}, None  # the engine is up but nothing has been indexed yet
        meta = next(iter(mapping.values()))["mappings"].get("_meta") or {}
        count = int((await client.count(index=index))["count"]) if want_count else None
        return meta, count

    try:
        version = (await client.info())["version"]["number"]
        study_meta, studies = await meta_and_count(settings.studies_index, want_count=True)
        chunk_meta, _ = await meta_and_count(settings.index_name, want_count=False)
    except TransportError as e:
        logger.warning("GET /info: OpenSearch request failed: %s", e)
        raise StudiesApiError(ErrorCode.backend_unavailable, "OpenSearch is not reachable") from e

    return version, IndexInfo(
        name=settings.studies_index,
        generation=study_meta.get("generation"),
        studies=studies,
        embedding_model=chunk_meta.get("embedding_model"),
        embedding_dim=chunk_meta.get("embedding_dim"),
    )


async def _qdrant_details(s: AppState) -> tuple[str | None, IndexInfo]:
    """Server version and the collection. Qdrant keeps no generation and no study-level count."""
    settings = s.settings
    collection = settings.qdrant_collection
    client = s.search.client  # type: ignore[attr-defined]
    try:
        version = (await client.info()).version
        dim = None
        if await client.collection_exists(collection):
            vectors = (await client.get_collection(collection)).config.params.vectors
            dim = getattr(vectors, "size", None)
    except (ResponseHandlingException, UnexpectedResponse) as e:
        logger.warning("GET /info: Qdrant request failed: %s", e)
        raise StudiesApiError(ErrorCode.backend_unavailable, "Qdrant is not reachable") from e

    return version, IndexInfo(name=collection, embedding_model=settings.embedding_model_id, embedding_dim=dim)


async def build_info(s: AppState) -> InfoResponse:
    engine = Engine(s.settings.search_backend)
    capabilities = capabilities_for(engine, s.settings)
    version, index = await (_opensearch_details(s) if engine is Engine.opensearch else _qdrant_details(s))
    return InfoResponse(
        engine=engine,
        engine_version=version,
        capabilities=capabilities,
        filters=list(CANONICAL_FILTERS) if capabilities.studies_search else [],
        sort_fields=list(SortField) if capabilities.studies_search else [],
        limits=limits_for(s.settings) if capabilities.studies_search else None,
        index=index,
    )


@info_router.get("/info", response_model=InfoResponse, dependencies=[Depends(studies_guard())])
async def info(s: AppState = Depends(get_state)) -> InfoResponse:
    """The running engine, its capabilities and a summary of its index."""
    return await build_info(s)
