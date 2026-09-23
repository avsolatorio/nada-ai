"""``POST /studies/search``: the standard study search (see ``docs/studies-search-contract.md``).

The route validates and resolves the request, runs the executor of the effective mode on the active engine, and
builds the contract response. Which modes exist is decided by the engine's ``EXECUTORS`` (through
``GET /info``'s ``IMPLEMENTED_STUDY_MODES``), so a mode that is not served answers ``unsupported_capability``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from opensearchpy.exceptions import TransportError

from nada_ai.app.auth import Principal
from nada_ai.app.info import modes_for
from nada_ai.app.keys_store import Role
from nada_ai.app.state import AppState, ensure_embedding_initialized, get_state
from nada_ai.app.studies_errors import StudiesApiError, studies_error_handler, studies_guard
from nada_ai.app.studies_schemas import (
    CANONICAL_FILTERS,
    MAX_OFFSET,
    Applied,
    AppliedSort,
    EffectiveMode,
    Engine,
    ErrorCode,
    ResponseWarning,
    SearchMode,
    SortField,
    SortOrder,
    StudyHit,
    StudySearchRequest,
    StudySearchResponse,
    WarningCode,
)
from nada_ai.search.backend.opensearch.studies_search import (
    EXECUTORS,
    EmbeddingUnavailable,
    EmbedFn,
    IndexNotReady,
    SearchJob,
    StudyPage,
    exact_idno_match,
)
from nada_ai.search.backend.opensearch.studies_semantic import StudyPolicy

logger = logging.getLogger(__name__)

studies_router = APIRouter()

STUDIES_SEARCH_PATH = "/studies/search"

#: For a query, the modes tried per requested mode, best first (``auto`` uses the best one the engine serves).
_MODE_PREFERENCE: dict[SearchMode, tuple[EffectiveMode, ...]] = {
    SearchMode.auto: (EffectiveMode.hybrid, EffectiveMode.lexical, EffectiveMode.semantic),
    SearchMode.lexical: (EffectiveMode.lexical,),
    SearchMode.semantic: (EffectiveMode.semantic,),
    SearchMode.hybrid: (EffectiveMode.hybrid,),
}


def _unsupported(capability: str, engine: Engine, message: str) -> StudiesApiError:
    return StudiesApiError(
        ErrorCode.unsupported_capability, message, {"capability": capability, "engine": engine.value}
    )


def resolve_mode(request: StudySearchRequest, engine: Engine, implemented: frozenset[str]) -> EffectiveMode:
    """The mode that will run: browse without a query, otherwise the best requested mode the engine serves."""
    if request.query is None:
        if EffectiveMode.browse.value not in implemented:
            raise _unsupported("browse", engine, f"The {engine.value} engine does not implement browse")
        return EffectiveMode.browse
    for mode in _MODE_PREFERENCE[request.mode]:
        if mode.value in implemented:
            return mode
    wanted = "lexical" if request.mode is SearchMode.auto else request.mode.value
    raise _unsupported(wanted, engine, f"The {engine.value} engine does not implement {wanted} search")


def resolve_sort(request: StudySearchRequest, mode: EffectiveMode) -> tuple[AppliedSort, list[ResponseWarning]]:
    """The effective sort, plus a warning when the request had to be adjusted.

    Default is ``relevance desc`` with a query and ``title asc`` without. ``relevance`` needs a query: without one
    it becomes the browse default and says so.
    """
    warnings: list[ResponseWarning] = []
    browse_default = AppliedSort(by=SortField.title, order=SortOrder.asc)
    requested = request.sort
    if mode is EffectiveMode.browse:
        if requested is None:
            return browse_default, warnings
        if requested.by is SortField.relevance:
            warnings.append(
                ResponseWarning(
                    code=WarningCode.sort_adjusted,
                    message="Sorting by relevance needs a query; results are sorted by title (ascending).",
                )
            )
            return browse_default, warnings
        return AppliedSort(by=requested.by, order=requested.order), warnings
    if requested is None:
        return AppliedSort(by=SortField.relevance, order=SortOrder.desc), warnings
    return AppliedSort(by=requested.by, order=requested.order), warnings


# ---------------------------------------------------------------------------------------
# Validation errors -> contract codes
# ---------------------------------------------------------------------------------------


def classify_validation_error(exc: RequestValidationError) -> StudiesApiError:
    """Map a request validation failure to the contract's error codes."""
    errors = exc.errors()
    unknown = sorted(
        str(e["loc"][2])
        for e in errors
        if e["type"] == "extra_forbidden" and len(e["loc"]) == 3 and tuple(e["loc"][:2]) == ("body", "filters")
    )
    if unknown:
        return StudiesApiError(
            ErrorCode.unknown_filter,
            f"Unknown filter key(s): {', '.join(unknown)}",
            {"filters": unknown, "supported": [spec.key for spec in CANONICAL_FILTERS]},
        )

    def message(error: dict[str, Any]) -> str:
        return str(error["msg"]).removeprefix("Value error, ")

    for error in errors:
        loc = tuple(error["loc"])
        if loc[:2] == ("body", "filters"):
            details = {"filter": str(loc[2])} if len(loc) > 2 else {}
            return StudiesApiError(ErrorCode.invalid_filter_value, message(error), details or None)
        if loc[:2] == ("body", "offset"):
            return StudiesApiError(ErrorCode.offset_out_of_range, message(error), {"max_offset": MAX_OFFSET})
    first = errors[0]
    where = ".".join(str(part) for part in first["loc"] if part != "body") or "body"
    return StudiesApiError(
        ErrorCode.invalid_request,
        f"Invalid request: {where}: {message(first)}",
        {"errors": [{"loc": [str(p) for p in e["loc"]], "message": message(e)} for e in errors[:10]]},
    )


#: Every route sharing this contract's validation envelope, and how each maps its own errors to a contract code.
#: ``app.variables_search`` registers itself into this at import time (before the app can serve a request), rather
#: than this module knowing about a sibling route.
VALIDATION_CLASSIFIERS: dict[str, Callable[[RequestValidationError], StudiesApiError]] = {
    STUDIES_SEARCH_PATH: classify_validation_error,
}


def register_validation_classifier(path: str, classifier: Callable[[RequestValidationError], StudiesApiError]) -> None:
    VALIDATION_CLASSIFIERS[path] = classifier


async def studies_validation_handler(request: Request, exc: RequestValidationError) -> Response:
    """Contract envelope for validation errors of every route in ``VALIDATION_CLASSIFIERS``; every other route
    keeps the framework default."""
    classifier = VALIDATION_CLASSIFIERS.get(request.url.path)
    if classifier is None:
        return await request_validation_exception_handler(request, exc)
    return await studies_error_handler(request, classifier(exc))


# ---------------------------------------------------------------------------------------
# Running a search
# ---------------------------------------------------------------------------------------


def _embedder(s: AppState) -> EmbedFn | None:
    """Embeds a query with the local model (loaded on first use); ``None`` when embeddings are not local."""
    if s.settings.embedding_backend != "local":
        return None

    async def embed(text: str) -> list[float]:
        try:
            await ensure_embedding_initialized(s)
            if s.embedding is None:
                raise RuntimeError("embedding service is not initialised")
            vector = await asyncio.to_thread(s.embedding.encode_query, text)
            return [float(x) for x in vector]
        except Exception as e:  # any failure to embed means the semantic leg is unavailable
            raise EmbeddingUnavailable(str(e)) from e

    return embed


async def _execute(
    body: StudySearchRequest,
    mode: EffectiveMode,
    sort: AppliedSort,
    warnings: list[ResponseWarning],
    implemented: frozenset[str],
    s: AppState,
) -> tuple[EffectiveMode, AppliedSort, StudyPage]:
    """Run the resolved mode. If the query cannot be embedded, ``auto`` degrades to keyword search (and says so);
    an explicit ``semantic`` or ``hybrid`` request fails instead of silently answering something else.

    A single-token query is checked against every study's own idno first (see ``exact_idno_match``): nothing a
    scored search does is as precise as an exact idno match, and it is what every other engine already gives NADA
    for this case.
    """
    job = SearchJob(
        client=s.client,
        index=s.settings.studies_index,
        chunk_index=s.settings.index_name,
        query=body.query,
        filters=body.filters,
        sort_by=sort.by,
        sort_order=sort.order,
        limit=body.limit,
        offset=body.offset,
        policy=StudyPolicy.from_settings(s.settings),
        embed=_embedder(s),
    )
    if mode is not EffectiveMode.browse and job.query is not None:
        exact = await exact_idno_match(job)
        if exact is not None:
            return mode, sort, exact

    try:
        return mode, sort, await EXECUTORS[mode](job)
    except EmbeddingUnavailable as e:
        if body.mode is not SearchMode.auto or EffectiveMode.lexical.value not in implemented:
            raise StudiesApiError(
                ErrorCode.embedding_unavailable, f"The query could not be embedded: {e}", {"mode": mode.value}
            ) from e
        logger.warning("POST /studies/search: embedding unavailable, answering with keyword search: %s", e)
        warnings.append(
            ResponseWarning(
                code=WarningCode.semantic_unavailable,
                message="The embedding model is unavailable; results are keyword matches only.",
            )
        )
        degraded, degraded_sort = EffectiveMode.lexical, resolve_sort(body, EffectiveMode.lexical)[0]
        page = await EXECUTORS[degraded](replace(job, sort_by=degraded_sort.by, sort_order=degraded_sort.order))
        return degraded, degraded_sort, page


# ---------------------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------------------


@studies_router.post(STUDIES_SEARCH_PATH, response_model=StudySearchResponse)
async def studies_search(
    body: StudySearchRequest,
    principal: Principal = Depends(studies_guard()),
    s: AppState = Depends(get_state),
) -> StudySearchResponse:
    """Ranked study ids plus counts; NADA hydrates the rows from its own database."""
    started = time.perf_counter()
    engine = Engine(s.settings.search_backend)
    implemented = modes_for(engine, s.settings)
    if not implemented:
        raise _unsupported("studies_search", engine, f"The {engine.value} engine does not support studies_search")
    if body.include_facets:
        raise _unsupported("facets", engine, "Facet counts are not implemented")
    if body.include_debug and principal.role is not Role.admin:
        raise StudiesApiError(ErrorCode.forbidden, "include_debug requires the admin role")

    mode = resolve_mode(body, engine, implemented)
    sort, warnings = resolve_sort(body, mode)
    if body.offset + body.limit > MAX_OFFSET:
        raise StudiesApiError(
            ErrorCode.offset_out_of_range,
            f"offset + limit must be at most {MAX_OFFSET}",
            {"max_offset": MAX_OFFSET},
        )

    try:
        mode, sort, page = await _execute(body, mode, sort, warnings, implemented, s)
    except IndexNotReady as e:
        raise StudiesApiError(
            ErrorCode.index_not_ready,
            "The search index is empty or is being rebuilt",
            {"index": str(e)},
        ) from e
    except TransportError as e:
        logger.warning("POST /studies/search: OpenSearch request failed: %s", e)
        raise StudiesApiError(ErrorCode.backend_unavailable, "OpenSearch is not reachable") from e

    return StudySearchResponse(
        engine=engine,
        found=page.found,
        limit=body.limit,
        offset=body.offset,
        truncated=page.found > MAX_OFFSET,
        search_counts_by_type=page.counts_by_type,
        hits=[StudyHit(rank=body.offset + i + 1, **hit) for i, hit in enumerate(page.hits)],
        applied=Applied(
            query=body.query,
            mode=mode,
            filters=body.filters.active(),
            sort=sort,
            limit=body.limit,
            offset=body.offset,
        ),
        warnings=warnings,
        timing_ms={"total": round((time.perf_counter() - started) * 1000, 1), "engine": float(page.took_ms or 0)},
        debug={"opensearch_requests": page.request_bodies} if body.include_debug else None,
    )
