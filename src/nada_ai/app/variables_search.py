"""``POST /variables/search``: the standard variable search (lexical only).

See ``docs/variables-search-contract.md``. Reuses the study search's error envelope, auth guard and rate limiter
(``studies_errors``) rather than defining a second one — the routes are siblings, not different APIs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.exceptions import RequestValidationError
from opensearchpy.exceptions import ConnectionTimeout, TransportError

from nada_ai.app.auth import Principal
from nada_ai.app.state import AppState, get_state
from nada_ai.app.studies_errors import StudiesApiError, opensearch_error, studies_guard
from nada_ai.app.studies_schemas import Engine, ErrorCode, StudyFilters
from nada_ai.app.studies_search import register_validation_classifier
from nada_ai.app.variables_schemas import (
    Paging,
    VariableApplied,
    VariableFilters,
    VariableHit,
    VariableSearchMode,
    VariableSearchRequest,
    VariableSearchResponse,
    VariableSortLiteral,
)
from nada_ai.search.backend.opensearch.variables_search import (
    BROWSE_SORT,
    SORT_VALUE_KINDS,
    IndexNotReady,
    TooManyStudies,
    VariableSearchJob,
    search_variables,
)
from nada_ai.search.cursor import Cursor, CursorError, decode

logger = logging.getLogger(__name__)

variables_router = APIRouter()

VARIABLES_SEARCH_PATH = "/variables/search"


def classify_variable_validation_error(exc: RequestValidationError) -> StudiesApiError:
    """Map a request validation failure to the contract's error codes (same codes the study search uses)."""
    errors = exc.errors()

    def message(error: dict[str, Any]) -> str:
        return str(error["msg"]).removeprefix("Value error, ")

    for error in errors:
        loc = tuple(error["loc"])
        if loc[:2] == ("body", "filters") and any(e["type"] == "extra_forbidden" for e in errors):
            unknown = sorted(
                str(e["loc"][2])
                for e in errors
                if e["type"] == "extra_forbidden" and len(e["loc"]) == 3 and tuple(e["loc"][:2]) == ("body", "filters")
            )
            return StudiesApiError(
                ErrorCode.unknown_filter,
                f"Unknown filter key(s): {', '.join(unknown)}",
                {"filters": unknown, "supported": list(VariableFilters.model_fields)},
            )
        if loc[:2] == ("body", "filters"):
            details = {"filter": str(loc[2])} if len(loc) > 2 else {}
            return StudiesApiError(ErrorCode.invalid_filter_value, message(error), details or None)
        if loc[:2] == ("body", "offset"):
            return StudiesApiError(ErrorCode.offset_out_of_range, message(error))
    first = errors[0]
    where = ".".join(str(part) for part in first["loc"] if part != "body") or "body"
    return StudiesApiError(
        ErrorCode.invalid_request,
        f"Invalid request: {where}: {message(first)}",
        {"errors": [{"loc": [str(p) for p in e["loc"]], "message": message(e)} for e in errors[:10]]},
    )


register_validation_classifier(VARIABLES_SEARCH_PATH, classify_variable_validation_error)


def _search_filters(filters: VariableFilters) -> StudyFilters:
    """The request's filters as the study search's: they select the studies whose variables may be returned."""
    return StudyFilters(**filters.model_dump(exclude_none=True))


def _resolve_sort(body: VariableSearchRequest) -> VariableSortLiteral:
    if body.sort is None:
        return VariableSortLiteral.relevance if body.query is not None else VariableSortLiteral(BROWSE_SORT)
    if body.sort is VariableSortLiteral.relevance and body.query is None:
        raise StudiesApiError(ErrorCode.invalid_request, "A relevance sort needs a query; browse by name or title")
    return body.sort


def _cursor(body: VariableSearchRequest, sort: VariableSortLiteral, max_offset: int) -> Cursor | None:
    """The request's cursor, checked against its sort; without one, the offset is checked against ``max_offset``."""
    if body.cursor is None:
        if body.offset + body.limit > max_offset:
            raise StudiesApiError(
                ErrorCode.offset_out_of_range,
                f"offset + limit must be at most {max_offset}; page further with next_cursor",
                {"max_offset": max_offset},
            )
        return None
    if "offset" in body.model_fields_set:
        raise StudiesApiError(ErrorCode.invalid_request, "Send either offset or cursor, not both")
    try:
        return decode(body.cursor, SORT_VALUE_KINDS[sort.value])
    except CursorError as e:
        raise StudiesApiError(
            ErrorCode.cursor_invalid,
            f"The cursor is not one of this search: {e}. Start again from the first page.",
        ) from e


@variables_router.post(VARIABLES_SEARCH_PATH, response_model=VariableSearchResponse)
async def variables_search(
    body: VariableSearchRequest,
    principal: Principal = Depends(studies_guard()),
    s: AppState = Depends(get_state),
) -> VariableSearchResponse:
    """Ranked variable hits by id (NADA reads the variables from its database), a page at a time: by offset for the
    first ``variables_max_offset`` rows, then by cursor."""
    del principal
    started = time.perf_counter()
    settings = s.settings
    engine = Engine(settings.search_backend)
    if engine is not Engine.opensearch:
        raise StudiesApiError(
            ErrorCode.unsupported_capability,
            f"The {engine.value} engine does not support variables_search",
            {"capability": "variables_search", "engine": engine.value},
        )
    sort = _resolve_sort(body)
    cursor = _cursor(body, sort, settings.variables_max_offset)

    job = VariableSearchJob(
        client=s.client,
        index=settings.variables_index,
        query=body.query,
        filters=_search_filters(body.filters),
        limit=body.limit,
        sort_by=sort.value,
        order=body.order.value,
        offset=body.offset,
        cursor=cursor,
        total_hits_cap=settings.variables_count_cap,
        timeout_seconds=settings.search_timeout_seconds,
        studies_index=settings.studies_index,
    )
    try:
        page = await search_variables(job)
    except IndexNotReady as e:
        raise StudiesApiError(
            ErrorCode.index_not_ready,
            "The variable or study search index is empty or is being rebuilt",
            {"index": str(e)},
        ) from e
    except TooManyStudies as e:
        raise StudiesApiError(
            ErrorCode.invalid_request,
            f"Too many studies match ({e}); narrow the search with filters",
        ) from e
    except ConnectionTimeout as e:
        logger.warning("POST /variables/search: no answer from OpenSearch within %ss", settings.search_timeout_seconds)
        raise StudiesApiError(
            ErrorCode.search_timeout,
            "The search took too long; try again, or narrow it with more words or filters",
            {"timeout_seconds": settings.search_timeout_seconds},
        ) from e
    except TransportError as e:
        raise opensearch_error("POST /variables/search", e) from e

    return VariableSearchResponse(
        engine=engine,
        found=page.found,
        found_is_lower_bound=page.found_is_lower_bound,
        limit=body.limit,
        offset=page.offset,
        max_offset=settings.variables_max_offset,
        next_cursor=page.next_cursor,
        prev_cursor=page.prev_cursor,
        hits=[
            VariableHit(rank=page.offset + i + 1, uid=hit.uid, sid=hit.sid, score=hit.score)
            for i, hit in enumerate(page.hits)
        ],
        applied=VariableApplied(
            query=body.query,
            mode=VariableSearchMode.browse if body.query is None else VariableSearchMode.lexical,
            filters=body.filters.active(),
            sort=sort,
            order=body.order,
            limit=body.limit,
            paging=Paging.offset if cursor is None else Paging.cursor,
        ),
        timing_ms={"total": round((time.perf_counter() - started) * 1000, 1), "engine": float(page.took_ms or 0)},
    )
