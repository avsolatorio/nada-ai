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
from opensearchpy.exceptions import TransportError

from nada_ai.app.auth import Principal
from nada_ai.app.state import AppState, get_state
from nada_ai.app.studies_errors import StudiesApiError, studies_guard
from nada_ai.app.studies_schemas import MAX_OFFSET, Engine, ErrorCode
from nada_ai.app.studies_search import register_validation_classifier
from nada_ai.app.variables_schemas import (
    VariableApplied,
    VariableFilters,
    VariableHit,
    VariableSearchRequest,
    VariableSearchResponse,
)
from nada_ai.search.backend.opensearch.variables_search import (
    IndexNotReady,
    VariableSearchJob,
    search_variables,
)
from nada_ai.search.backend.opensearch.variables_search import (
    VariableFilters as SearchFilters,
)

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
                {"filters": unknown, "supported": ["sids", "types"]},
            )
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


register_validation_classifier(VARIABLES_SEARCH_PATH, classify_variable_validation_error)


def _search_filters(filters: VariableFilters) -> SearchFilters:
    return SearchFilters(sids=tuple(filters.sids or ()), types=tuple(filters.types or ()))


@variables_router.post(VARIABLES_SEARCH_PATH, response_model=VariableSearchResponse)
async def variables_search(
    body: VariableSearchRequest,
    principal: Principal = Depends(studies_guard()),
    s: AppState = Depends(get_state),
) -> VariableSearchResponse:
    """Ranked variable hits, denormalized with their study's idno/title/nation; NADA hydrates nothing further."""
    del principal
    started = time.perf_counter()
    engine = Engine(s.settings.search_backend)
    if engine is not Engine.opensearch:
        raise StudiesApiError(
            ErrorCode.unsupported_capability,
            f"The {engine.value} engine does not support variables_search",
            {"capability": "variables_search", "engine": engine.value},
        )
    if body.offset + body.limit > MAX_OFFSET:
        raise StudiesApiError(
            ErrorCode.offset_out_of_range,
            f"offset + limit must be at most {MAX_OFFSET}",
            {"max_offset": MAX_OFFSET},
        )

    job = VariableSearchJob(
        client=s.client,
        index=s.settings.variables_index,
        query=body.query,
        filters=_search_filters(body.filters),
        limit=body.limit,
        offset=body.offset,
        sort_by=body.sort.value,
    )
    try:
        page = await search_variables(job)
    except IndexNotReady as e:
        raise StudiesApiError(
            ErrorCode.index_not_ready,
            "The variable search index is empty or is being rebuilt",
            {"index": str(e)},
        ) from e
    except TransportError as e:
        logger.warning("POST /variables/search: OpenSearch request failed: %s", e)
        raise StudiesApiError(ErrorCode.backend_unavailable, "OpenSearch is not reachable") from e

    return VariableSearchResponse(
        engine=engine,
        found=page.found,
        limit=body.limit,
        offset=body.offset,
        truncated=page.found > MAX_OFFSET,
        hits=[
            VariableHit(
                rank=body.offset + i + 1,
                uid=hit.uid,
                sid=hit.sid,
                idno=hit.idno,
                fid=hit.fid,
                vid=hit.vid,
                name=hit.name,
                label=hit.label,
                question=hit.question,
                title=hit.title,
                nation=hit.nation,
                dataset_type=hit.dataset_type,
                year_start=hit.year_start,
                year_end=hit.year_end,
                score=hit.score,
            )
            for i, hit in enumerate(page.hits)
        ],
        applied=VariableApplied(
            query=body.query,
            filters=body.filters.active(),
            sort=body.sort,
            limit=body.limit,
            offset=body.offset,
        ),
        timing_ms={"total": round((time.perf_counter() - started) * 1000, 1), "engine": float(page.took_ms or 0)},
    )
