"""``POST /citations/search``: the standard citation search (lexical only).

See ``docs/citations-search-contract.md``. Reuses the study search's error envelope, auth guard and rate limiter
(``studies_errors``), like the variable search: the routes are siblings, not different APIs. NADA takes the ordered
``citation_id``s and hydrates the rows from its own database.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.exceptions import RequestValidationError
from opensearchpy.exceptions import TransportError

from nada_ai.app.auth import Principal
from nada_ai.app.citations_schemas import (
    CitationApplied,
    CitationHit,
    CitationSearchRequest,
    CitationSearchResponse,
)
from nada_ai.app.state import AppState, get_state
from nada_ai.app.studies_errors import StudiesApiError, opensearch_error, studies_guard
from nada_ai.app.studies_schemas import MAX_OFFSET, Engine, ErrorCode
from nada_ai.app.studies_search import register_validation_classifier
from nada_ai.search.backend.opensearch.citations_search import (
    CitationFilters as SearchFilters,
)
from nada_ai.search.backend.opensearch.citations_search import (
    CitationSearchJob,
    IndexNotReady,
    search_citations,
)

logger = logging.getLogger(__name__)

citations_router = APIRouter()

CITATIONS_SEARCH_PATH = "/citations/search"

_SUPPORTED_FILTERS = ["ctypes", "year_from", "year_to"]


def classify_citation_validation_error(exc: RequestValidationError) -> StudiesApiError:
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
                {"filters": unknown, "supported": _SUPPORTED_FILTERS},
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


register_validation_classifier(CITATIONS_SEARCH_PATH, classify_citation_validation_error)


@citations_router.post(CITATIONS_SEARCH_PATH, response_model=CitationSearchResponse)
async def citations_search(
    body: CitationSearchRequest,
    principal: Principal = Depends(studies_guard()),
    s: AppState = Depends(get_state),
) -> CitationSearchResponse:
    """Ranked citation hits (ids first: NADA hydrates the full rows from its database)."""
    del principal
    started = time.perf_counter()
    engine = Engine(s.settings.search_backend)
    if engine is not Engine.opensearch:
        raise StudiesApiError(
            ErrorCode.unsupported_capability,
            f"The {engine.value} engine does not support citations_search",
            {"capability": "citations_search", "engine": engine.value},
        )
    if body.offset + body.limit > MAX_OFFSET:
        raise StudiesApiError(
            ErrorCode.offset_out_of_range,
            f"offset + limit must be at most {MAX_OFFSET}",
            {"max_offset": MAX_OFFSET},
        )

    job = CitationSearchJob(
        client=s.client,
        index=s.settings.citations_index,
        query=body.query,
        filters=SearchFilters(
            ctypes=tuple(body.filters.ctypes or ()),
            year_from=body.filters.year_from,
            year_to=body.filters.year_to,
        ),
        limit=body.limit,
        offset=body.offset,
        sort_by=body.sort.value,
        order=body.order.value,
    )
    try:
        page = await search_citations(job)
    except IndexNotReady as e:
        raise StudiesApiError(
            ErrorCode.index_not_ready,
            "The citation search index is empty or is being rebuilt",
            {"index": str(e)},
        ) from e
    except TransportError as e:
        raise opensearch_error("POST /citations/search", e) from e

    return CitationSearchResponse(
        engine=engine,
        found=page.found,
        limit=body.limit,
        offset=body.offset,
        truncated=page.found > MAX_OFFSET,
        hits=[
            CitationHit(
                rank=body.offset + i + 1,
                citation_id=hit.citation_id,
                uuid=hit.uuid,
                title=hit.title,
                authors=hit.authors,
                ctype=hit.ctype,
                pub_year=hit.pub_year,
                doi=hit.doi,
                score=hit.score,
            )
            for i, hit in enumerate(page.hits)
        ],
        applied=CitationApplied(
            query=body.query,
            filters=body.filters.active(),
            sort=body.sort,
            order=body.order,
            limit=body.limit,
            offset=body.offset,
        ),
        timing_ms={"total": round((time.perf_counter() - started) * 1000, 1), "engine": float(page.took_ms or 0)},
    )
