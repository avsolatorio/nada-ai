from __future__ import annotations

import logging
from typing import Any

import httpx
from ai4data.discovery.auth import get_catalog_auth_headers, get_catalog_cookies
from ai4data.discovery.config import metadata_catalog

from nada_ai.nada.models import CatalogSearchRequest, CatalogSearchResponse, CatalogStudyRow

_logger = logging.getLogger(__name__)

_CATALOG_SEARCH_TIMEOUT = 30.0


def _catalog_search_url() -> str:
    return f"{metadata_catalog.url.rstrip('/')}/api/catalog/search"


def _parse_study_rows(rows: list[dict[str, Any]]) -> list[CatalogStudyRow]:
    return [CatalogStudyRow.model_validate(row) for row in rows]


def _build_paged_response(
    request: CatalogSearchRequest,
    result: dict[str, Any],
    *,
    facets: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> CatalogSearchResponse:
    rows = result.get("rows") or []
    items = _parse_study_rows(rows)
    total_count = int(result.get("found") or result.get("total") or 0)
    page = request.page
    page_size = request.page_size
    has_more = page * page_size < total_count

    return CatalogSearchResponse(
        items=items,
        count=len(items),
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_more=has_more,
        next_page=page + 1 if has_more else None,
        search_counts_by_type=result.get("search_counts_by_type"),
        facets=facets,
        params=params,
    )


async def search_catalog(request: CatalogSearchRequest) -> CatalogSearchResponse:
    """Search the NADA catalog via GET /api/catalog/search."""
    url = _catalog_search_url()
    params = request.to_api_params()

    try:
        async with httpx.AsyncClient(timeout=_CATALOG_SEARCH_TIMEOUT) as client:
            response = await client.get(
                url,
                params=params,
                headers=get_catalog_auth_headers(),
                cookies=get_catalog_cookies(),
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        _logger.exception("Catalog search HTTP error")
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        return CatalogSearchResponse(error=f"Catalog search failed ({exc.response.status_code}): {detail}")
    except httpx.HTTPError as exc:
        _logger.exception("Catalog search network error")
        return CatalogSearchResponse(error=f"Catalog search failed: {exc}")

    status = str(payload.get("status") or "").lower()
    if status not in {"", "success", "ok"}:
        message = payload.get("message") or payload.get("error") or status
        return CatalogSearchResponse(error=f"Catalog search API error: {message}")

    result = payload.get("result") or {}
    facets = result.get("facets") if request.include_facets else None
    return _build_paged_response(
        request,
        result,
        facets=facets,
        params=payload.get("params"),
    )
