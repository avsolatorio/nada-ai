from __future__ import annotations

import logging
from typing import Any

import httpx
from ai4data.discovery.auth import get_catalog_auth_headers, get_catalog_cookies
from ai4data.discovery.config import metadata_catalog

from nada_ai.nada.models import (
    CatalogMetadataDataset,
    CatalogMetadataResponse,
    CatalogSearchRequest,
    CatalogSearchResponse,
    CatalogStudyRow,
    TimeseriesDataResponse,
    TimeseriesDataRow,
)

_logger = logging.getLogger(__name__)

_CATALOG_SEARCH_TIMEOUT = 30.0
_TIMESERIES_DATA_TIMEOUT = 30.0


def _catalog_search_url() -> str:
    return f"{metadata_catalog.url.rstrip('/')}/api/catalog/search"


def _catalog_metadata_url(idno: str) -> str:
    return f"{metadata_catalog.url.rstrip('/')}/api/catalog/{idno}"


def _timeseries_data_url(idno: str) -> str:
    return f"{metadata_catalog.url.rstrip('/')}/api/timeseries/data/{idno}"


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


def get_metadata(idno: str) -> CatalogMetadataResponse:
    """Get metadata for a given idno."""

    url = _catalog_metadata_url(idno)
    try:
        response = httpx.get(
            url,
            headers=get_catalog_auth_headers(),
            cookies=get_catalog_cookies(),
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        _logger.exception("Catalog metadata HTTP error")
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        return CatalogMetadataResponse(
            status="error",
            error=f"Catalog metadata failed ({exc.response.status_code}): {detail}",
        )
    except httpx.HTTPError as exc:
        _logger.exception("Catalog metadata network error")
        return CatalogMetadataResponse(status="error", error=f"Catalog metadata failed: {exc}")

    status = str(payload.get("status") or "").lower()
    if status not in {"", "success", "ok"}:
        message = payload.get("message") or payload.get("error") or status
        return CatalogMetadataResponse(status=status or "error", error=f"Catalog metadata API error: {message}")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        return CatalogMetadataResponse(status=status or "error", error="Catalog metadata API returned no dataset")

    return CatalogMetadataResponse(
        status=str(payload.get("status") or "success"),
        dataset=CatalogMetadataDataset.model_validate(dataset),
    )


async def get_timeseries_data(
    idno: str,
    *,
    limit: int = 100,
    offset: int = 0,
    from_year: int | None = None,
    to_year: int | None = None,
    country_codes: list[str] | None = None,
    sort_by: str | None = None,
    sort: str | None = None,
    dimensions: dict[str, str] | None = None,
) -> TimeseriesDataResponse:
    """Fetch observation rows from the timeseries data API.

    Args:
        idno: Indicator idno (e.g. ``VC.IHR.PSRC.P5``).
        limit: Max rows to return.
        offset: Pagination offset.
        from_year: Filter observations from this reporting year (inclusive).
        to_year: Filter observations up to this reporting year (inclusive).
        country_codes: ISO3 country codes to filter on (maps to ``c[COUNTRY_CODE]``).
        sort_by: Column name to sort by (e.g. ``OBS_VALUE``, ``TIME_PERIOD``).
        sort: Sort direction (``asc`` or ``desc``).
        dimensions: Arbitrary ``d[KEY]=value`` or ``c[KEY]=value`` filters beyond country.
    """
    url = _timeseries_data_url(idno)
    params: dict[str, str | int] = {"limit": limit, "offset": offset}

    if from_year is not None:
        params["from"] = from_year
    if to_year is not None:
        params["to"] = to_year
    if sort_by is not None:
        params["sort_by"] = sort_by
    if sort is not None:
        params["sort"] = sort
    if country_codes:
        params["c[COUNTRY_CODE]"] = ",".join(country_codes)
    if dimensions:
        for key, val in dimensions.items():
            params[key] = val

    try:
        async with httpx.AsyncClient(timeout=_TIMESERIES_DATA_TIMEOUT) as client:
            response = await client.get(
                url,
                params=params,
                headers=get_catalog_auth_headers(),
                cookies=get_catalog_cookies(),
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        _logger.exception("Timeseries data HTTP error")
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        return TimeseriesDataResponse(
            idno=idno,
            error=f"Timeseries data failed ({exc.response.status_code}): {detail}",
        )
    except httpx.HTTPError as exc:
        _logger.exception("Timeseries data network error")
        return TimeseriesDataResponse(idno=idno, error=f"Timeseries data failed: {exc}")

    status = str(payload.get("status") or "").lower()
    if status not in {"", "success", "ok"}:
        message = payload.get("message") or payload.get("error") or status
        return TimeseriesDataResponse(idno=idno, error=f"Timeseries data API error: {message}")

    result = payload.get("result") or {}
    rows = [TimeseriesDataRow.model_validate(r) for r in (result.get("data") or [])]
    total = int(result.get("total") or 0)
    found = int(result.get("found") or len(rows))

    return TimeseriesDataResponse(
        idno=idno,
        data=rows,
        total=total,
        found=found,
        limit=int(result.get("limit") or limit),
        offset=int(result.get("offset") or offset),
        has_more=(offset + found) < total,
    )
