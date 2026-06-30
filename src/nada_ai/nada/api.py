from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx
from ai4data.discovery.auth import get_catalog_auth_headers, get_catalog_cookies
from ai4data.discovery.config import metadata_catalog

from nada_ai.nada.models import (
    CatalogMetadataDataset,
    CatalogMetadataResponse,
    CatalogSearchRequest,
    CatalogSearchResponse,
    CatalogStudyRow,
    CodelistEntry,
    CodelistResponse,
    IndicatorSchema,
    IndicatorSchemaResponse,
    TimeseriesDataResponse,
    TimeseriesDataRow,
)

_logger = logging.getLogger(__name__)

_CATALOG_SEARCH_TIMEOUT = 30.0
_TIMESERIES_DATA_TIMEOUT = 30.0

# Query params reserved by get_timeseries_data — dimension keys must not overwrite these
_RESERVED_TIMESERIES_PARAMS: frozenset[str] = frozenset({
    "limit", "offset", "from", "to", "sort_by", "sort",
})


def _catalog_search_url() -> str:
    return f"{metadata_catalog.url.rstrip('/')}/api/catalog/search"


def _catalog_metadata_url(idno: str) -> str:
    return f"{metadata_catalog.url.rstrip('/')}/api/catalog/{quote(idno, safe='')}"


def _timeseries_data_url(idno: str) -> str:
    return f"{metadata_catalog.url.rstrip('/')}/api/timeseries/data/{quote(idno, safe='')}"


def _timeseries_schema_url(idno: str) -> str:
    return f"{metadata_catalog.url.rstrip('/')}/api/timeseries/data/{quote(idno, safe='')}/schema"


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
    total_pages = max(1, -(-total_count // page_size)) if total_count > 0 else 1

    return CatalogSearchResponse(
        items=items,
        count=len(items),
        total_count=total_count,
        total_pages=total_pages,
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
        detail = f"HTTP {exc.response.status_code}" if exc.response is not None else "network error"
        return CatalogSearchResponse(error=f"Catalog search failed: {detail}")
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
        detail = f"HTTP {exc.response.status_code}" if exc.response is not None else "network error"
        return CatalogMetadataResponse(
            status="error",
            error=f"Catalog metadata failed: {detail}",
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
    geo_column: str = "COUNTRY_CODE",
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
        country_codes: Geography codes to filter on. The filter key sent to the server
            is ``c[{geo_column}]``, so pass the actual DSD geography column name via
            ``geo_column`` for non-country indicators (e.g. ``"PROVINCE_CODE"``).
        geo_column: Name of the DSD geography column used to build the server-side
            filter key (default ``"COUNTRY_CODE"``). Resolved from
            ``IndicatorSchema.geo_column`` by callers that have already fetched schema.
        sort_by: Column name to sort by (e.g. ``OBS_VALUE``, ``TIME_PERIOD``).
        sort: Sort direction (``asc`` or ``desc``).
        dimensions: Arbitrary ``d[KEY]=value`` or ``c[KEY]=value`` filters beyond geography.
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
        params[f"c[{geo_column}]"] = ",".join(country_codes)
    if dimensions:
        for key, val in dimensions.items():
            if key not in _RESERVED_TIMESERIES_PARAMS:
                params[key] = val
            else:
                _logger.warning("Ignoring dimension key '%s' — reserved query parameter", key)

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
        detail = f"HTTP {exc.response.status_code}" if exc.response is not None else "network error"
        return TimeseriesDataResponse(
            idno=idno,
            error=f"Timeseries data failed: {detail}",
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

    # When the server omits `total` (returns 0), fall back to found-vs-limit heuristic
    if total > 0:
        has_more = (offset + found) < total
    else:
        has_more = found >= limit

    return TimeseriesDataResponse(
        idno=idno,
        data=rows,
        total=total,
        found=found,
        limit=int(result.get("limit") or limit),
        offset=int(result.get("offset") or offset),
        has_more=has_more,
    )


async def get_indicator_schema(idno: str) -> IndicatorSchemaResponse:
    """Fetch the DSD schema for a timeseries indicator."""
    url = _timeseries_schema_url(idno)
    try:
        async with httpx.AsyncClient(timeout=_TIMESERIES_DATA_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=get_catalog_auth_headers(),
                cookies=get_catalog_cookies(),
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        _logger.exception("Schema fetch HTTP error")
        detail = f"HTTP {exc.response.status_code}" if exc.response is not None else "network error"
        return IndicatorSchemaResponse(error=f"Schema fetch failed: {detail}")
    except httpx.HTTPError as exc:
        _logger.exception("Schema fetch network error")
        return IndicatorSchemaResponse(error=f"Schema fetch failed: {exc}")

    status = str(payload.get("status") or "").lower()
    if status not in {"", "success", "ok"}:
        message = payload.get("message") or payload.get("error") or status
        return IndicatorSchemaResponse(error=f"Schema API error: {message}")

    result = payload.get("result") or {}
    return IndicatorSchemaResponse(schema=IndicatorSchema.from_api_result(idno, result))


async def get_codelist(
    idno: str,
    component_name: str,
    *,
    max_entries: int = 500,
) -> CodelistResponse:
    """Derive distinct code/label pairs for a DSD component from data sampling.

    Fetches up to ``max_entries`` rows, extracts unique values for
    ``component_name``, and pairs them with a companion label column when
    one exists (e.g. COUNTRY_CODE → COUNTRY_NAME).
    """
    schema_resp = await get_indicator_schema(idno)
    if schema_resp.error or schema_resp.schema_ is None:
        return CodelistResponse(idno=idno, component=component_name, error=schema_resp.error or "Schema unavailable")

    schema = schema_resp.schema_
    component_names = {c.name for c in schema.components}
    if component_name not in component_names:
        return CodelistResponse(
            idno=idno,
            component=component_name,
            error=f"Component '{component_name}' not found in schema. Available: {sorted(component_names)}",
        )

    # Find a companion label column — look for an attribute column that shares a name prefix
    label_column: str | None = None
    for c in schema.components:
        if c.column_type == "attribute" and c.name != component_name:
            # Heuristic: if the attribute name contains the component name minus a suffix like _CODE
            base = component_name.replace("_CODE", "").replace("_ID", "")
            if base in c.name and "NAME" in c.name:
                label_column = c.name
                break

    data_resp = await get_timeseries_data(idno, limit=max_entries)
    if data_resp.error:
        return CodelistResponse(idno=idno, component=component_name, error=data_resp.error)

    seen: dict[str, str | None] = {}
    for row in data_resp.data:
        row_dict = row.model_dump()
        code = row_dict.get(component_name)
        if code is None:
            code = (row.model_extra or {}).get(component_name)
        if code is None:
            continue
        code = str(code)
        if code not in seen:
            label = None
            if label_column:
                label = row_dict.get(label_column) or (row.model_extra or {}).get(label_column)
                label = str(label) if label is not None else None
            seen[code] = label

    entries = [CodelistEntry(code=c, label=l) for c, l in sorted(seen.items())]
    is_complete = data_resp.total <= max_entries

    return CodelistResponse(
        idno=idno,
        component=component_name,
        label_column=label_column,
        entries=entries,
        is_complete=is_complete,
    )


async def get_all_timeseries_data(
    idno: str,
    *,
    max_rows: int = 10_000,
    page_size: int = 1_000,
    max_pages: int = 50,
    from_year: int | None = None,
    to_year: int | None = None,
    country_codes: list[str] | None = None,
    geo_column: str = "COUNTRY_CODE",
    sort_by: str | None = None,
    sort: str | None = None,
    dimensions: dict[str, str] | None = None,
) -> TimeseriesDataResponse:
    """Fetch all matching rows up to ``max_rows``, auto-paginating.

    Pass ``geo_column=schema.geo_column`` when filtering non-country indicators
    so the server-side filter key matches the actual DSD column name.
    """
    all_rows: list[TimeseriesDataRow] = []
    offset = 0
    pages_fetched = 0

    first = await get_timeseries_data(
        idno,
        limit=min(page_size, max_rows),
        offset=0,
        from_year=from_year,
        to_year=to_year,
        country_codes=country_codes,
        geo_column=geo_column,
        sort_by=sort_by,
        sort=sort,
        dimensions=dimensions,
    )
    if first.error:
        return first

    all_rows.extend(first.data)
    total = first.total
    offset = len(all_rows)
    last_has_more = first.has_more
    pages_fetched = 1

    while len(all_rows) < min(total or max_rows, max_rows) and last_has_more and pages_fetched < max_pages:
        page = await get_timeseries_data(
            idno,
            limit=min(page_size, max_rows - len(all_rows)),
            offset=offset,
            from_year=from_year,
            to_year=to_year,
            country_codes=country_codes,
            geo_column=geo_column,
            sort_by=sort_by,
            sort=sort,
            dimensions=dimensions,
        )
        if page.error:
            break
        if not page.data:
            break
        all_rows.extend(page.data)
        offset = len(all_rows)
        total = page.total
        last_has_more = page.has_more
        pages_fetched += 1

    capped = len(all_rows) >= max_rows or pages_fetched >= max_pages
    return TimeseriesDataResponse(
        idno=idno,
        data=all_rows,
        total=total,
        found=len(all_rows),
        limit=max_rows,
        offset=0,
        has_more=capped and last_has_more,
    )
