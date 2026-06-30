"""MCP Tools for the NADA server.

Thin wrapper layer that registers API functions as MCP tools with optimized signatures,
concise docstrings to reduce token context bloat, and validation schemas.
"""

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import (
    CatalogDataAccessType,
    CatalogMetadataResponse,
    CatalogSearchRequest,
    CatalogSearchResponse,
    CatalogSortBy,
    CatalogSortOrder,
    CodelistResponse,
    CompareResponse,
    ExtremesResponse,
    GrowthResponse,
    IndicatorSchemaResponse,
    RankResponse,
    SummarizeResponse,
    TimeseriesDataResponse,
)
from nada_ai.mcp_server import analytics

from ._server_definition import mcp
from .tool_config import get_mcp_tool_texts
from .tool_spans import instrument_mcp_tool

_TOOL_TEXTS = get_mcp_tool_texts()


async def _search_catalog(
    keywords: str | None = None,
    type: str = "timeseries",
    from_year: int | None = None,
    to_year: int | None = None,
    country: str | None = None,
    country_iso3: str | None = None,
    include_iso3: bool = False,
    include_countries: bool = False,
    collection: str | None = None,
    topic: str | None = None,
    tag: str | None = None,
    region: str | None = None,
    data_class: str | None = None,
    data_access_type: CatalogDataAccessType | None = None,
    study_id: int | None = None,
    repository: str | None = None,
    varcount: str | None = None,
    created: str | None = None,
    include_resources: bool = False,
    include_facets: bool = False,
    page_size: int = 15,
    page: int = 1,
    sort_by: CatalogSortBy = "title",
    sort_order: CatalogSortOrder = "asc",
) -> CatalogSearchResponse:
    """Search the metadata catalog (step 1 of the catalog workflow).

    Returns matching catalog entries with `idno`, `title`, and often `abstract`.
    For definition or methodology questions, search here first, then call get_metadata
    on the best `idno` to read full fields (e.g. indicator definition, survey methodology).

    Args:
        keywords: Semantic search over study metadata (titles, abstracts, definitions, methodology) by topic. Omit to browse with filters only.
        type: Dataset type filter (default timeseries). Use survey, document, geospatial, table, etc. Comma-separated for multiple.
        from_year: Start year for data collection period (inclusive).
        to_year: End year for data collection period (inclusive).
        country: Country name or ISO3 filter (pipe-separated, e.g. Afghanistan|Indonesia).
        country_iso3: ISO3 country code filter (pipe-separated, e.g. afg|ind).
        include_iso3: Include iso3 field on each result row.
        include_countries: Include countries array on each result row.
        collection: Collection repository ID (comma-separated).
        topic: Topic ID or name (pipe-separated).
        tag: Tag filter (pipe-separated).
        region: Region ID(s) (comma or pipe-separated).
        data_class: Data classification ID(s).
        data_access_type: Data access type (open, direct, public, licensed, enclave, remote, other).
        study_id: Internal numeric study ID (returns single matching study).
        repository: Repository ID filter.
        varcount: Variable count filter (e.g. >100, <50, =200).
        created: Creation date (YYYY/MM/DD or YYYY/MM/DD-YYYY/MM/DD).
        include_resources: Include external resource links on each row.
        include_facets: Include facet counts alongside results.
        page_size: Results per page (default 15, max 50).
        page: Page number, 1-based (default 1).
        sort_by: Sort field (rank/relevance require keywords; popularity sorts by views).
        sort_order: Sort direction (asc or desc).
    """
    request = CatalogSearchRequest(
        keywords=keywords,
        type=type,
        from_year=from_year,
        to_year=to_year,
        country=country,
        country_iso3=country_iso3,
        include_iso3=include_iso3,
        include_countries=include_countries,
        collection=collection,
        topic=topic,
        tag=tag,
        region=region,
        data_class=data_class,
        data_access_type=data_access_type,
        study_id=study_id,
        repository=repository,
        varcount=varcount,
        created=created,
        include_resources=include_resources,
        include_facets=include_facets,
        page_size=page_size,
        page=page,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return await nada_api.search_catalog(request)


def _get_metadata(idno: str) -> CatalogMetadataResponse:
    """Fetch full metadata for one catalog item by idno (step 2 of the catalog workflow).

    Use after search to answer definition, methodology, coverage, and producer questions
    from catalog fields (e.g. indicator `definition`, survey sampling details, document `abstract`).

    Args:
        idno: Catalog identifier from a prior search result (`items[].idno`). Required; do not guess.
    """
    return nada_api.get_metadata(idno)


async def _get_timeseries_data(
    idno: str,
    limit: int = 100,
    offset: int = 0,
    from_year: int | None = None,
    to_year: int | None = None,
    country_codes: list[str] | None = None,
    sort_by: str | None = None,
    sort: str | None = None,
) -> TimeseriesDataResponse:
    """Fetch observation rows for a timeseries indicator (step 3 of the catalog workflow).

    Use after get_metadata to retrieve actual data values for a known indicator idno.
    Returns paged observation rows with country, year, and numeric value columns.

    Args:
        idno: Indicator idno from a prior search or metadata result (e.g. ``VC.IHR.PSRC.P5``). Required.
        limit: Maximum rows to return per request (default 100, server may cap lower).
        offset: Pagination offset for subsequent pages.
        from_year: Filter to observations from this reporting year (inclusive).
        to_year: Filter to observations up to this reporting year (inclusive).
        country_codes: ISO3 country codes to filter on (e.g. ``["KEN", "UGA", "TZA"]``).
        sort_by: Column to sort by (e.g. ``OBS_VALUE``, ``TIME_PERIOD``, ``COUNTRY_CODE``).
        sort: Sort direction — ``asc`` or ``desc``.
    """
    return await nada_api.get_timeseries_data(
        idno,
        limit=limit,
        offset=offset,
        from_year=from_year,
        to_year=to_year,
        country_codes=country_codes,
        sort_by=sort_by,
        sort=sort,
    )


_get_data_tool_name = _TOOL_TEXTS.prefix + "_get_data"
_get_schema_tool_name = _TOOL_TEXTS.prefix + "_get_schema"
_get_codelist_tool_name = _TOOL_TEXTS.prefix + "_get_codelist"
_rank_tool_name = _TOOL_TEXTS.prefix + "_rank"
_extremes_tool_name = _TOOL_TEXTS.prefix + "_extremes"
_compare_tool_name = _TOOL_TEXTS.prefix + "_compare"
_summarize_tool_name = _TOOL_TEXTS.prefix + "_summarize"
_growth_tool_name = _TOOL_TEXTS.prefix + "_growth"

search_catalog = mcp.tool(
    instrument_mcp_tool(_search_catalog, tool_name=_TOOL_TEXTS.search_tool_name),
    name=_TOOL_TEXTS.search_tool_name,
    description=_TOOL_TEXTS.search_description,
)

get_metadata = mcp.tool(
    instrument_mcp_tool(_get_metadata, tool_name=_TOOL_TEXTS.get_metadata_tool_name),
    name=_TOOL_TEXTS.get_metadata_tool_name,
    description=_TOOL_TEXTS.get_metadata_description,
)

get_data = mcp.tool(
    instrument_mcp_tool(_get_timeseries_data, tool_name=_get_data_tool_name),
    name=_get_data_tool_name,
    description=(
        f"STEP 3 — Fetch actual observation data for a timeseries indicator. "
        f"Requires an `idno` from a prior {_TOOL_TEXTS.search_tool_name} or "
        f"{_TOOL_TEXTS.get_metadata_tool_name} result.\n\n"
        "Returns paged rows, each with the geography code, time period, and observation value columns. "
        "Use `from_year`/`to_year` to narrow the time range, `country_codes` for specific geographies, "
        "and `sort_by`/`sort` to order results (e.g. sort_by='OBS_VALUE', sort='desc' for top values). "
        "Paginate with `limit`/`offset` when `has_more` is true. "
        f"Call {_get_schema_tool_name} first to discover the column names and available dimensions for the indicator."
    ),
)


# ---------------------------------------------------------------------------
# Schema & codelist tools
# ---------------------------------------------------------------------------


async def _nada_get_schema(idno: str) -> IndicatorSchemaResponse:
    """Fetch the Data Structure Definition (DSD) for a timeseries indicator.

    Returns all column names, their structural roles (geography, time_period,
    observation_value, attribute, dimension, etc.), codelist IDs, time period
    format, and reporting year bounds.

    Call this before any analytical tool to discover:
    - The exact column names for geography, time, and observation value
    - Any disaggregation dimensions (e.g. SEX, AGE_GROUP, EDUCATION_LEVEL)
    - The time period format (YYYY for annual, YYYY-MM for monthly, etc.)
    - Valid year range for filtering

    Args:
        idno: Indicator idno (e.g. ``SP.POP.TOTL``). Required.
    """
    return await nada_api.get_indicator_schema(idno)


async def _nada_get_codelist(idno: str, component_name: str) -> CodelistResponse:
    """Get distinct values (codes + labels) for one DSD component.

    Derived by sampling data rows — no dedicated codelist endpoint exists.
    Use to discover valid filter values for dimension columns before calling
    analytical tools (e.g. valid SEX codes: M, F, T).

    Args:
        idno: Indicator idno. Required.
        component_name: Column name from the schema (e.g. ``COUNTRY_CODE``, ``SEX``). Required.
    """
    return await nada_api.get_codelist(idno, component_name)


get_schema = mcp.tool(
    instrument_mcp_tool(_nada_get_schema, tool_name=_get_schema_tool_name),
    name=_get_schema_tool_name,
    description=(
        f"Fetch the Data Structure Definition (DSD) for a timeseries indicator. "
        "Returns column names with their structural roles (geography, time_period, "
        "observation_value, attribute, dimension), codelist IDs, time period format, "
        "and reporting year bounds.\n\n"
        f"Call this BEFORE any analytical tool ({_rank_tool_name}, {_extremes_tool_name}, "
        f"{_compare_tool_name}, {_summarize_tool_name}, {_growth_tool_name}) to discover "
        "the correct column names and any disaggregation dimensions that require filtering."
    ),
)

get_codelist = mcp.tool(
    instrument_mcp_tool(_nada_get_codelist, tool_name=_get_codelist_tool_name),
    name=_get_codelist_tool_name,
    description=(
        "Get distinct values (codes + labels) for one DSD dimension component. "
        "Use after get_schema to discover valid filter values before calling analytical tools. "
        f"Example: call with component_name='SEX' to find valid codes (M, F, T) before passing "
        f"dimensions={{'SEX': 'F'}} to {_rank_tool_name} or other analytical tools."
    ),
)


# ---------------------------------------------------------------------------
# Analytical tools — all auto-paginate via get_all_timeseries_data
# ---------------------------------------------------------------------------


async def _nada_rank(
    idno: str,
    period: str,
    n: int = 10,
    ascending: bool = False,
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> RankResponse:
    """Rank ref areas (countries, provinces, etc.) by indicator value for a given period.

    Fetches all data for the period, then returns the top or bottom N ref areas
    sorted by observation value. Schema-driven: works for any indicator regardless
    of geography type or time period format.

    Args:
        idno: Indicator idno. Required.
        period: The time period to rank within — must exactly match TIME_PERIOD values
            (e.g. ``"2022"`` for annual, ``"2022-06"`` for monthly). Required.
        n: Number of ref areas to return (default 10).
        ascending: If True, return the lowest values first (bottom-N). Default False (top-N).
        from_year: Optional year filter passed to data fetch to reduce payload size.
        to_year: Optional year filter passed to data fetch.
        dimensions: Disaggregation filters — required when the indicator has dimension
            columns beyond geography and time (e.g. ``{"SEX": "F"}``). Call
            ``get_schema`` and ``get_codelist`` first to find valid values.
    """
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return RankResponse(idno=idno, period=period, n=n, ascending=ascending,
                            geo_column="", time_column="", obs_column="",
                            error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_

    data = await nada_api.get_all_timeseries_data(
        idno, from_year=from_year, to_year=to_year, dimensions=dimensions
    )
    if data.error:
        return RankResponse(idno=idno, period=period, n=n, ascending=ascending,
                            geo_column=schema.geo_column or "", time_column=schema.time_column or "",
                            obs_column=schema.obs_column or "", error=data.error)

    return analytics.rank(data.data, schema, period=period, n=n, ascending=ascending, dimensions=dimensions)


async def _nada_extremes(
    idno: str,
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> ExtremesResponse:
    """Find the global maximum and minimum observation across all periods and ref areas.

    Args:
        idno: Indicator idno. Required.
        from_year: Narrow to observations from this year (inclusive).
        to_year: Narrow to observations up to this year (inclusive).
        dimensions: Disaggregation filters (e.g. ``{"SEX": "F"}``).
    """
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return ExtremesResponse(idno=idno, geo_column="", time_column="", obs_column="",
                                error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_

    data = await nada_api.get_all_timeseries_data(
        idno, from_year=from_year, to_year=to_year, dimensions=dimensions
    )
    if data.error:
        return ExtremesResponse(idno=idno, geo_column=schema.geo_column or "",
                                time_column=schema.time_column or "",
                                obs_column=schema.obs_column or "", error=data.error)

    return analytics.get_extremes(data.data, schema, dimensions=dimensions)


async def _nada_compare(
    idno: str,
    ref_areas: list[str],
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> CompareResponse:
    """Build a pivoted time-series comparison for a set of ref areas.

    Returns one row per time period, with a value column per requested ref area.
    Useful for comparing trends across countries, provinces, or other geographies.

    Args:
        idno: Indicator idno. Required.
        ref_areas: List of geography codes to compare (e.g. ``["KEN", "UGA", "TZA"]``). Required.
        from_year: Filter to observations from this year (inclusive).
        to_year: Filter to observations up to this year (inclusive).
        dimensions: Disaggregation filters (e.g. ``{"SEX": "F"}``).
    """
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return CompareResponse(idno=idno, geo_column="", time_column="", obs_column="",
                               error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_

    data = await nada_api.get_all_timeseries_data(
        idno,
        from_year=from_year,
        to_year=to_year,
        country_codes=ref_areas,
        dimensions=dimensions,
    )
    if data.error:
        return CompareResponse(idno=idno, geo_column=schema.geo_column or "",
                               time_column=schema.time_column or "",
                               obs_column=schema.obs_column or "", error=data.error)

    return analytics.compare(data.data, schema, ref_areas=ref_areas, dimensions=dimensions)


async def _nada_summarize(
    idno: str,
    period: str,
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> SummarizeResponse:
    """Compute descriptive statistics (min, max, mean, median, std) across all ref areas for a period.

    Args:
        idno: Indicator idno. Required.
        period: The time period to summarize (must match TIME_PERIOD values exactly). Required.
        from_year: Optional year filter for data fetch.
        to_year: Optional year filter for data fetch.
        dimensions: Disaggregation filters (e.g. ``{"SEX": "F"}``).
    """
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return SummarizeResponse(idno=idno, period=period, geo_column="", obs_column="",
                                 error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_

    data = await nada_api.get_all_timeseries_data(
        idno, from_year=from_year, to_year=to_year, dimensions=dimensions
    )
    if data.error:
        return SummarizeResponse(idno=idno, period=period, geo_column=schema.geo_column or "",
                                 obs_column=schema.obs_column or "", error=data.error)

    return analytics.summarize(data.data, schema, period=period, dimensions=dimensions)


async def _nada_growth(
    idno: str,
    base_period: str,
    end_period: str,
    ref_areas: list[str] | None = None,
    dimensions: dict[str, str] | None = None,
) -> GrowthResponse:
    """Compute period-over-period absolute and percentage change per ref area.

    Args:
        idno: Indicator idno. Required.
        base_period: Starting period string (e.g. ``"2015"``). Required.
        end_period: Ending period string (e.g. ``"2022"``). Required.
        ref_areas: Specific geography codes to include. If None, all are returned.
        dimensions: Disaggregation filters (e.g. ``{"SEX": "F"}``).
    """
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return GrowthResponse(idno=idno, base_period=base_period, end_period=end_period,
                              geo_column="", obs_column="",
                              error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_

    # Fetch only the two periods needed
    data = await nada_api.get_all_timeseries_data(
        idno,
        country_codes=ref_areas,
        dimensions=dimensions,
    )
    if data.error:
        return GrowthResponse(idno=idno, base_period=base_period, end_period=end_period,
                              geo_column=schema.geo_column or "", obs_column=schema.obs_column or "",
                              error=data.error)

    return analytics.growth(data.data, schema, ref_areas=ref_areas,
                            base_period=base_period, end_period=end_period, dimensions=dimensions)


rank_tool = mcp.tool(
    instrument_mcp_tool(_nada_rank, tool_name=_rank_tool_name),
    name=_rank_tool_name,
    description=(
        f"Rank ref areas by indicator value for a specific period. "
        "Returns the top-N (or bottom-N) geographies sorted by observation value. "
        "Schema-driven: works for countries, provinces, districts, or any geography type.\n\n"
        f"Prerequisite: call {_get_schema_tool_name} first to verify column names and discover "
        "disaggregation dimensions. Pass `dimensions` to slice disaggregated indicators "
        "(e.g. rank female literacy rates by setting dimensions={'SEX': 'F'})."
    ),
)

extremes_tool = mcp.tool(
    instrument_mcp_tool(_nada_extremes, tool_name=_extremes_tool_name),
    name=_extremes_tool_name,
    description=(
        "Find the global maximum and minimum observation for an indicator across all "
        "periods and ref areas. Useful for answering 'which country had the highest X ever' "
        "or 'what was the worst year for Y'.\n\n"
        f"Prerequisite: call {_get_schema_tool_name} first. Pass `dimensions` for disaggregated indicators."
    ),
)

compare_tool = mcp.tool(
    instrument_mcp_tool(_nada_compare, tool_name=_compare_tool_name),
    name=_compare_tool_name,
    description=(
        "Build a pivoted time-series comparison for a set of ref areas. "
        "Returns one row per period with a value column per requested geography — "
        "ideal for trend comparisons across countries or regions.\n\n"
        f"Prerequisite: call {_get_schema_tool_name} first. Pass `dimensions` for disaggregated indicators."
    ),
)

summarize_tool = mcp.tool(
    instrument_mcp_tool(_nada_summarize, tool_name=_summarize_tool_name),
    name=_summarize_tool_name,
    description=(
        "Compute descriptive statistics (min, max, mean, median, std) across all ref areas "
        "for a given period. Useful for understanding the distribution of an indicator.\n\n"
        f"Prerequisite: call {_get_schema_tool_name} first. Pass `dimensions` for disaggregated indicators."
    ),
)

growth_tool = mcp.tool(
    instrument_mcp_tool(_nada_growth, tool_name=_growth_tool_name),
    name=_growth_tool_name,
    description=(
        "Compute period-over-period absolute and percentage change per ref area. "
        "Returns base value, end value, absolute change, and % change for each geography.\n\n"
        f"Prerequisite: call {_get_schema_tool_name} first. Pass `dimensions` for disaggregated indicators."
    ),
)
