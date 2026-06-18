"""MCP Tools for the NADA server.

Thin wrapper layer that registers API functions as MCP tools with optimized signatures,
concise docstrings to reduce token context bloat, and validation schemas.
"""

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import (
    CatalogDataAccessType,
    CatalogSearchRequest,
    CatalogSearchResponse,
    CatalogSortBy,
    CatalogSortOrder,
)

from ._server_definition import mcp
from .tool_spans import instrument_mcp_tool


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
    """Search and filter published studies in the NADA catalog.

    Use when the user wants to find surveys, documents, timeseries metadata, geospatial
    datasets, or browse the catalog. Catalog host is configured via AI4DATA_METADATA_CATALOG_URL
    (e.g. https://training.ihsn.org/index.php).

    Provide `keywords` for full-text search; omit keywords to browse with filters only.
    Set `include_facets=True` when facet counts by type, country, or topic are needed.
    Use `page` and `page_size` for pagination; check `has_more` and `next_page` in the response.
    Each result includes `idno` and `url` for follow-up metadata retrieval.

    Args:
        keywords: Full-text search across study-level metadata.
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


search_catalog = mcp.tool(
    instrument_mcp_tool(_search_catalog, tool_name="nada_search_catalog"),
    name="nada_search_catalog",
)
