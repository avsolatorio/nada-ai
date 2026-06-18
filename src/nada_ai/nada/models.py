from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MCPPagedResponse(BaseModel):
    """Response model for MCP paged results."""

    count: int = Field(default=0, description="Number of results in the current page")
    total_count: int | None = Field(default=None, description="Total number of results")
    page: int | None = Field(default=None, description="Current page number (1-based)")
    page_size: int | None = Field(default=None, description="Results per page")
    has_more: bool | None = Field(default=None, description="Whether there are more results")
    next_page: int | None = Field(default=None, description="Next page number when has_more is true")


CatalogSortBy = Literal[
    "rank",
    "relevance",
    "title",
    "nation",
    "country",
    "year",
    "created",
    "changed",
    "popularity",
    "total_views",
]

CatalogSortOrder = Literal["asc", "desc"]

CatalogDataAccessType = Literal[
    "open",
    "direct",
    "public",
    "licensed",
    "enclave",
    "remote",
    "other",
]


class CatalogSearchRequest(BaseModel):
    """Request model for NADA catalog search (/api/catalog/search)."""

    keywords: str | None = Field(default=None, description="Full-text search across study metadata (API: sk)")
    type: str = Field(
        default="timeseries",
        description="Dataset type filter: survey, geospatial, document, table, timeseries, etc. (comma-separated)",
    )
    from_year: int | None = Field(default=None, description="Start year for data collection period (inclusive)")
    to_year: int | None = Field(default=None, description="End year for data collection period (inclusive)")
    country: str | None = Field(
        default=None,
        description="Country name or ISO3 code filter (pipe-separated, e.g. Afghanistan|Indonesia)",
    )
    country_iso3: str | None = Field(
        default=None,
        description="ISO3 country code filter (pipe-separated, e.g. afg|ind)",
    )
    include_iso3: bool = Field(default=False, description="Include iso3 field on each result row")
    include_countries: bool = Field(default=False, description="Include countries array on each result row")
    collection: str | None = Field(default=None, description="Collection repository ID (comma-separated)")
    topic: str | None = Field(default=None, description="Topic ID or name (pipe-separated)")
    tag: str | None = Field(default=None, description="Tag filter (pipe-separated)")
    region: str | None = Field(default=None, description="Region ID(s) (comma or pipe-separated)")
    data_class: str | None = Field(default=None, description="Data classification ID(s)")
    data_access_type: CatalogDataAccessType | None = Field(
        default=None,
        description="Data access type filter (API: dtype)",
    )
    study_id: int | None = Field(default=None, description="Internal study ID (API: sid)")
    repository: str | None = Field(default=None, description="Repository ID filter (API: repo)")
    varcount: str | None = Field(
        default=None,
        description="Variable count filter (e.g. >100, <50, =200, or plain integer)",
    )
    created: str | None = Field(
        default=None,
        description="Creation date filter (YYYY/MM/DD or YYYY/MM/DD-YYYY/MM/DD)",
    )
    include_resources: bool = Field(default=False, description="Include external resource links on each row")
    include_facets: bool = Field(default=False, description="Include facet counts alongside results")
    page_size: int = Field(default=15, ge=1, le=50, description="Results per page (API: ps)")
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    sort_by: CatalogSortBy = Field(default="title", description="Sort field")
    sort_order: CatalogSortOrder = Field(default="asc", description="Sort direction")

    def to_api_params(self) -> dict[str, str | int]:
        """Map request fields to catalog search API query parameter names."""
        params: dict[str, str | int] = {
            "ps": self.page_size,
            "page": self.page,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
        }

        if self.keywords is not None:
            params["sk"] = self.keywords
        params["type"] = self.type
        if self.from_year is not None:
            params["from"] = self.from_year
        if self.to_year is not None:
            params["to"] = self.to_year
        if self.country is not None:
            params["country"] = self.country
        if self.country_iso3 is not None:
            params["country_iso3"] = self.country_iso3
        if self.include_iso3:
            params["inc_iso"] = 1
        if self.include_countries:
            params["inc_countries"] = 1
        if self.collection is not None:
            params["collection"] = self.collection
        if self.topic is not None:
            params["topic"] = self.topic
        if self.tag is not None:
            params["tag"] = self.tag
        if self.region is not None:
            params["region"] = self.region
        if self.data_class is not None:
            params["data_class"] = self.data_class
        if self.data_access_type is not None:
            params["dtype"] = self.data_access_type
        if self.study_id is not None:
            params["sid"] = self.study_id
        if self.repository is not None:
            params["repo"] = self.repository
        if self.varcount is not None:
            params["varcount"] = self.varcount
        if self.created is not None:
            params["created"] = self.created
        if self.include_resources:
            params["include_resources"] = "true"
        if self.include_facets:
            params["include_facets"] = 1

        return params


class CatalogStudyRow(BaseModel):
    """A single study row from catalog search results."""

    id: str
    type: str
    idno: str
    title: str
    url: str | None = None
    subtitle: str | None = None
    nation: str | None = None
    authoring_entity: str | None = None
    form_model: str | None = None
    data_class_id: str | None = None
    year_start: str | None = None
    year_end: str | None = None
    thumbnail: str | None = None
    repositoryid: str | None = None
    link_da: str | None = None
    repo_title: str | None = None
    created: str | None = None
    changed: str | None = None
    total_views: str | None = None
    total_downloads: str | None = None
    varcount: str | None = None
    abstract: str | None = None
    ts_dimensions: str | None = None
    ts_frequency: str | None = None
    ts_data_count: str | None = None
    ts_db_study_id: str | int | None = None
    ts_db_title: str | None = None
    doi: str | None = None
    iso3: str | None = None
    countries: list[str] | None = None
    resources: list[dict[str, Any]] | None = None

    model_config = {"extra": "ignore"}


class CatalogSearchResponse(MCPPagedResponse):
    """Response model for NADA catalog search."""

    items: list[CatalogStudyRow] = Field(default_factory=list)
    search_counts_by_type: dict[str, str] | None = None
    facets: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    error: str | None = None
