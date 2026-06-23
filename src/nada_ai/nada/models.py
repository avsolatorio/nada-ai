from __future__ import annotations

from typing import Any, Literal

from metadataschemas.document_schema import ScriptSchemaDraft
from metadataschemas.geospatial_schema import GeospatialSchema
from metadataschemas.indicator_schema import TimeseriesSchema
from metadataschemas.microdata_schema import MicrodataSchema
from metadataschemas.table_schema import Model as TableSchema
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class CatalogMetadataRequest(BaseModel):
    """Request model for NADA catalog metadata (/api/catalog/{idno})."""

    idno: str = Field(description="The identifier number for the catalog item")


class TimeseriesCatalogMetadata(TimeseriesSchema):
    """IHSN timeseries / indicator metadata (``schematype`` timeseries or indicator)."""

    schematype: Literal["timeseries", "indicator"] = Field(
        default="timeseries",
        description="Schema discriminator for timeseries catalog metadata",
    )


class DocumentCatalogMetadata(ScriptSchemaDraft):
    """IHSN document metadata."""

    schematype: Literal["document"] = Field(
        default="document",
        description="Schema discriminator for document catalog metadata",
    )


class MicrodataCatalogMetadata(MicrodataSchema):
    """IHSN microdata / survey metadata."""

    schematype: Literal["microdata", "survey"] = Field(
        default="microdata",
        description="Schema discriminator for microdata catalog metadata",
    )


class GeospatialCatalogMetadata(GeospatialSchema):
    """IHSN geospatial metadata."""

    schematype: Literal["geospatial"] = Field(
        default="geospatial",
        description="Schema discriminator for geospatial catalog metadata",
    )


class TableCatalogMetadata(TableSchema):
    """IHSN table metadata."""

    schematype: Literal["table"] = Field(
        default="table",
        description="Schema discriminator for table catalog metadata",
    )


class FallbackCatalogMetadata(BaseModel):
    """Untyped metadata fallback when ``schematype`` is unknown or missing."""

    model_config = ConfigDict(extra="allow")

    schematype: str | None = Field(default=None, description="Reported schema type when present")


_METADATA_BY_SCHEMATYPE: dict[str, type[BaseModel]] = {
    "timeseries": TimeseriesCatalogMetadata,
    "indicator": TimeseriesCatalogMetadata,
    "document": DocumentCatalogMetadata,
    "microdata": MicrodataCatalogMetadata,
    "survey": MicrodataCatalogMetadata,
    "geospatial": GeospatialCatalogMetadata,
    "table": TableCatalogMetadata,
}

_METADATA_BY_DATASET_TYPE: dict[str, type[BaseModel]] = {
    "timeseries": TimeseriesCatalogMetadata,
    "indicator": TimeseriesCatalogMetadata,
    "document": DocumentCatalogMetadata,
    "microdata": MicrodataCatalogMetadata,
    "survey": MicrodataCatalogMetadata,
    "geospatial": GeospatialCatalogMetadata,
    "table": TableCatalogMetadata,
}


def _metadata_model_for(schema_type: str) -> type[BaseModel]:
    return _METADATA_BY_SCHEMATYPE.get(schema_type, FallbackCatalogMetadata)


def _parse_catalog_metadata(value: Any, *, dataset_type: str | None = None) -> Any:
    if not isinstance(value, dict):
        return value

    schema_type = str(value.get("schematype") or dataset_type or "").lower()
    return _metadata_model_for(schema_type).model_validate(value)


CatalogMetadata = (
    TimeseriesCatalogMetadata
    | DocumentCatalogMetadata
    | MicrodataCatalogMetadata
    | GeospatialCatalogMetadata
    | TableCatalogMetadata
    | FallbackCatalogMetadata
)


class CatalogMetadataDataset(BaseModel):
    """A catalog study row with typed metadata from GET /api/catalog/{idno}."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Internal catalog study ID")
    doi: str | None = Field(default=None, description="Digital object identifier")
    repositoryid: str | None = Field(default=None, description="Owning repository ID")
    type: str = Field(description="Dataset type (timeseries, document, survey, etc.)")
    idno: str = Field(description="Catalog identifier number")
    title: str = Field(description="Study title")
    year_start: str | None = Field(default=None, description="Start year of data coverage")
    year_end: str | None = Field(default=None, description="End year of data coverage")
    nation: str | None = Field(default=None, description="Primary nation or country label")
    authoring_entity: str | None = Field(default=None, description="Authoring organization")
    published: str | None = Field(default=None, description="Publication flag from catalog")
    created: str | None = Field(default=None, description="Record creation timestamp")
    changed: str | None = Field(default=None, description="Record last-changed timestamp")
    varcount: str | None = Field(default=None, description="Variable count when applicable")
    total_views: str | None = Field(default=None, description="Total catalog views")
    total_downloads: str | None = Field(default=None, description="Total catalog downloads")
    formid: str | None = Field(default=None, description="Associated form ID")
    data_access_type: str | None = Field(default=None, description="Data access classification")
    remote_data_url: str | None = Field(default=None, description="Remote data access URL")
    data_class_id: str | None = Field(default=None, description="Data classification ID")
    data_class_code: str | None = Field(default=None, description="Data classification code")
    data_class_title: str | None = Field(default=None, description="Data classification title")
    thumbnail: str | None = Field(default=None, description="Thumbnail asset filename or URL")
    abstract: str | None = Field(default=None, description="Study abstract or summary")
    link_study: str | None = Field(default=None, description="Linked study URL")
    link_indicator: str | None = Field(default=None, description="Linked indicator URL")
    link_report: str | None = Field(default=None, description="Linked report URL")
    # metadata: CatalogMetadata = Field(description="Type-specific IHSN metadata payload")
    metadata: dict[str, Any] = Field(description="Type-specific IHSN metadata payload")

    # @model_validator(mode="before")
    # @classmethod
    # def parse_typed_metadata(cls, data: Any) -> Any:
    #     if not isinstance(data, dict):
    #         return data

    #     raw_meta = data.get("metadata")
    #     if isinstance(raw_meta, dict):
    #         data = {
    #             **data,
    #             "metadata": _parse_catalog_metadata(raw_meta, dataset_type=str(data.get("type") or "")),
    #         }
    #     return data


class CatalogMetadataResponse(BaseModel):
    """Response model for NADA catalog metadata (/api/catalog/{idno})."""

    status: str = Field(description="API status (e.g. success)")
    dataset: CatalogMetadataDataset | None = Field(
        default=None,
        description="Catalog study payload when the request succeeds",
    )
    error: str | None = Field(default=None, description="Error message when the request fails")
