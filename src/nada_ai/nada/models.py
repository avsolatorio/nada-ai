from __future__ import annotations

from typing import Any, Literal

from metadataschemas.document_schema import ScriptSchemaDraft
from metadataschemas.geospatial_schema import GeospatialSchema
from metadataschemas.indicator_schema import TimeseriesSchema
from metadataschemas.microdata_schema import MicrodataSchema
from metadataschemas.table_schema import Model as TableSchema
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


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

    keywords: str | None = Field(
        default=None,
        description=(
            "Semantic search over study metadata by topic or meaning, including titles, abstracts, "
            "definitions, and methodology text (API: sk). Omit to browse with filters only."
        ),
    )
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
    total_pages: int | None = Field(default=None, description="Total number of pages")
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


# ---------------------------------------------------------------------------
# Timeseries data API  (/api/timeseries/data/{idno})
# ---------------------------------------------------------------------------


class TimeseriesDataRow(BaseModel):
    """One observation row from the timeseries data API."""

    model_config = ConfigDict(extra="allow")

    DATASET: str | None = None
    INDICATOR: str | None = None
    INDICATOR_NAME: str | None = None
    COUNTRY_CODE: str | None = None
    COUNTRY_NAME: str | None = None
    ISO3C: str | None = None
    FREQ: str | None = None
    TIME_PERIOD: str | None = None
    OBS_VALUE: str | float | None = None
    CELL_NOTE: str | None = None
    reporting_year: int | None = None


class TimeseriesDataResponse(BaseModel):
    """Response model for the timeseries data API."""

    idno: str = Field(description="Indicator idno that was queried")
    data: list[TimeseriesDataRow] = Field(default_factory=list)
    total: int = Field(default=0, description="Total matching observations")
    found: int = Field(default=0, description="Observations returned in this page")
    limit: int = Field(default=0, description="Requested page size")
    offset: int = Field(default=0, description="Pagination offset used")
    has_more: bool = Field(default=False, description="Whether more rows are available")
    error: str | None = Field(default=None, description="Error message if the request failed")


# ---------------------------------------------------------------------------
# DSD schema  (/api/timeseries/data/{idno}/schema)
# ---------------------------------------------------------------------------

# Column types that are not analytical dimensions — they are metadata / roles
_NON_DIMENSION_COLUMN_TYPES: frozenset[str] = frozenset({
    "attribute",
    "indicator_id",
    "geography",
    "periodicity",
    "time_period",
    "observation_value",
})


class DSComponent(BaseModel):
    """One component (column) from a DSD."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Column name in data rows")
    label: str | None = Field(default=None, description="Human-readable label")
    description: str | None = Field(default=None, description="Long description")
    data_type: str | None = Field(default=None, description="Data type (string, double, …)")
    column_type: str = Field(description="Structural role of the column")
    codelist_id: str | None = Field(default=None, description="Codelist ID when present")
    time_period_format: str | None = Field(default=None, description="Time format when column_type=time_period")


class IndicatorSchema(BaseModel):
    """Parsed DSD schema for a timeseries indicator."""

    idno: str
    sid: int | None = None
    dsd_id: int | None = None
    components: list[DSComponent] = Field(default_factory=list)

    # Derived role columns — resolved at model construction
    geo_column: str | None = Field(default=None, description="Name of the geography column")
    time_column: str | None = Field(default=None, description="Name of the time_period column")
    obs_column: str | None = Field(default=None, description="Name of the observation_value column")
    freq_column: str | None = Field(default=None, description="Name of the periodicity column")
    dimension_columns: list[str] = Field(default_factory=list, description="Free disaggregation dimension column names")

    time_period_format: str | None = Field(default=None, description="Time period format string (e.g. YYYY)")
    reporting_year_bounds: dict[str, int] | None = Field(default=None, description="Min/max reporting years")

    @classmethod
    def from_api_result(cls, idno: str, result: dict[str, Any]) -> "IndicatorSchema":
        """Build from raw /api/timeseries/data/{idno}/schema result dict."""
        raw_components = result.get("components") or []
        components = [DSComponent.model_validate(c) for c in raw_components]

        geo_column = next((c.name for c in components if c.column_type == "geography"), None)
        time_column = next((c.name for c in components if c.column_type == "time_period"), None)
        obs_column = next((c.name for c in components if c.column_type == "observation_value"), None)
        freq_column = next((c.name for c in components if c.column_type == "periodicity"), None)
        dimension_columns = [c.name for c in components if c.column_type not in _NON_DIMENSION_COLUMN_TYPES]

        time_comp = next((c for c in components if c.column_type == "time_period"), None)
        time_format = time_comp.time_period_format if time_comp else None

        return cls(
            idno=idno,
            sid=result.get("sid"),
            dsd_id=result.get("dsd_id"),
            components=components,
            geo_column=geo_column,
            time_column=time_column,
            obs_column=obs_column,
            freq_column=freq_column,
            dimension_columns=dimension_columns,
            time_period_format=time_format,
            reporting_year_bounds=result.get("reporting_year_bounds"),
        )


class IndicatorSchemaResponse(BaseModel):
    """MCP response wrapping IndicatorSchema with error handling."""

    schema_: IndicatorSchema | None = Field(default=None, alias="schema")
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Codelist  (derived from data — no dedicated endpoint)
# ---------------------------------------------------------------------------


class CodelistEntry(BaseModel):
    """One code/label pair from a dimension codelist."""

    code: str
    label: str | None = None


class CodelistResponse(BaseModel):
    """Distinct values for one DSD component, derived from data sampling."""

    idno: str
    component: str
    label_column: str | None = Field(default=None, description="Companion label column used, if any")
    entries: list[CodelistEntry] = Field(default_factory=list)
    is_complete: bool = Field(default=False, description="True only if all distinct values were retrieved")
    error: str | None = None


# ---------------------------------------------------------------------------
# Analytical response models
# ---------------------------------------------------------------------------


class RankRow(BaseModel):
    """One row in a ranked result."""

    rank: int
    ref_area: str
    ref_area_label: str | None = None
    period: str
    value: float


class RankResponse(BaseModel):
    """Result of nada_rank — top/bottom N ref areas for a period."""

    idno: str
    indicator_name: str | None = None
    period: str
    n: int
    ascending: bool
    geo_column: str | None
    time_column: str | None
    obs_column: str | None
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    rows: list[RankRow] = Field(default_factory=list)
    total_ref_areas: int = Field(default=0, description="Total ref areas with data for this period")
    error: str | None = None


class ExtremePoint(BaseModel):
    """A single max or min observation."""

    ref_area: str
    ref_area_label: str | None = None
    period: str
    value: float


class ExtremesResponse(BaseModel):
    """Result of nada_extremes — global max and min observation."""

    idno: str
    indicator_name: str | None = None
    geo_column: str | None
    time_column: str | None
    obs_column: str | None
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    maximum: ExtremePoint | None = None
    minimum: ExtremePoint | None = None
    total_observations: int = 0
    error: str | None = None


class CompareRow(BaseModel):
    """One time period row in a cross-ref-area comparison."""

    period: str
    values: dict[str, float | None] = Field(default_factory=dict, description="ref_area → value")


class CompareResponse(BaseModel):
    """Result of nada_compare — pivoted time series for multiple ref areas."""

    idno: str
    indicator_name: str | None = None
    ref_areas: list[str] = Field(default_factory=list)
    ref_area_labels: dict[str, str] = Field(default_factory=dict)
    geo_column: str | None
    time_column: str | None
    obs_column: str | None
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    rows: list[CompareRow] = Field(default_factory=list)
    error: str | None = None

    @computed_field
    @property
    def rows_flat(self) -> list[dict[str, Any]]:
        """Pivot rows as flat dicts: {period, ref_area_1: val, ref_area_2: val, ...}."""
        return [
            {"period": row.period, **{k: v for k, v in row.values.items()}}
            for row in self.rows
        ]

    @computed_field
    @property
    def rows_unpivoted(self) -> list[dict[str, Any]]:
        """Unpivoted rows: one dict per (period, ref_area) pair with keys period/ref_area/value."""
        result = []
        for row in self.rows:
            for ref_area, value in row.values.items():
                result.append({"period": row.period, "ref_area": ref_area, "value": value})
        return result


class SummaryStats(BaseModel):
    """Descriptive statistics across ref areas for one period."""

    count: int = 0
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min_ref_area: str | None = None
    max_ref_area: str | None = None


class SummarizeResponse(BaseModel):
    """Result of nada_summarize — descriptive stats for a period."""

    idno: str
    indicator_name: str | None = None
    period: str
    geo_column: str | None
    obs_column: str | None
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    stats: SummaryStats = Field(default_factory=SummaryStats)
    error: str | None = None


class GrowthRow(BaseModel):
    """Period-over-period change for one ref area."""

    ref_area: str
    ref_area_label: str | None = None
    base_value: float | None = None
    end_value: float | None = None
    absolute_change: float | None = None
    pct_change: float | None = None


class GrowthResponse(BaseModel):
    """Result of nada_growth — period-over-period change per ref area."""

    idno: str
    indicator_name: str | None = None
    base_period: str
    end_period: str
    geo_column: str | None
    obs_column: str | None
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    rows: list[GrowthRow] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class CorrelatePoint(BaseModel):
    """One ref area in a cross-indicator correlation scatter."""

    ref_area: str
    ref_area_label: str | None = None
    value1: float | None = None
    value2: float | None = None


class CorrelateResponse(BaseModel):
    """Result of nada_correlate — Pearson r between two indicators for a period."""

    idno1: str
    idno2: str
    indicator_name1: str | None = None
    indicator_name2: str | None = None
    period: str
    geo_column: str | None = None
    n: int = 0
    pearson_r: float | None = None
    rows: list[CorrelatePoint] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


class OutlierRow(BaseModel):
    """One ref area with its Z-score and outlier flag."""

    ref_area: str
    ref_area_label: str | None = None
    value: float
    z_score: float
    is_outlier: bool


class OutliersResponse(BaseModel):
    """Result of nada_outliers — Z-score outlier detection for a period."""

    idno: str
    indicator_name: str | None = None
    period: str
    geo_column: str | None = None
    obs_column: str | None = None
    threshold: float = 2.0
    peer_mean: float | None = None
    peer_std: float | None = None
    n_outliers: int = 0
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    rows: list[OutlierRow] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


class TrendRow(BaseModel):
    """Linear trend for one ref area over a time window."""

    ref_area: str
    ref_area_label: str | None = None
    slope: float | None = None
    intercept: float | None = None
    r_squared: float | None = None
    n_periods: int = 0
    first_period: str | None = None
    last_period: str | None = None
    direction: str | None = None  # "improving", "declining", "stable"


class TrendResponse(BaseModel):
    """Result of nada_trend — linear regression per ref area."""

    idno: str
    indicator_name: str | None = None
    geo_column: str | None = None
    obs_column: str | None = None
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    rows: list[TrendRow] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class BenchmarkRow(BaseModel):
    """Benchmark result for one ref area against a peer group."""

    ref_area: str
    ref_area_label: str | None = None
    value: float | None = None
    percentile_rank: float | None = None
    z_score: float | None = None
    vs_mean: float | None = None
    vs_median: float | None = None


class BenchmarkResponse(BaseModel):
    """Result of nada_benchmark — ref area(s) vs peer group for a period."""

    idno: str
    indicator_name: str | None = None
    period: str
    geo_column: str | None = None
    obs_column: str | None = None
    peer_count: int = 0
    peer_mean: float | None = None
    peer_median: float | None = None
    peer_std: float | None = None
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    rows: list[BenchmarkRow] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class CoverageSummary(BaseModel):
    """Data coverage summary for one ref area."""

    ref_area: str
    ref_area_label: str | None = None
    n_periods: int = 0
    first_period: str | None = None
    last_period: str | None = None
    coverage_pct: float | None = None


class CoverageResponse(BaseModel):
    """Result of nada_coverage — data availability per ref area."""

    idno: str
    indicator_name: str | None = None
    geo_column: str | None = None
    time_column: str | None = None
    total_periods: int = 0
    total_ref_areas: int = 0
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    rows: list[CoverageSummary] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Join (cross-indicator)
# ---------------------------------------------------------------------------


class JoinRow(BaseModel):
    """One aligned observation from two indicators."""

    ref_area: str
    ref_area_label: str | None = None
    period: str
    value1: float | None = None
    value2: float | None = None


class JoinResponse(BaseModel):
    """Result of nada_join — row-aligned merge of two indicators."""

    idno1: str
    idno2: str
    indicator_name1: str | None = None
    indicator_name2: str | None = None
    geo_column: str | None = None
    n_matched: int = 0
    dimensions_applied1: dict[str, str] = Field(default_factory=dict)
    dimensions_applied2: dict[str, str] = Field(default_factory=dict)
    rows: list[JoinRow] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Aggregate (group aggregation)
# ---------------------------------------------------------------------------


class AggregateRow(BaseModel):
    """Aggregate statistics across a group of ref areas for one period."""

    period: str
    n_ref_areas: int = 0
    mean: float | None = None
    median: float | None = None
    total: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    std: float | None = None


class AggregateResponse(BaseModel):
    """Result of nada_aggregate — group-level statistics per period."""

    idno: str
    indicator_name: str | None = None
    ref_areas: list[str] = Field(default_factory=list)
    geo_column: str | None = None
    obs_column: str | None = None
    dimensions_applied: dict[str, str] = Field(default_factory=dict)
    rows: list[AggregateRow] = Field(default_factory=list)
    error: str | None = None
