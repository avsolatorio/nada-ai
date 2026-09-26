"""Contract models for the standard study search (``GET /info``, ``POST /studies/search``).

This module defines the wire contract only; it contains no search logic. The written specification is
``docs/studies-search-contract.md`` and the sample payloads are in ``tests/fixtures/studies_search/``.

Design rules (see the spec for the reasoning):

* Studies are identified by the NADA internal id ``sid`` (``surveys.id``); ``idno`` is returned as a guard.
* NADA resolves vocabulary (country names, regions, form codes, ...) into ids before calling; nada-ai only
  matches ids.
* Nothing is dropped silently: unknown filter keys and malformed values are rejected, and the effective
  search is echoed back in ``applied``.
* The contract is not versioned. New optional fields may be added; clients must ignore unknown response
  fields and use ``GET /info`` capabilities for feature detection.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

# --------------------------------------------------------------------------------------
# Contract constants
# --------------------------------------------------------------------------------------

DEFAULT_LIMIT = 15
MAX_LIMIT = 100
MAX_OFFSET = 10_000
MAX_QUERY_LENGTH = 500

PositiveInt = Annotated[int, Field(ge=1)]
Token = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
FacetName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class SearchMode(StrEnum):
    """Requested retrieval mode. ``auto`` lets the engine use its best available combination."""

    auto = "auto"
    lexical = "lexical"
    semantic = "semantic"
    hybrid = "hybrid"


class EffectiveMode(StrEnum):
    """Mode that actually ran (``browse`` = no query, filters and sort only)."""

    browse = "browse"
    lexical = "lexical"
    semantic = "semantic"
    hybrid = "hybrid"


class SortField(StrEnum):
    relevance = "relevance"
    title = "title"
    nation = "nation"
    year = "year"
    popularity = "popularity"
    created = "created"
    changed = "changed"


class SortOrder(StrEnum):
    asc = "asc"
    desc = "desc"


class MatchedBy(StrEnum):
    lexical = "lexical"
    semantic = "semantic"
    idno = "idno"  # an exact match on the study's own idno; never combined with lexical/semantic


class Engine(StrEnum):
    qdrant = "qdrant"
    opensearch = "opensearch"
    solr = "solr"


class WarningCode(StrEnum):
    semantic_unavailable = "semantic_unavailable"  # ``auto`` mode degraded to lexical
    sort_adjusted = "sort_adjusted"  # e.g. ``relevance`` requested without a query


class ErrorCode(StrEnum):
    invalid_request = "invalid_request"
    unknown_filter = "unknown_filter"
    invalid_filter_value = "invalid_filter_value"
    offset_out_of_range = "offset_out_of_range"
    unsupported_capability = "unsupported_capability"
    index_not_ready = "index_not_ready"
    backend_unavailable = "backend_unavailable"
    embedding_unavailable = "embedding_unavailable"
    query_rejected = "query_rejected"
    unauthorized = "unauthorized"
    forbidden = "forbidden"
    rate_limited = "rate_limited"


ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.invalid_request: 422,
    ErrorCode.unknown_filter: 422,
    ErrorCode.invalid_filter_value: 422,
    ErrorCode.offset_out_of_range: 422,
    ErrorCode.unsupported_capability: 501,
    ErrorCode.index_not_ready: 503,
    ErrorCode.backend_unavailable: 503,
    ErrorCode.embedding_unavailable: 503,
    ErrorCode.query_rejected: 400,
    ErrorCode.unauthorized: 401,
    ErrorCode.forbidden: 403,
    ErrorCode.rate_limited: 429,
}


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------


class FilterSpec(BaseModel):
    """One supported filter, as advertised by ``GET /info``."""

    model_config = ConfigDict(extra="forbid")

    key: str
    type: Literal["string_list", "int_list", "int", "string", "facet_map"]
    description: str


# Canonical filter registry: the single list of request filter keys. ``StudyFilters`` must declare exactly
# these fields (checked by tests); ``GET /info`` advertises them.
CANONICAL_FILTERS: tuple[FilterSpec, ...] = (
    FilterSpec(key="types", type="string_list", description="NADA dataset types (surveys.type); any-of"),
    FilterSpec(key="countries", type="int_list", description="Country ids; any-of"),
    FilterSpec(key="year_from", type="int", description="Data collection year lower bound (inclusive)"),
    FilterSpec(key="year_to", type="int", description="Data collection year upper bound (inclusive)"),
    FilterSpec(key="repository", type="string", description="Active repository scope (primary or secondary)"),
    FilterSpec(key="collections", type="string_list", description="Repository ids (collection facet); any-of"),
    FilterSpec(key="form_ids", type="int_list", description="Data access form ids; any-of"),
    FilterSpec(key="data_class_ids", type="int_list", description="Data classification ids; any-of"),
    FilterSpec(key="tags", type="string_list", description="Study tags; any-of"),
    FilterSpec(key="facets", type="facet_map", description="User-defined facets: name -> term ids (any-of per name)"),
    FilterSpec(key="sids", type="int_list", description="Restrict to these internal study ids"),
    FilterSpec(key="created_from", type="int", description="Created timestamp lower bound (unix seconds)"),
    FilterSpec(key="created_to", type="int", description="Created timestamp upper bound (unix seconds)"),
)


def _dedupe(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


class StudyFilters(BaseModel):
    """Filters combine with AND across keys; values within a key are any-of.

    An omitted key, ``null`` or an empty list means "no constraint". A non-empty list of values that match
    nothing yields zero results (never an error). Unknown keys are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    types: list[Token] | None = Field(default=None, max_length=20)
    countries: list[PositiveInt] | None = Field(default=None, max_length=300)
    year_from: int | None = Field(default=None, ge=1, le=9999)
    year_to: int | None = Field(default=None, ge=1, le=9999)
    repository: Token | None = None
    collections: list[Token] | None = Field(default=None, max_length=100)
    form_ids: list[PositiveInt] | None = Field(default=None, max_length=50)
    data_class_ids: list[PositiveInt] | None = Field(default=None, max_length=50)
    tags: list[Token] | None = Field(default=None, max_length=50)
    facets: dict[FacetName, list[PositiveInt]] | None = None
    sids: list[PositiveInt] | None = Field(default=None, max_length=5000)
    created_from: int | None = Field(default=None, ge=0)
    created_to: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _normalise(self) -> StudyFilters:
        for name in ("types", "countries", "collections", "form_ids", "data_class_ids", "tags", "sids"):
            values = getattr(self, name)
            if values is not None:
                values = _dedupe(values)
                setattr(self, name, values or None)
        if self.facets is not None:
            cleaned = {name: _dedupe(terms) for name, terms in self.facets.items() if terms}
            self.facets = cleaned or None
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must be less than or equal to year_to")
        if self.created_from is not None and self.created_to is not None and self.created_from > self.created_to:
            raise ValueError("created_from must be less than or equal to created_to")
        return self

    def active(self) -> dict[str, Any]:
        """Normalised filters that constrain the search (what ``applied.filters`` echoes)."""
        return self.model_dump(exclude_none=True)


# --------------------------------------------------------------------------------------
# POST /studies/search - request
# --------------------------------------------------------------------------------------


class StudySort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: SortField
    order: SortOrder = SortOrder.asc


class StudySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(default=None, max_length=MAX_QUERY_LENGTH)
    mode: SearchMode = SearchMode.auto
    filters: StudyFilters = Field(default_factory=StudyFilters)
    sort: StudySort | None = None
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
    offset: int = Field(default=0, ge=0, le=MAX_OFFSET)
    include_facets: bool = Field(default=False, description="Reserved; requires the ``facets`` capability")
    include_debug: bool = False

    @model_validator(mode="after")
    def _normalise(self) -> StudySearchRequest:
        if self.query is not None:
            self.query = self.query.strip() or None
        return self


# --------------------------------------------------------------------------------------
# POST /studies/search - response
# --------------------------------------------------------------------------------------


class Passage(BaseModel):
    """A matching passage inside a document study (replaces parsing raw engine hits)."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1, description="1-based page number")
    total_pages: int | None = Field(default=None, ge=1)
    score: float | None = None
    excerpt: str | None = None


class StudyHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sid: int = Field(ge=1)
    idno: str
    rank: int = Field(ge=1, description="1-based position in the full result set (offset + index + 1)")
    score: float | None = Field(
        default=None, description="Opaque fused relevance score; only comparable within one response; null in browse"
    )
    matched_by: list[MatchedBy] = Field(default_factory=list)
    passages: list[Passage] | None = None


class AppliedSort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: SortField
    order: SortOrder


class Applied(BaseModel):
    """The effective search, so nothing that shaped the result is hidden from the caller."""

    model_config = ConfigDict(extra="forbid")

    query: str | None
    mode: EffectiveMode
    filters: dict[str, Any]
    sort: AppliedSort
    limit: int
    offset: int


class ResponseWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: WarningCode
    message: str


class StudySearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Engine
    found: int = Field(ge=0, description="Studies in the result: all the keyword matches plus the semantic block")
    limit: int
    offset: int
    truncated: bool = Field(
        description="True when ``found`` exceeds ``limits.max_offset``: only that many can be paged"
    )
    search_counts_by_type: dict[str, int] = Field(
        description="Distinct studies per NADA dataset type, ignoring the ``types`` filter"
    )
    hits: list[StudyHit]
    applied: Applied
    warnings: list[ResponseWarning] = Field(default_factory=list)
    timing_ms: dict[str, float] | None = None
    debug: dict[str, Any] | None = None

    def invariant_violations(self) -> list[str]:
        """Contract invariants every response must satisfy (reused by implementation tests)."""
        problems: list[str] = []
        ranks = [h.rank for h in self.hits]
        if ranks != list(range(self.offset + 1, self.offset + len(ranks) + 1)):
            problems.append("hit ranks must be contiguous starting at offset + 1")
        if len(self.hits) > self.limit:
            problems.append("hits exceeds limit")
        if self.offset + len(self.hits) > self.found:
            problems.append("offset + hits exceeds found")
        if len({h.sid for h in self.hits}) != len(self.hits):
            problems.append("duplicate sid in hits")

        types = self.applied.filters.get("types")
        if types:
            expected = sum(self.search_counts_by_type.get(t, 0) for t in types)
            if self.found != expected:
                problems.append("found must equal the sum of search_counts_by_type over the types filter")
        elif sum(self.search_counts_by_type.values()) != self.found:
            problems.append("search_counts_by_type must add up to found when no types filter is applied")

        if self.truncated != (self.found > MAX_OFFSET):
            problems.append("truncated must be true exactly when found exceeds max_offset")

        if self.applied.mode == EffectiveMode.browse:
            if any(h.score is not None or h.matched_by for h in self.hits):
                problems.append("browse hits have no score and no matched_by")
        elif any(not h.matched_by for h in self.hits):
            problems.append("relevance hits need matched_by")
        return problems


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


# --------------------------------------------------------------------------------------
# GET /info
# --------------------------------------------------------------------------------------


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    studies_search: bool
    lexical: bool
    semantic: bool
    hybrid: bool
    browse: bool
    facets: bool
    variables_search: bool
    citations_search: bool


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_limit: int
    max_offset: int
    semantic_window: int = Field(description="The most studies the semantic side adds to a relevance search")
    max_query_length: int


class IndexInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    generation: str | None = Field(default=None, description="Changes whenever the index is rebuilt or engine switched")
    studies: int | None = Field(default=None, description="Distinct studies in the index")
    embedding_model: str | None = None
    embedding_dim: int | None = None


class InfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Engine
    engine_version: str | None = None
    capabilities: Capabilities
    id_key: Literal["sid"] = "sid"
    filters: list[FilterSpec] = Field(default_factory=list)
    sort_fields: list[SortField] = Field(default_factory=list)
    limits: Limits | None = None
    index: IndexInfo
