from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

SearchMode = Literal["keyword", "vector", "hybrid"]


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str | None = None
    idno: str | None = None
    idnos: list[str] | None = None
    geographies: list[str] | None = None
    source: str | list[str] | None = None
    periodicity: str | None = None
    document_type: str | None = None
    authors: list[str] | None = None
    year_start: int | None = None
    year_end: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class CollapseInnerHits(BaseModel):
    """Nested hits for each collapse group (other qfields / rows sharing the same collapse field)."""

    name: str = Field(default="variants", min_length=1, description="Label in the response under hits[].inner_hits.<name>")
    size: int = Field(default=10, ge=1, le=100, description="Max docs per group beyond the representative hit")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: SearchMode = "hybrid"
    filters: SearchFilters | None = None
    size: int = Field(default=10, ge=1, le=100)
    from_: int = Field(default=0, ge=0, alias="from")
    knn_k: int = Field(default=50, ge=1, le=500)
    collapse_field: str | None = Field(
        default=None,
        description="OpenSearch field collapse: one hit per distinct value (e.g. 'idno' to dedupe qfield rows).",
    )
    collapse_inner_hits: CollapseInnerHits | None = Field(
        default=None,
        description="If set with collapse_field, returns additional matching docs per group (field collapse inner_hits).",
    )
    include_embedding: bool = Field(
        default=False,
        description="If false, dense vectors are omitted from stored payloads / _source in the response.",
    )
    include_debug_request: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_debug_request", "include_opensearch_body"),
        description="If true, include backend debug payload in the response (alias ``include_opensearch_body``).",
    )
    include_facets: bool = Field(
        default=False, description="If true, include facet bucket counts for whitelisted fields."
    )
    facet_fields: list[str] | None = Field(
        default=None,
        description="Subset of facet field names to aggregate; default is the full whitelist when ``include_facets`` is true.",
    )
    vector_score_threshold: float | None = Field(
        default=None,
        description=(
            "Qdrant vector / hybrid / recommend: minimum similarity score for dense neighbors "
            "(``query_points`` ``score_threshold``). When set, search ``total`` uses a paginated count of neighbors "
            "above this threshold (capped by ``NADA_QDRANT_VECTOR_COUNT_SCAN_CAP``). Omit to use "
            "``NADA_QDRANT_VECTOR_SCORE_THRESHOLD`` or no threshold."
        ),
    )
    query_prompt_name: str | None = Field(
        default=None,
        description=(
            "Override server ``query_prompt_name`` for this request (local embeddings, vector/hybrid). "
            "Ignored when ``query_prompt`` is set."
        ),
    )
    query_prompt: str | None = Field(
        default=None,
        description=(
            "Literal ``prompt=`` prefix for asymmetric query encoding (local embeddings, vector/hybrid). "
            "Overrides ``query_prompt_name`` and server defaults when set."
        ),
    )

    model_config = {"populate_by_name": True}

    @field_validator("query_prompt_name", "query_prompt", mode="before")
    @classmethod
    def empty_query_prompt_strings_to_none(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @model_validator(mode="after")
    def collapse_inner_hits_needs_field(self) -> SearchRequest:
        if self.collapse_inner_hits is not None and not self.collapse_field:
            raise ValueError("collapse_inner_hits requires collapse_field")
        return self


class FacetBucket(BaseModel):
    """One facet bucket (normalized OpenSearch ``terms`` agg / Qdrant facet hit)."""

    value: Any = Field(description="Bucket key (string, int, …).")
    count: int = Field(ge=0)


def coerce_search_facets(facets: dict[str, list[dict[str, Any]]] | None) -> dict[str, list[FacetBucket]] | None:
    """Coerce backend facet dicts into :class:`FacetBucket` lists for the API model."""
    if not facets:
        return None
    out: dict[str, list[FacetBucket]] = {}
    for field, rows in facets.items():
        out[field] = [FacetBucket(value=r.get("value"), count=int(r.get("count", 0))) for r in rows]
    return out


class SearchHit(BaseModel):
    id: str | None = None
    score: float | None = None
    source: dict[str, Any]


class SearchResponse(BaseModel):
    total: int | None = None
    hits: list[dict[str, Any]]
    facets: dict[str, list[FacetBucket]] | None = None
    opensearch_body: dict[str, Any] | None = Field(
        default=None,
        description="Deprecated. Same as ``debug_request`` when a debug payload was requested.",
    )
    debug_request: dict[str, Any] | None = Field(
        default=None,
        description="Present when the request asked for a debug payload; backend-specific shape.",
    )


class RecommendRequest(BaseModel):
    idno: str = Field(..., min_length=1)
    size: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters | None = None
    exclude_idno: bool = True
    vector_strategy: Literal["mean", "max_pool"] = "mean"
    knn_k: int = Field(default=50, ge=1, le=500)
    include_facets: bool = False
    facet_fields: list[str] | None = None
    vector_score_threshold: float | None = Field(
        default=None,
        description="Qdrant recommendations: same semantics as ``SearchRequest.vector_score_threshold``.",
    )


class ExplainSearchRequest(BaseModel):
    idno: str = Field(..., min_length=1)
    filters: SearchFilters | None = None
