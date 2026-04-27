from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SearchMode = Literal["keyword", "vector", "hybrid"]


class SearchFilters(BaseModel):
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
        description="If false, knn vectors are omitted from _source in the response (OpenSearch _source.excludes).",
    )
    include_opensearch_body: bool = Field(
        default=False,
        description="If true, the response includes the exact JSON body sent to OpenSearch ``search`` (for debugging / demos).",
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def collapse_inner_hits_needs_field(self) -> SearchRequest:
        if self.collapse_inner_hits is not None and not self.collapse_field:
            raise ValueError("collapse_inner_hits requires collapse_field")
        return self


class SearchHit(BaseModel):
    id: str | None = None
    score: float | None = None
    source: dict[str, Any]


class SearchResponse(BaseModel):
    total: int | None = None
    hits: list[dict[str, Any]]
    opensearch_body: dict[str, Any] | None = Field(
        default=None,
        description="Present when the request set ``include_opensearch_body``; the ``body`` argument passed to the OpenSearch client.",
    )
