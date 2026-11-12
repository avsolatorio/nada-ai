"""Contract models for the standard citation search (``POST /citations/search``).

Lexical only — see ``docs/citations-search-contract.md``. Error shape, roles and rate limiting are the same envelope as
the study search (``studies_schemas.ErrorCode``/``ErrorResponse``): the variable search reuses it too.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nada_ai.app.studies_schemas import Engine, PositiveInt, Token
from nada_ai.app.variables_schemas import Paging

DEFAULT_LIMIT = 15
MAX_LIMIT = 100
MAX_QUERY_LENGTH = 500

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)]
Year = Annotated[int, Field(ge=1000, le=9999)]


class CitationSortLiteral(StrEnum):
    relevance = "relevance"
    title = "title"
    year = "year"


class SortOrder(StrEnum):
    asc = "asc"
    desc = "desc"


class CitationFilters(BaseModel):
    """Filters combine with AND; values within a key are any-of. Only published citations are ever searched — that is
    not a filter, it is always true."""

    model_config = ConfigDict(extra="forbid")

    ctypes: list[Token] | None = Field(default=None, max_length=50)
    year_from: Year | None = None
    year_to: Year | None = None

    @model_validator(mode="after")
    def _normalise(self) -> CitationFilters:
        if self.ctypes is not None:
            self.ctypes = list(dict.fromkeys(self.ctypes)) or None
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must not be after year_to")
        return self

    def active(self) -> dict[str, object]:
        return {k: v for k, v in self.model_dump().items() if v not in (None, [], "")}


class CitationSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Query
    filters: CitationFilters = Field(default_factory=CitationFilters)
    sort: CitationSortLiteral = CitationSortLiteral.relevance
    order: SortOrder = Field(default=SortOrder.asc, description="Direction of a title or year sort.")
    limit: PositiveInt = Field(default=DEFAULT_LIMIT, le=MAX_LIMIT)
    offset: int = Field(default=0, ge=0, description="Offset paging, up to limits.citations_max_offset rows.")
    cursor: str | None = Field(
        default=None, description="A next_cursor or prev_cursor of an earlier page of this search, in place of offset."
    )


class CitationHit(BaseModel):
    """A matching citation by id; the caller reads the citation itself from its database (``citation_id``)."""

    rank: int = Field(ge=1, description="1-based position in the whole result list (offset + index + 1)")
    citation_id: int
    uuid: str | None
    score: float | None


class CitationApplied(BaseModel):
    query: str
    filters: dict[str, object]
    sort: CitationSortLiteral
    order: SortOrder
    limit: int
    paging: Paging


class CitationSearchResponse(BaseModel):
    engine: Engine
    found: int
    limit: int
    offset: int = Field(description="0-based position of the first hit in the whole result list")
    max_offset: int = Field(description="Rows reachable by offset paging; past them, page with the cursors")
    next_cursor: str | None = Field(description="Cursor of the next page; null on the last page")
    prev_cursor: str | None = Field(description="Cursor of the previous page; null on the first page")
    hits: list[CitationHit]
    applied: CitationApplied
    timing_ms: dict[str, float]
