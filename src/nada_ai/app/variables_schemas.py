"""Contract models for the standard variable search (``POST /variables/search``).

Lexical only — see ``docs/variables-search-contract.md`` for why, and for what is deliberately out of scope
(semantic/hybrid). Error shape, roles and rate
limiting are the same envelope as the study search (``studies_schemas.ErrorCode``/``ErrorResponse``): both
reuse it rather than defining a second one.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nada_ai.app.studies_schemas import Engine, PositiveInt, SortOrder, Token

DEFAULT_LIMIT = 15
MAX_LIMIT = 100
MAX_QUERY_LENGTH = 500

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)]


class VariableSortLiteral(StrEnum):
    relevance = "relevance"
    name = "name"
    title = "title"


class VariableSearchMode(StrEnum):
    browse = "browse"  # no query: filters and sort only
    lexical = "lexical"


class Paging(StrEnum):
    offset = "offset"
    cursor = "cursor"


class VariableFilters(BaseModel):
    """Filters combine with AND; values within a key are any-of. They are all about the variable's study, and mean
    what the study search's filters of the same name mean (they are applied through the study index). Only variables
    of published studies are ever searched — that is not a filter, it is always true."""

    model_config = ConfigDict(extra="forbid")

    sids: list[PositiveInt] | None = Field(default=None, max_length=5000)
    types: list[Token] | None = Field(default=None, max_length=20)
    countries: list[PositiveInt] | None = Field(default=None, max_length=300)
    year_from: int | None = Field(default=None, ge=1, le=9999)
    year_to: int | None = Field(default=None, ge=1, le=9999)
    repository: Token | None = Field(default=None, description="Active repository scope (primary or secondary)")
    collections: list[Token] | None = Field(default=None, max_length=100, description="Repository ids; any-of")
    form_ids: list[PositiveInt] | None = Field(default=None, max_length=50, description="Data access form ids; any-of")

    @model_validator(mode="after")
    def _normalise(self) -> VariableFilters:
        for name in ("sids", "types", "countries", "collections", "form_ids"):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, list(dict.fromkeys(values)) or None)
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must be less than or equal to year_to")
        return self

    def active(self) -> dict[str, object]:
        return {k: v for k, v in self.model_dump().items() if v}


class VariableSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Query | None = Field(default=None, description="Omit to browse: filters and sort only.")
    filters: VariableFilters = Field(default_factory=VariableFilters)
    sort: VariableSortLiteral | None = Field(
        default=None, description="Default: relevance with a query, title (the study, then name) without one."
    )
    order: SortOrder = Field(default=SortOrder.asc, description="Direction of a name or title sort.")
    limit: PositiveInt = Field(default=DEFAULT_LIMIT, le=MAX_LIMIT)
    offset: int = Field(default=0, ge=0, description="Offset paging, up to limits.variables_max_offset rows.")
    cursor: str | None = Field(
        default=None, description="A next_cursor or prev_cursor of an earlier page of this search, in place of offset."
    )


class VariableHit(BaseModel):
    """A matching variable by id; the caller reads the variable itself from its database (``uid``)."""

    rank: int = Field(ge=1, description="1-based position in the whole result list (offset + index + 1)")
    uid: int
    sid: int
    score: float | None


class VariableApplied(BaseModel):
    query: str | None
    mode: VariableSearchMode
    filters: dict[str, object]
    sort: VariableSortLiteral
    order: SortOrder
    limit: int
    paging: Paging


class VariableSearchResponse(BaseModel):
    engine: Engine
    found: int
    found_is_lower_bound: bool = Field(
        description="True when only the first variables_count_cap matches were counted: found is a lower bound"
    )
    limit: int
    offset: int = Field(description="0-based position of the first hit in the whole result list")
    max_offset: int = Field(description="Rows reachable by offset paging; past them, page with the cursors")
    next_cursor: str | None = Field(description="Cursor of the next page; null on the last page")
    prev_cursor: str | None = Field(description="Cursor of the previous page; null on the first page")
    hits: list[VariableHit]
    applied: VariableApplied
    timing_ms: dict[str, float]
