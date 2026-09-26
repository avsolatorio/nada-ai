"""Contract models for the standard variable search (``POST /variables/search``).

Lexical only — see ``docs/variables-search-contract.md`` for why, and for what is deliberately out of scope
(semantic/hybrid, countries/years/collections/repository filters the DB search has). Error shape, roles and rate
limiting are the same envelope as the study search (``studies_schemas.ErrorCode``/``ErrorResponse``): both
reuse it rather than defining a second one.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nada_ai.app.studies_schemas import MAX_OFFSET, Engine, PositiveInt, SortOrder, Token

DEFAULT_LIMIT = 15
MAX_LIMIT = 100
MAX_QUERY_LENGTH = 500

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)]


class VariableSortLiteral(StrEnum):
    relevance = "relevance"
    name = "name"
    title = "title"


class VariableFilters(BaseModel):
    """Filters combine with AND; values within a key are any-of. Only published variables (of published studies)
    are ever searched — that is not a filter, it is always true."""

    model_config = ConfigDict(extra="forbid")

    sids: list[PositiveInt] | None = Field(default=None, max_length=5000)
    types: list[Token] | None = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def _normalise(self) -> VariableFilters:
        for name in ("sids", "types"):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, list(dict.fromkeys(values)) or None)
        return self

    def active(self) -> dict[str, object]:
        return {k: v for k, v in self.model_dump().items() if v}


class VariableSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Query
    filters: VariableFilters = Field(default_factory=VariableFilters)
    sort: VariableSortLiteral = VariableSortLiteral.relevance
    order: SortOrder = Field(default=SortOrder.asc, description="Direction of a name or title sort.")
    limit: PositiveInt = Field(default=DEFAULT_LIMIT, le=MAX_LIMIT)
    offset: int = Field(default=0, ge=0, le=MAX_OFFSET)


class VariableHit(BaseModel):
    rank: int
    uid: int
    sid: int
    idno: str | None
    fid: str | None
    vid: str | None
    name: str | None
    label: str | None
    question: str | None
    title: str | None
    nation: str | None
    dataset_type: str | None
    year_start: int | None
    year_end: int | None
    score: float | None


class VariableApplied(BaseModel):
    query: str
    filters: dict[str, object]
    sort: VariableSortLiteral
    order: SortOrder
    limit: int
    offset: int


class VariableSearchResponse(BaseModel):
    engine: Engine
    found: int
    limit: int
    offset: int
    truncated: bool
    hits: list[VariableHit]
    applied: VariableApplied
    timing_ms: dict[str, float]
