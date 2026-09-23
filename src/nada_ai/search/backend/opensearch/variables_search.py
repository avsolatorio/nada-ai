"""Standard variable search on OpenSearch: lexical only (see ``docs/variables-search-contract.md``).

There is no semantic or hybrid mode here (deliberately — see the user's scoping decision in the contract doc) and
no golden-query eval yet, unlike the study search this mirrors the shape of: ``LEXICAL_FIELDS``' weights are a
starting point, not a tuned result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Weighted like a study's title vs. its other fields: the label is what a person searching variables reads: a
#: matching name (often a short code, e.g. "hhid") or question is worth less, and shared category text least. Must
#: name every field of ``mapping.VARIABLE_TEXT_FIELDS`` (checked by a test).
LEXICAL_FIELDS = ("label^10", "name^5", "question^3", "categories")


class IndexNotReady(RuntimeError):
    """Raised when the variable index does not exist yet."""


@dataclass(frozen=True)
class VariableFilters:
    sids: tuple[int, ...] = ()
    types: tuple[str, ...] = ()


@dataclass(frozen=True)
class VariableSearchJob:
    client: Any
    index: str
    query: str
    filters: VariableFilters
    limit: int
    offset: int
    sort_by: str  # "relevance", "name" or "title"


@dataclass(frozen=True)
class VariableHit:
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


@dataclass(frozen=True)
class VariablePage:
    found: int
    hits: list[VariableHit] = field(default_factory=list)
    took_ms: int | None = None


def lexical_query(query: str) -> dict[str, Any]:
    return {
        "multi_match": {
            "query": query,
            "fields": list(LEXICAL_FIELDS),
            "type": "best_fields",
        }
    }


def filter_clauses(filters: VariableFilters) -> list[dict[str, Any]]:
    """``bool.filter`` clauses; filters combine with AND, values within one key are any-of. Only published
    variables (of published studies) are ever searched."""
    clauses: list[dict[str, Any]] = [{"term": {"published": 1}}]
    if filters.sids:
        clauses.append({"terms": {"sid": list(filters.sids)}})
    if filters.types:
        clauses.append({"terms": {"dataset_type": list(filters.types)}})
    return clauses


_SORT_FIELDS: dict[str, str] = {"name": "name", "title": "title"}


def search_body(job: VariableSearchJob) -> dict[str, Any]:
    body: dict[str, Any] = {
        "size": job.limit,
        "from": job.offset,
        "track_total_hits": True,
        "query": {"bool": {"must": [lexical_query(job.query)], "filter": filter_clauses(job.filters)}},
        "_source": [
            "uid",
            "sid",
            "idno",
            "fid",
            "vid",
            "name",
            "label",
            "question",
            "title",
            "nation",
            "dataset_type",
            "year_start",
            "year_end",
        ],
    }
    if job.sort_by != "relevance":
        sort_field = _SORT_FIELDS.get(job.sort_by, "name")
        body["sort"] = [{sort_field: "asc"}, "_score"]
    return body


def _hit(raw: dict[str, Any]) -> VariableHit:
    source = raw["_source"]
    return VariableHit(
        uid=int(source["uid"]),
        sid=int(source["sid"]),
        idno=source.get("idno"),
        fid=source.get("fid"),
        vid=source.get("vid"),
        name=source.get("name"),
        label=source.get("label"),
        question=source.get("question"),
        title=source.get("title"),
        nation=source.get("nation"),
        dataset_type=source.get("dataset_type"),
        year_start=source.get("year_start"),
        year_end=source.get("year_end"),
        score=raw.get("_score"),
    )


async def search_variables(job: VariableSearchJob) -> VariablePage:
    from opensearchpy.exceptions import NotFoundError

    try:
        response = await job.client.search(index=job.index, body=search_body(job))
    except NotFoundError as e:
        raise IndexNotReady(job.index) from e

    hits = response["hits"]
    found = hits["total"]["value"] if isinstance(hits["total"], dict) else int(hits["total"])
    return VariablePage(
        found=int(found),
        hits=[_hit(h) for h in hits["hits"]],
        took_ms=response.get("took"),
    )
