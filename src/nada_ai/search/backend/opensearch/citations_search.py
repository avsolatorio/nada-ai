"""Standard citation search on OpenSearch: lexical only (see ``docs/citations-search-contract.md``).

Mirrors ``variables_search.py``: no semantic or hybrid mode, and ``LEXICAL_FIELDS``' weights are a starting point, not
a tuned result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: A citation is found by its title first, then its authors and subtitle, then keywords, and last the longer texts.
#: Must name every field of ``mapping.CITATION_TEXT_FIELDS`` (checked by a test).
LEXICAL_FIELDS = ("title^5", "authors^3", "subtitle^2", "keywords^2", "abstract", "notes")

#: A DOI is looked up, not tokenized: an exact (case-folded) match on it outranks any text match.
DOI_BOOST = 10


class IndexNotReady(RuntimeError):
    """Raised when the citation index does not exist yet."""


@dataclass(frozen=True)
class CitationFilters:
    ctypes: tuple[str, ...] = ()
    year_from: int | None = None
    year_to: int | None = None


@dataclass(frozen=True)
class CitationSearchJob:
    client: Any
    index: str
    query: str
    filters: CitationFilters
    limit: int
    offset: int
    sort_by: str  # "relevance", "title" or "year"
    order: str  # "asc" or "desc"; ignored for "relevance"


@dataclass(frozen=True)
class CitationHit:
    citation_id: int
    uuid: str | None
    title: str | None
    authors: str | None
    ctype: str | None
    pub_year: int | None
    doi: str | None
    score: float | None


@dataclass(frozen=True)
class CitationPage:
    found: int
    hits: list[CitationHit] = field(default_factory=list)
    took_ms: int | None = None


def lexical_query(query: str) -> dict[str, Any]:
    return {
        "bool": {
            "should": [
                {"multi_match": {"query": query, "fields": list(LEXICAL_FIELDS), "type": "best_fields"}},
                {"term": {"doi": {"value": query, "boost": DOI_BOOST}}},
            ],
            "minimum_should_match": 1,
        }
    }


def filter_clauses(filters: CitationFilters) -> list[dict[str, Any]]:
    """``bool.filter`` clauses; filters combine with AND, values within one key are any-of. Only published citations
    are ever searched."""
    clauses: list[dict[str, Any]] = [{"term": {"published": 1}}]
    if filters.ctypes:
        clauses.append({"terms": {"ctype": list(filters.ctypes)}})
    if filters.year_from is not None or filters.year_to is not None:
        bounds = {
            bound: value for bound, value in (("gte", filters.year_from), ("lte", filters.year_to)) if value is not None
        }
        clauses.append({"range": {"pub_year": bounds}})
    return clauses


_SORT_FIELDS: dict[str, str] = {"title": "title_sort", "year": "pub_year"}


def search_body(job: CitationSearchJob) -> dict[str, Any]:
    body: dict[str, Any] = {
        "size": job.limit,
        "from": job.offset,
        "track_total_hits": True,
        "query": {"bool": {"must": [lexical_query(job.query)], "filter": filter_clauses(job.filters)}},
        "_source": ["citation_id", "uuid", "title", "authors", "ctype", "pub_year", "doi"],
    }
    if job.sort_by != "relevance":
        sort_field = _SORT_FIELDS[job.sort_by]
        body["sort"] = [{sort_field: {"order": job.order, "missing": "_last"}}, "_score"]
    return body


def _hit(raw: dict[str, Any]) -> CitationHit:
    source = raw["_source"]
    return CitationHit(
        citation_id=int(source["citation_id"]),
        uuid=source.get("uuid"),
        title=source.get("title"),
        authors=source.get("authors"),
        ctype=source.get("ctype"),
        pub_year=source.get("pub_year"),
        doi=source.get("doi"),
        score=raw.get("_score"),
    )


async def search_citations(job: CitationSearchJob) -> CitationPage:
    from opensearchpy.exceptions import NotFoundError

    try:
        response = await job.client.search(index=job.index, body=search_body(job))
    except NotFoundError as e:
        raise IndexNotReady(job.index) from e

    hits = response["hits"]
    found = hits["total"]["value"] if isinstance(hits["total"], dict) else int(hits["total"])
    return CitationPage(
        found=int(found),
        hits=[_hit(h) for h in hits["hits"]],
        took_ms=response.get("took"),
    )
