"""Standard citation search on OpenSearch: lexical only (see ``docs/citations-search-contract.md``).

Mirrors ``variables_search.py``: no semantic or hybrid mode, and ``LEXICAL_FIELDS``' weights are a starting point, not
a tuned result. Paging is the same too:

* **offset** — ``from``/``size``, for the first ``citations_max_offset`` rows;
* **cursor** — ``search_after`` from a row's sort values (``nada_ai.search.cursor``), for any depth. A cursor going
  back (``before``) reads the reversed sort and turns the rows around.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from nada_ai.search.cursor import Cursor, Direction, SortValueKind, encode

#: A citation is found by its title first, then its authors and subtitle, then keywords, and last the longer texts.
#: Must name every field of ``mapping.CITATION_TEXT_FIELDS`` (checked by a test).
LEXICAL_FIELDS = ("title^5", "authors^3", "subtitle^2", "keywords^2", "abstract", "notes")

#: A DOI is looked up, not tokenized: an exact (case-folded) match on it outranks any text match.
DOI_BOOST = 10

#: Each sort as ``(field, fixed order)`` pairs; a ``None`` order follows the request's ``order``. Every sort ends on
#: ``citation_id``, the unique tiebreaker ``search_after`` needs (and what keeps equal titles or years in one order
#: from page to page).
SORTS: dict[str, tuple[tuple[str, str | None], ...]] = {
    "relevance": (("_score", "desc"), ("citation_id", "asc")),
    "title": (("title_sort", None), ("citation_id", None)),
    "year": (("pub_year", None), ("title_sort", None), ("citation_id", None)),
}

#: What a cursor's sort values must hold for each sort (one kind per field of ``SORTS``).
SORT_VALUE_KINDS: dict[str, tuple[SortValueKind, ...]] = {
    "relevance": (SortValueKind.score, SortValueKind.id),
    "title": (SortValueKind.text, SortValueKind.id),
    "year": (SortValueKind.number, SortValueKind.text, SortValueKind.id),
}

#: Fields sorted on that have no value in every document; the rest (``_score``, ``citation_id``) always do.
_MAY_BE_MISSING = frozenset({"title_sort", "pub_year"})


class IndexNotReady(RuntimeError):
    """Raised when the citation index does not exist yet, or holds no documents."""


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
    offset: int  # offset paging; ignored when ``cursor`` is set
    sort_by: str  # "relevance", "title" or "year"
    order: str  # "asc" or "desc"; ignored for "relevance"
    cursor: Cursor | None = None
    timeout_seconds: float = 5.0  # each OpenSearch request (setting search_timeout_seconds)


@dataclass(frozen=True)
class CitationHit:
    """A matching citation: its ids only — the caller (NADA) reads the citation itself from its database."""

    citation_id: int
    uuid: str | None
    score: float | None


@dataclass(frozen=True)
class CitationPage:
    found: int
    offset: int = 0  # 0-based position of the first hit in the whole result list
    hits: list[CitationHit] = field(default_factory=list)
    next_cursor: str | None = None
    prev_cursor: str | None = None
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


def _flip(order: str) -> str:
    return "asc" if order == "desc" else "desc"


def sort_clauses(sort_by: str, order: str, *, reverse: bool = False) -> list[dict[str, Any]]:
    """The OpenSearch sort of ``sort_by``; ``reverse`` gives the exact reverse order (every field flipped, and rows
    with no value moved from the end to the start), which is how a page *before* a row is read."""
    clauses: list[dict[str, Any]] = []
    for name, fixed in SORTS[sort_by]:
        direction = fixed or order
        spec: dict[str, Any] = {"order": _flip(direction) if reverse else direction}
        if name in _MAY_BE_MISSING:
            spec["missing"] = "_first" if reverse else "_last"
        clauses.append({name: spec})
    return clauses


def search_body(job: CitationSearchJob) -> dict[str, Any]:
    before = job.cursor is not None and job.cursor.direction is Direction.before
    body: dict[str, Any] = {
        "query": {"bool": {"must": [lexical_query(job.query)], "filter": filter_clauses(job.filters)}},
        "sort": sort_clauses(job.sort_by, job.order, reverse=before),
        "_source": ["citation_id", "uuid"],
        "track_scores": True,  # a search sorted on fields scores no hit unless asked (hits report their score)
    }
    if job.cursor is None:
        body["from"] = job.offset
        body["size"] = job.limit
        body["track_total_hits"] = True
    else:
        body["size"] = job.limit + 1  # one more than a page: whether another page follows in that direction
        body["search_after"] = list(job.cursor.sort_values)
        body["track_total_hits"] = False  # the total travels in the cursor
    return body


def preference(job: CitationSearchJob) -> str:
    """Routes every page of one search to the same shard copies, so relevance scores (which differ slightly between a
    primary and its replica) stay consistent from page to page."""
    key = json.dumps(
        [job.query, list(job.filters.ctypes), job.filters.year_from, job.filters.year_to, job.sort_by, job.order],
        sort_keys=True,
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _hit(raw: dict[str, Any]) -> CitationHit:
    source = raw["_source"]
    return CitationHit(citation_id=int(source["citation_id"]), uuid=source.get("uuid"), score=raw.get("_score"))


async def search_citations(job: CitationSearchJob) -> CitationPage:
    from opensearchpy.exceptions import NotFoundError

    try:
        response = await job.client.search(
            index=job.index,
            body=search_body(job),
            preference=preference(job),
            request_timeout=job.timeout_seconds,
        )
        hits = response["hits"]
        raw: list[dict[str, Any]] = hits["hits"]
        if job.cursor is None:
            found = int(hits["total"]["value"] if isinstance(hits["total"], dict) else hits["total"])
            # Zero results is either "nothing matches" or "nothing is indexed" (e.g. a backfill that stopped before
            # its first write); only the second is an error.
            if (
                found == 0
                and int((await job.client.count(index=job.index, request_timeout=job.timeout_seconds))["count"]) == 0
            ):
                raise IndexNotReady(job.index)
    except NotFoundError as e:
        raise IndexNotReady(job.index) from e

    cursor = job.cursor
    if cursor is None:
        rows, offset = raw, job.offset
        has_prev, has_next = offset > 0, offset + len(rows) < found
    else:
        found = cursor.total
        more = len(raw) > job.limit
        if cursor.direction is Direction.after:
            rows, offset = raw[: job.limit], cursor.position
            has_prev, has_next = cursor.position > 0, more
        else:
            rows = list(reversed(raw[: job.limit]))
            # With nothing more before, this page is the first one, whatever the cursor's position said.
            offset = max(cursor.position - len(rows), 0) if more else 0
            has_prev, has_next = more, True

    def cursor_from(row: dict[str, Any], direction: Direction, position: int) -> str:
        return encode(Cursor(direction, tuple(row["sort"]), position, found, False))

    return CitationPage(
        found=found,
        offset=offset,
        hits=[_hit(r) for r in rows],
        next_cursor=cursor_from(rows[-1], Direction.after, offset + len(rows)) if has_next and rows else None,
        prev_cursor=cursor_from(rows[0], Direction.before, offset) if has_prev and rows else None,
        took_ms=response.get("took"),
    )
