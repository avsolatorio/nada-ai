"""Standard variable search on OpenSearch: lexical only (see ``docs/variables-search-contract.md``).

There is no semantic or hybrid mode here (deliberately — see the user's scoping decision in the contract doc) and
no golden-query eval yet, unlike the study search this mirrors the shape of: ``LEXICAL_FIELDS``' weights are a
starting point, not a tuned result.

Two ways to page, both over the same sort (every sort ends on ``uid``, so rows never tie and a page boundary is
exact):

* **offset** — ``from``/``size``, for the first ``max_offset`` rows (a setting, at most OpenSearch's
  ``max_result_window``);
* **cursor** — ``search_after`` from a row's sort values (``nada_ai.search.cursor``), for any depth. A cursor going
  back (``before``) runs the sort reversed and flips the hits back into order.

A variable document holds the variable only (and its study's title, to sort by). Everything about its study —
published, type, countries, years, collections, repository, data access — is read from the study index on every
search: the study search's own filters give the ids of the studies whose variables may be returned, and the variable
query keeps those. A study's options changing (publish, collections, access) therefore touches no variable. It is an
allow-list: a study missing from the study index (a study index being rebuilt, a failed write) hides its variables
rather than showing them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from nada_ai.app.studies_schemas import StudyFilters
from nada_ai.search.backend.opensearch.mapping import FILTER_FACETS_KEY
from nada_ai.search.backend.opensearch.studies_search import filter_clauses as study_filter_clauses
from nada_ai.search.cursor import Cursor, Direction, SortValueKind, encode

#: Weighted like a study's title vs. its other fields: the label is what a person searching variables reads: a
#: matching name (often a short code, e.g. "hhid") or question is worth less, and shared category text least. Must
#: name every field of ``mapping.VARIABLE_TEXT_FIELDS`` (checked by a test).
LEXICAL_FIELDS = ("label^10", "name^5", "question^3", "categories")

#: Each sort as ``(field, fixed order)`` pairs; a ``None`` order follows the request's ``order``. Every sort ends on
#: ``uid``, the unique tiebreaker ``search_after`` needs. ``title`` is the study's title: its variables follow by name.
SORTS: dict[str, tuple[tuple[str, str | None], ...]] = {
    "relevance": (("_score", "desc"), ("uid", "asc")),
    "name": (("name.sort", None), ("uid", None)),
    "title": (("title", None), ("name.sort", None), ("uid", None)),
}

#: What a cursor's sort values must hold for each sort (one kind per field of ``SORTS``).
SORT_VALUE_KINDS: dict[str, tuple[SortValueKind, ...]] = {
    "relevance": (SortValueKind.score, SortValueKind.id),
    "name": (SortValueKind.text, SortValueKind.id),
    "title": (SortValueKind.text, SortValueKind.text, SortValueKind.id),
}

#: Sort of a search with no query (a browse): the study, then its variables by name — NADA's database default.
BROWSE_SORT = "title"

#: Fields sorted on that have no value in every document; the rest (``_score``, ``uid``) always do.
_MAY_BE_MISSING = frozenset({"name.sort", "title"})

#: Most studies one search may allow: the terms aggregation that reads them (``search.max_buckets`` is 65,535) and
#: the terms filter that uses them (``index.max_terms_count`` is 65,536) both stop there.
MAX_ALLOWED_STUDIES = 65_535


class IndexNotReady(RuntimeError):
    """Raised when the variable index, or the study index it reads study state from, is missing or empty."""


class TooManyStudies(RuntimeError):
    """Raised when more studies match than one search can allow (``MAX_ALLOWED_STUDIES``)."""


@dataclass(frozen=True)
class VariableSearchJob:
    client: Any
    index: str
    query: str | None  # None: browse (filters and sort only)
    filters: StudyFilters  # study-level only: they are applied through the study index
    limit: int
    sort_by: str  # a key of SORTS
    order: str = "asc"  # "asc" or "desc"; ignored for "relevance"
    offset: int = 0  # offset paging; ignored when ``cursor`` is set
    cursor: Cursor | None = None
    total_hits_cap: int = 20_000  # a query's matches are counted exactly up to this many
    timeout_seconds: float = 5.0
    studies_index: str = "nada-metadata-studies"


@dataclass(frozen=True)
class VariableHit:
    """A matching variable: its ids only — the caller (NADA) reads the variable itself from its database by uid."""

    uid: int
    sid: int
    score: float | None


@dataclass(frozen=True)
class VariablePage:
    found: int
    found_is_lower_bound: bool
    offset: int  # 0-based position of the first hit in the whole result list
    hits: list[VariableHit] = field(default_factory=list)
    next_cursor: str | None = None
    prev_cursor: str | None = None
    took_ms: int | None = None


def lexical_query(query: str) -> dict[str, Any]:
    return {
        "multi_match": {
            "query": query,
            "fields": list(LEXICAL_FIELDS),
            "type": "best_fields",
        }
    }


def study_clauses(filters: StudyFilters) -> list[dict[str, Any]]:
    """The study index query of the studies whose variables a search may return: the study search's own filters
    (published only, countries, years, repository, collections, data access, sids) plus the dataset types, which the
    study search applies separately (a post filter, for its tab counts)."""
    clauses = study_filter_clauses(filters)
    if filters.types:
        clauses.append({"terms": {f"{FILTER_FACETS_KEY}.dataset_type": filters.types}})
    return clauses


async def allowed_sids(client: Any, studies_index: str, filters: StudyFilters, timeout_seconds: float) -> list[int]:
    """The ids of the published studies matching ``filters``, read from the study index.

    Raises :class:`IndexNotReady` when the study index is missing or holds no published study (nothing would be
    returned, and that is not "nothing matches"), and :class:`TooManyStudies` past ``MAX_ALLOWED_STUDIES``.
    """
    from opensearchpy.exceptions import NotFoundError

    body = {
        "size": 0,
        "query": {"bool": {"filter": study_clauses(filters)}},
        "aggs": {"sids": {"terms": {"field": "sid", "size": MAX_ALLOWED_STUDIES}}},
    }
    try:
        response = await client.search(index=studies_index, body=body, request_timeout=timeout_seconds)
        agg = response["aggregations"]["sids"]
        if agg.get("sum_other_doc_count"):
            raise TooManyStudies(f"more than {MAX_ALLOWED_STUDIES} studies match")
        sids = [int(b["key"]) for b in agg["buckets"]]
        if not sids:
            published = {"query": {"term": {f"{FILTER_FACETS_KEY}.published": 1}}}
            counted = await client.count(index=studies_index, body=published, request_timeout=timeout_seconds)
            if int(counted["count"]) == 0:
                raise IndexNotReady(studies_index)
    except NotFoundError as e:
        raise IndexNotReady(studies_index) from e
    return sids


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


def search_body(job: VariableSearchJob, sids: list[int]) -> dict[str, Any]:
    """The variable query, kept to the variables of ``sids`` (the allowed studies, see :func:`allowed_sids`)."""
    filters: list[dict[str, Any]] = [{"terms": {"sid": sids}}]
    query: dict[str, Any] = (
        {"bool": {"filter": filters}}
        if job.query is None
        else {"bool": {"must": [lexical_query(job.query)], "filter": filters}}
    )
    before = job.cursor is not None and job.cursor.direction is Direction.before
    body: dict[str, Any] = {
        "query": query,
        "sort": sort_clauses(job.sort_by, job.order, reverse=before),
        "_source": ["uid", "sid"],
    }
    if job.cursor is None:
        body["from"] = job.offset
        body["size"] = job.limit
        # A browse is a filter only (nothing is scored), so counting all of it is cheap; a query's matches are
        # counted up to the cap and reported as a lower bound past it.
        body["track_total_hits"] = True if job.query is None else job.total_hits_cap
    else:
        body["size"] = job.limit + 1  # one more than a page: whether another page follows in that direction
        body["search_after"] = list(job.cursor.sort_values)
        body["track_total_hits"] = False  # the total travels in the cursor
    return body


def preference(job: VariableSearchJob) -> str:
    """Routes every page of one search to the same shard copies, so relevance scores (which differ slightly between a
    primary and its replica) stay consistent from page to page."""
    key = json.dumps([job.query, job.filters.model_dump(mode="json"), job.sort_by, job.order], sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _hit(raw: dict[str, Any]) -> VariableHit:
    source = raw["_source"]
    return VariableHit(uid=int(source["uid"]), sid=int(source["sid"]), score=raw.get("_score"))


def _total(hits: dict[str, Any]) -> tuple[int, bool]:
    total = hits["total"]
    if isinstance(total, dict):
        return int(total["value"]), total.get("relation") == "gte"
    return int(total), False


async def search_variables(job: VariableSearchJob) -> VariablePage:
    from opensearchpy.exceptions import NotFoundError

    sids = await allowed_sids(job.client, job.studies_index, job.filters, job.timeout_seconds)
    if not sids:
        # no study matches the filters: no variable can, so the variable index is not asked
        found, lower = (job.cursor.total, job.cursor.total_is_lower_bound) if job.cursor else (0, False)
        return VariablePage(found=found, found_is_lower_bound=lower, offset=job.cursor.position if job.cursor else 0)
    try:
        response = await job.client.search(
            index=job.index,
            body=search_body(job, sids),
            preference=preference(job),
            request_timeout=job.timeout_seconds,
        )
        hits = response["hits"]
        raw: list[dict[str, Any]] = hits["hits"]
        if job.cursor is None:
            found, lower_bound = _total(hits)
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
        found, lower_bound = cursor.total, cursor.total_is_lower_bound
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
        return encode(Cursor(direction, tuple(row["sort"]), position, found, lower_bound))

    return VariablePage(
        found=found,
        found_is_lower_bound=lower_bound,
        offset=offset,
        hits=[_hit(r) for r in rows],
        next_cursor=cursor_from(rows[-1], Direction.after, offset + len(rows)) if has_next and rows else None,
        prev_cursor=cursor_from(rows[0], Direction.before, offset) if has_prev and rows else None,
        took_ms=response.get("took"),
    )
