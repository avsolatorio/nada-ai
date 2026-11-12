"""Standard study search on OpenSearch: the query builders and one executor per search mode.

``EXECUTORS`` is the single list of modes this engine serves; ``GET /info`` derives its capabilities from it, so a
mode is advertised exactly when its executor exists.

* ``browse``: no query, filters and sort only, on the study index.
* ``lexical``: keyword search on the study index. Every matching study is returned and paged, best match first.
* ``semantic``: vector search on the chunk index, collapsed to one hit per study, at most ``semantic_window`` studies.
* ``hybrid``: the semantic studies (at most ``semantic_window``) fused, by rank, with the best ``fusion_window`` keyword
  matches, followed by all the other keyword matches in score order. Any other sort orders the union of the two by
  that sort.

The study index (one document per study) is the source of truth for what a study is: its NADA idno, its type, the
counts. Filters are the flat ``filter_facets.<key>`` fields written at ingest on both indexes (see
``mapping.filter_facets_mapping``).

Facet counts (``facets`` in the request) come from one more request on the study index, run next to the page request:
see :func:`facets_body` and ``docs/studies-search-facets.md``.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from opensearchpy.exceptions import NotFoundError

from nada_ai.app.studies_schemas import (
    EffectiveMode,
    FacetBucket,
    FacetKey,
    FacetRequest,
    FacetValues,
    SortField,
    SortOrder,
    StudyFacets,
    StudyFilters,
    YearRange,
)
from nada_ai.search.backend.opensearch.mapping import FILTER_FACETS_KEY, METADATA_OBJECT_KEY, STUDY_ALL_TEXT_FIELD
from nada_ai.search.backend.opensearch.studies_semantic import (
    Ranked,
    StudyPolicy,
    apply_semantic_policy,
    knn_body,
    parse_semantic,
    rrf_fuse,
)
from nada_ai.search.cursor import Cursor, Direction, SortValueKind, encode
from nada_ai.search.query_heuristics import split_quoted

#: Sort field -> the study-index field it orders by.
SORT_FIELDS: dict[SortField, str] = {
    SortField.title: "title_sort",
    SortField.nation: "nation_sort",
    SortField.year: "year_start",
    SortField.popularity: "total_views",
    SortField.created: "created",
    SortField.changed: "changed",
}

#: After the chosen sort key, ties break by these, then by ``sid``, so paging is deterministic.
_TIE_BREAKERS: tuple[tuple[str, SortOrder], ...] = (
    ("year_start", SortOrder.desc),
    ("title_sort", SortOrder.asc),
    ("sid", SortOrder.asc),
)

#: Distinct dataset types shown as tabs; far above any real catalog.
_TYPE_BUCKETS = 50


# ---------------------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldMap:
    """Where the filterable fields live in a document: the study index keeps them at the root, chunks under
    ``metadata``."""

    facets: str
    sid: str
    created: str


STUDY_FIELDS = FieldMap(FILTER_FACETS_KEY, "sid", "created")
CHUNK_FIELDS = FieldMap(
    f"{METADATA_OBJECT_KEY}.{FILTER_FACETS_KEY}", f"{METADATA_OBJECT_KEY}.sid", f"{METADATA_OBJECT_KEY}.created"
)


#: Facet key -> the ``filter_facets`` field it counts (and its filter matches).
FACET_FIELDS: dict[FacetKey, str] = {
    FacetKey.countries: "countries",
    FacetKey.regions: "regions",
    FacetKey.collections: "repositories",
    FacetKey.form_ids: "formid",
    FacetKey.data_class_ids: "data_class_id",
    FacetKey.tags: "tags",
    FacetKey.databases: "ts_databases",
    FacetKey.years: "years",
}

#: Facet key -> the ``StudyFilters`` field that selects its values (``years`` is a range, not a selection).
_FACET_SELECTION: dict[FacetKey, str] = {
    FacetKey.countries: "countries",
    FacetKey.regions: "regions",
    FacetKey.collections: "collections",
    FacetKey.form_ids: "form_ids",
    FacetKey.data_class_ids: "data_class_ids",
    FacetKey.tags: "tags",
    FacetKey.databases: "databases",
}


def user_facet_field(name: str) -> str:
    """The ``filter_facets`` field of user-defined facet ``name`` (its term ids)."""
    return f"fq_{name}"


def scope_clauses(filters: StudyFilters, fields: FieldMap = STUDY_FIELDS) -> list[dict[str, Any]]:
    """The constraints no facet ignores: published, the active repository, ``sids`` and the created range."""
    clauses: list[dict[str, Any]] = [{"term": {f"{fields.facets}.published": 1}}]
    # repository (the active scope) and collections (the facet) are separate constraints, both against the
    # primary-or-secondary membership list
    if filters.repository:
        clauses.append({"term": {f"{fields.facets}.repositories": filters.repository}})
    if filters.sids:
        clauses.append({"terms": {fields.sid: filters.sids}})
    if filters.created_from is not None or filters.created_to is not None:
        bounds = {
            bound: value
            for bound, value in (("gte", filters.created_from), ("lte", filters.created_to))
            if value is not None
        }
        clauses.append({"range": {fields.created: bounds}})
    return clauses


def facet_clauses(filters: StudyFilters, fields: FieldMap = STUDY_FIELDS) -> dict[str, dict[str, Any]]:
    """The filters a facet can ignore, by the facet that ignores it: a :class:`FacetKey` value, ``fq_<name>`` for a
    user facet, or ``topics`` (no facet counts it, so it always applies)."""

    def facet(key: str) -> str:
        return f"{fields.facets}.{key}"

    clauses: dict[str, dict[str, Any]] = {}
    for key, attr in _FACET_SELECTION.items():
        if values := getattr(filters, attr):
            clauses[key.value] = {"terms": {facet(FACET_FIELDS[key]): values}}
    if filters.year_from is not None or filters.year_to is not None:
        bounds = {
            bound: value for bound, value in (("gte", filters.year_from), ("lte", filters.year_to)) if value is not None
        }
        clauses[FacetKey.years.value] = {"range": {facet(FACET_FIELDS[FacetKey.years]): bounds}}
    if filters.topics:
        clauses["topics"] = {"terms": {facet("topics"): filters.topics}}
    for name, term_ids in (filters.facets or {}).items():
        clauses[user_facet_field(name)] = {"terms": {facet(user_facet_field(name)): term_ids}}
    return clauses


def filter_clauses(filters: StudyFilters, fields: FieldMap = STUDY_FIELDS) -> list[dict[str, Any]]:
    """``bool.filter`` clauses for every filter except ``types`` (see :func:`types_post_filter`).

    Filters combine with AND; values within one key are any-of. Only published studies are ever searched.
    ``fields`` says where the fields live, so the same filters serve the study index and the chunk index.
    """
    return [*scope_clauses(filters, fields), *facet_clauses(filters, fields).values()]


def _types_clause(filters: StudyFilters) -> dict[str, Any] | None:
    return {"terms": {f"{FILTER_FACETS_KEY}.dataset_type": filters.types}} if filters.types else None


def types_post_filter(filters: StudyFilters) -> dict[str, Any] | None:
    """The ``types`` filter, applied as a ``post_filter`` so the per-type counts (tabs) ignore it.

    A post filter narrows the hits and ``hits.total`` but not the aggregations, which is exactly the contract:
    ``found`` honors ``types``; ``search_counts_by_type`` does not.
    """
    return _types_clause(filters)


# ---------------------------------------------------------------------------------------
# Facets
# ---------------------------------------------------------------------------------------


def _facet_aggregation(
    field: str, others: list[dict[str, Any]], selected: list[Any] | None, size: int
) -> dict[str, Any]:
    """The counts of one facet: its values under every other filter, and the selected values (always listed)."""
    aggs: dict[str, Any] = {
        "values": {"terms": {"field": field, "size": size, "order": [{"_count": "desc"}, {"_key": "asc"}]}}
    }
    if selected:
        aggs["selected"] = {"filters": {"filters": {str(v): {"term": {field: v}} for v in selected}}}
    return {"filter": {"bool": {"filter": others}}, "aggs": aggs}


def facets_body(
    match: dict[str, Any] | None, filters: StudyFilters, request: FacetRequest, size: int
) -> dict[str, Any]:
    """The facet counts of a search, as one ``size: 0`` request on the study index.

    ``match`` selects the search's result set before the facet filters (``None`` in a browse). The scope (published,
    repository, ``sids``, created) and ``match`` are the query; each facet's aggregation applies the tab (``types``) and
    every facet filter except its own, so a facet lists the values a user can add to the selection.
    """
    by_key = facet_clauses(filters)
    types = _types_clause(filters)

    def others(own: str) -> list[dict[str, Any]]:
        clauses = [clause for key, clause in by_key.items() if key != own]
        return clauses + [types] if types is not None else clauses

    def field(name: str) -> str:
        return f"{FILTER_FACETS_KEY}.{name}"

    aggs: dict[str, Any] = {}
    for key in request.keys:
        if key is FacetKey.years:
            years = field(FACET_FIELDS[key])
            aggs[f"f_{key.value}"] = {
                "filter": {"bool": {"filter": others(key.value)}},
                "aggs": {"min": {"min": {"field": years}}, "max": {"max": {"field": years}}},
            }
            continue
        selected = getattr(filters, _FACET_SELECTION[key])
        aggs[f"f_{key.value}"] = _facet_aggregation(field(FACET_FIELDS[key]), others(key.value), selected, size)
    for name in request.user:
        own = user_facet_field(name)
        selected = (filters.facets or {}).get(name)
        aggs[f"u_{name}"] = _facet_aggregation(field(own), others(own), selected, size)

    query_filter = scope_clauses(filters)
    if match is not None:
        query_filter.append(match)
    return {"size": 0, "track_total_hits": False, "query": {"bool": {"filter": query_filter}}, "aggs": aggs}


def _facet_values(agg: dict[str, Any]) -> FacetValues:
    values = agg["values"]
    buckets = [FacetBucket(value=str(b["key"]), count=int(b["doc_count"])) for b in values["buckets"]]
    listed = {b.value for b in buckets}
    # a selected value beyond the first ``size`` is added; one that matches nothing is not listed
    for value, bucket in (agg.get("selected") or {}).get("buckets", {}).items():
        if value not in listed and int(bucket["doc_count"]) > 0:
            buckets.append(FacetBucket(value=value, count=int(bucket["doc_count"])))
    return FacetValues(buckets=buckets, more=int(values.get("sum_other_doc_count") or 0) > 0)


def parse_facets(response: dict[str, Any], request: FacetRequest) -> StudyFacets:
    aggs = response["aggregations"]
    facets = StudyFacets()
    for key in request.keys:
        agg = aggs[f"f_{key.value}"]
        if key is FacetKey.years:
            low, high = agg["min"]["value"], agg["max"]["value"]
            facets.years = YearRange(min=None if low is None else int(low), max=None if high is None else int(high))
        else:
            facets.values[key] = _facet_values(agg)
    for name in request.user:
        facets.user[name] = _facet_values(aggs[f"u_{name}"])
    return facets


#: What each sorted field's values are, for checking a browse cursor (``nada_ai.search.cursor``).
_SORT_VALUE_KINDS: dict[str, SortValueKind] = {
    "title_sort": SortValueKind.text,
    "nation_sort": SortValueKind.text,
    "year_start": SortValueKind.number,
    "total_views": SortValueKind.number,
    "created": SortValueKind.number,
    "changed": SortValueKind.number,
    "sid": SortValueKind.id,
}


def _sort_keys(by: SortField, order: SortOrder) -> list[tuple[str, SortOrder]]:
    field_name = SORT_FIELDS[by]
    keys: list[tuple[str, SortOrder]] = [(field_name, order)]
    return keys + [(name, direction) for name, direction in _TIE_BREAKERS if name != field_name]


def sort_clause(by: SortField, order: SortOrder) -> list[dict[str, Any]]:
    """OpenSearch ``sort``: the chosen key, then the deterministic tie-breakers (never the same field twice), ending
    on ``sid`` so that no two studies tie (what a browse cursor needs)."""
    return [{name: {"order": direction.value, "missing": "_last"}} for name, direction in _sort_keys(by, order)]


def browse_sort_value_kinds(by: SortField, order: SortOrder) -> tuple[SortValueKind, ...]:
    """What a browse cursor's sort values must hold for this sort (one kind per sorted field)."""
    return tuple(_SORT_VALUE_KINDS[name] for name, _ in _sort_keys(by, order))


def reverse_sort(clauses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The exact reverse of a ``sort``: every field flipped, and studies with no value moved from the end to the start,
    which is how the page *before* a study is read."""
    flipped = []
    for clause in clauses:
        ((name, spec),) = clause.items()
        flipped.append(
            {
                name: {
                    "order": "asc" if spec["order"] == "desc" else "desc",
                    "missing": "_first" if spec.get("missing") == "_last" else "_last",
                }
            }
        )
    return flipped


def browse_body(
    filters: StudyFilters,
    by: SortField,
    order: SortOrder,
    limit: int,
    offset: int,
    cursor: Cursor | None = None,
) -> dict[str, Any]:
    """A page of the filter-only listing: by ``offset``, or after (before) the study ``cursor`` points at. Every page
    counts exactly (a browse is a filter over the study index: cheap) and carries the per-type counts."""
    sort = sort_clause(by, order)
    body: dict[str, Any] = {
        "track_total_hits": True,
        "_source": ["sid", "idno"],
        "query": {"bool": {"filter": filter_clauses(filters)}},
        "aggs": _type_aggregation(),
    }
    if cursor is None:
        body.update({"from": offset, "size": limit, "sort": sort})
    else:
        body.update(
            {
                "size": limit + 1,  # one more than a page: whether another page follows in that direction
                "search_after": list(cursor.sort_values),
                "sort": reverse_sort(sort) if cursor.direction is Direction.before else sort,
            }
        )
    if (post := types_post_filter(filters)) is not None:
        body["post_filter"] = post
    return body


def _type_aggregation() -> dict[str, Any]:
    """Distinct studies per dataset type. Aggregations run before ``post_filter``, so the counts ignore ``types``."""
    return {"by_type": {"terms": {"field": f"{FILTER_FACETS_KEY}.dataset_type", "size": _TYPE_BUCKETS}}}


# ---------------------------------------------------------------------------------------
# Jobs and pages
# ---------------------------------------------------------------------------------------


class IndexNotReady(Exception):
    """An index does not exist or the study index holds no studies (nothing indexed yet, or being rebuilt)."""


class EmbeddingUnavailable(Exception):
    """The query could not be embedded (no local embedding backend, or the model failed)."""


EmbedFn = Callable[[str], Awaitable[list[float]]]


@dataclass(frozen=True)
class SearchJob:
    """One resolved study search, handed to an executor."""

    client: Any
    index: str  # the study index
    chunk_index: str
    query: str | None
    filters: StudyFilters
    sort_by: SortField
    sort_order: SortOrder
    limit: int
    offset: int
    policy: StudyPolicy
    #: Turns the query into a vector; ``None`` when no local embedding backend exists.
    embed: EmbedFn | None = None
    #: Browse only: the page after (before) the study this points at, in place of ``offset``.
    cursor: Cursor | None = None
    #: Seconds each OpenSearch request may take (setting ``search_timeout_seconds``).
    timeout_seconds: float = 5.0
    #: The facet counts asked for, or ``None``.
    facets: FacetRequest | None = None
    #: The most values one facet lists (setting ``studies_facet_size``).
    facet_size: int = 500


@dataclass
class StudyPage:
    """What an executor returns; the route turns it into the response."""

    found: int
    counts_by_type: dict[str, int]
    hits: list[dict[str, Any]]  # each {"sid", "idno", "score", "matched_by", optional "passages"}
    took_ms: float | None = None
    request_bodies: list[dict[str, Any]] = field(default_factory=list)
    #: Where the page starts in the whole result list, when the executor knows it better than the request (a cursor).
    offset: int | None = None
    next_cursor: str | None = None
    prev_cursor: str | None = None
    facets: StudyFacets | None = None


def _total(response: dict[str, Any]) -> int:
    total = response["hits"]["total"]
    return int(total["value"] if isinstance(total, dict) else total)


def _took(response: dict[str, Any]) -> float:
    return float(response.get("took") or 0)


async def _search(client: Any, index: str, body: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    """One OpenSearch search, abandoned after ``timeout_seconds`` (``opensearchpy.ConnectionTimeout``: the route
    answers ``search_timeout``, so one slow query is not mistaken for an outage)."""
    try:
        return await client.search(index=index, body=body, request_timeout=timeout_seconds)
    except NotFoundError as e:
        raise IndexNotReady(index) from e


@dataclass
class FacetCounts:
    """The facet request of a search: its counts (``None`` when none were asked for), time and body."""

    facets: StudyFacets | None = None
    took_ms: float = 0.0
    request_bodies: list[dict[str, Any]] = field(default_factory=list)

    def into(self, page: StudyPage) -> StudyPage:
        page.facets = self.facets
        page.took_ms = (page.took_ms or 0.0) + self.took_ms
        page.request_bodies = [*page.request_bodies, *self.request_bodies]
        return page


async def facet_counts(job: SearchJob, match: dict[str, Any] | None) -> FacetCounts:
    """The facet counts of the result set ``match`` selects (see :func:`facets_body`), when the job asks for them."""
    if job.facets is None:
        return FacetCounts()
    body = facets_body(match, job.filters, job.facets, job.facet_size)
    response = await _search(job.client, job.index, body, job.timeout_seconds)
    return FacetCounts(parse_facets(response, job.facets), _took(response), [body])


def _semantic_match(sids: list[int]) -> dict[str, Any]:
    return {"terms": {"sid": sids}}


def _hybrid_match(text: str, sids: list[int]) -> dict[str, Any]:
    return {"bool": {"should": [lexical_query(text), _semantic_match(sids)], "minimum_should_match": 1}}


# ---------------------------------------------------------------------------------------
# browse
# ---------------------------------------------------------------------------------------


def _by_type(response: dict[str, Any]) -> dict[str, int]:
    return {str(b["key"]): int(b["doc_count"]) for b in response["aggregations"]["by_type"]["buckets"]}


async def _raise_if_index_empty(job: SearchJob) -> None:
    """Zero results is either "nothing matches" or "nothing is indexed"; only the second is an error. An index deleted
    since the search (e.g. by a recreate) is not ready either."""
    try:
        response = await job.client.count(index=job.index, request_timeout=job.timeout_seconds)
    except NotFoundError as e:
        raise IndexNotReady(job.index) from e
    if int(response["count"]) == 0:
        raise IndexNotReady(job.index)


async def browse(job: SearchJob) -> StudyPage:
    """Filter-only listing: exact ``found``, exact per-type counts, deterministic order, and the cursors of the next
    and previous pages (a browse pages past ``max_offset`` with them)."""
    body = browse_body(job.filters, job.sort_by, job.sort_order, job.limit, job.offset, job.cursor)
    response, facets = await asyncio.gather(
        _search(job.client, job.index, body, job.timeout_seconds), facet_counts(job, None)
    )
    found = _total(response)
    if found == 0:
        await _raise_if_index_empty(job)
    raw = response["hits"]["hits"]

    cursor = job.cursor
    if cursor is None:
        rows, offset = raw, job.offset
        has_prev, has_next = offset > 0, offset + len(rows) < found
    else:
        more = len(raw) > job.limit
        if cursor.direction is Direction.after:
            rows, offset = raw[: job.limit], cursor.position
            has_prev, has_next = cursor.position > 0, more
        else:
            rows = list(reversed(raw[: job.limit]))
            offset = max(cursor.position - len(rows), 0) if more else 0  # nothing more before: the first page
            has_prev, has_next = more, True

    def cursor_from(row: dict[str, Any], direction: Direction, position: int) -> str:
        return encode(Cursor(direction, tuple(row["sort"]), position, found, False))

    page = StudyPage(
        found=found,
        counts_by_type=_by_type(response),
        hits=[
            {"sid": int(h["_source"]["sid"]), "idno": h["_source"]["idno"], "score": None, "matched_by": []}
            for h in rows
        ],
        took_ms=_took(response),
        request_bodies=[body],
        offset=offset,
        next_cursor=cursor_from(rows[-1], Direction.after, offset + len(rows)) if has_next and rows else None,
        prev_cursor=cursor_from(rows[0], Direction.before, offset) if has_prev and rows else None,
    )
    return facets.into(page)


# ---------------------------------------------------------------------------------------
# Keyword query
# ---------------------------------------------------------------------------------------

# Fields searched by keyword search and their boosts. What identifies a study (idno, title, country, producer) and what
# describes it (abstract) count most; the two long blobs (``keywords`` is NADA's combined metadata text, ``var_keywords``
# the variable names, labels and questions) count least, because a word found somewhere in thousands of characters says
# little about what the study is. (NADA's own search boosted the blobs 10 and 15 and the abstract 1; on the golden
# queries this ranks a little better, and it stops a study whose variable labels happen to contain the query words from
# beating one that describes them.)
LEXICAL_FIELDS = (
    "idno.text^60",
    "title^40",
    "nation^30",
    "authoring_entity^10",
    "abstract^10",
    "keywords",
    "methodology",
    "var_keywords",
)

#: A query of two or more words also scores the words as a phrase (this far apart at most), in any field: a study that
#: says "foreign direct investment" outranks one that mentions the three words in unrelated places. It only adds to the
#: score of studies the match rules already accept, so it never changes which studies match.
PHRASE_SLOP = 2
PHRASE_BOOST = 2

#: Weight of a match in ``all_text``, below every named field: a study that says it in its title or abstract comes first.
ALL_TEXT_BOOST = 1


def _words_query(text: str) -> dict[str, Any]:
    """Loose keyword match over the study text fields, or every word somewhere in ``all_text``; a phrase bonus for
    two or more words.

    ``minimum_should_match: 2<75%``: one or two terms must all match, otherwise at least 75% of them (per field, as in
    NADA's own search); the terms may be anywhere in the field. ``fuzziness: AUTO:5,9`` forgives one typo in words of
    5-8 letters and two in longer ones, and none in shorter words (with ``AUTO``, 3-4 letter words and long
    variable-label fields matched unrelated words); ``prefix_length: 2`` keeps the first two letters exact. Fields are
    scored separately and the scores add up (``most_fields``).

    ``all_text`` (every other metadata value) is matched strictly -- all words, no typos -- and weighs least: loose
    matching on a field that large would accept most studies for most queries.
    """
    loose: dict[str, Any] = {
        "multi_match": {
            "query": text,
            "fields": list(LEXICAL_FIELDS),
            "type": "most_fields",
            "minimum_should_match": "2<75%",
            "fuzziness": "AUTO:5,9",
            "prefix_length": 2,
        }
    }
    everywhere = {"match": {STUDY_ALL_TEXT_FIELD: {"query": text, "operator": "and", "boost": ALL_TEXT_BOOST}}}
    match: dict[str, Any] = {"bool": {"should": [loose, everywhere], "minimum_should_match": 1}}
    if len(text.split()) < 2:
        return match
    phrase = {
        "multi_match": {
            "query": text,
            "type": "phrase",
            "fields": list(LEXICAL_FIELDS),
            "slop": PHRASE_SLOP,
            "boost": PHRASE_BOOST,
        }
    }
    return {"bool": {"must": [match], "should": [phrase]}}


def _exact_phrase_query(phrase: str) -> dict[str, Any]:
    """``phrase`` word for word (case and accents aside), in any study text field or in ``all_text``."""
    return {
        "multi_match": {
            "query": phrase,
            "type": "phrase",
            "slop": 0,
            "fields": [*LEXICAL_FIELDS, f"{STUDY_ALL_TEXT_FIELD}^{ALL_TEXT_BOOST}"],
        }
    }


def lexical_query(text: str) -> dict[str, Any]:
    """Keyword match over the study's text: every ``"…"`` phrase must appear exactly, the other words match as in
    :func:`_words_query`.

    Every study this matches is a keyword match: there is no score cutoff. The best matches (title, idno) simply
    score highest, and a study that mentions the words only in a low-weight field comes after them.
    """
    parts = split_quoted(text)
    if not parts.phrases:
        return _words_query(parts.rest or text)
    must = [_exact_phrase_query(phrase) for phrase in parts.phrases]
    if parts.rest:
        must.append(_words_query(parts.rest))
    return {"bool": {"must": must}}


def relevance_sort(order: SortOrder) -> list[dict[str, Any]]:
    """Order by keyword score, ties by ``sid`` so that every page of a result is stable."""
    return [{"_score": {"order": order.value}}, {"sid": {"order": "asc"}}]


def keyword_body(
    text: str,
    filters: StudyFilters,
    *,
    sort: list[dict[str, Any]],
    limit: int,
    offset: int,
    exclude_sids: list[int] | None = None,
    union_sids: list[int] | None = None,
) -> dict[str, Any]:
    """The keyword matches that pass ``filters``, paged, with per-type counts over all of them.

    ``exclude_sids`` leaves studies out (they are shown elsewhere in the result); ``union_sids`` adds studies that
    match without the keyword. The ``types`` filter narrows the hits and ``hits.total`` but not the counts.
    """
    keyword: dict[str, Any] = lexical_query(text)
    if union_sids:
        keyword = {"bool": {"should": [keyword, {"terms": {"sid": union_sids}}], "minimum_should_match": 1}}
    query: dict[str, Any] = {"bool": {"must": [keyword], "filter": filter_clauses(filters)}}
    if exclude_sids:
        query["bool"]["must_not"] = [{"terms": {"sid": exclude_sids}}]
    body: dict[str, Any] = {
        "from": offset,
        "size": limit,
        "track_total_hits": True,
        "_source": ["sid", "idno", "title", f"{FILTER_FACETS_KEY}.dataset_type"],
        "query": query,
        "sort": sort,
        "aggs": _type_aggregation(),
    }
    if (post := types_post_filter(filters)) is not None:
        body["post_filter"] = post
    return body


#: Stopwords the title-completeness check below ignores when reading what a query asks for (NADA's own English
#: stopword list, application/config/noise_words.php): a query word does not have to appear in a title if nothing
#: would have made that word searchable in the database's own search either.
_QUERY_STOPWORDS = frozenset(
    """
    a able about across after all almost also am among an and any are as at be because been but by can cannot
    could dear did do does either else ever every for from get got had has have he her hers him his how however
    i if in into is it its just least let like likely may me might most must my neither no nor not of off often
    on only or other our own rather said say says she should since so some than that the their them then there
    these they this tis to too twas us wants was we were what when where which while who whom why will with
    would yet you your
    """.split()
)

_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

#: A title match is promoted (see ``hybrid``) only for a query of at least this many real words. Study titles are
#: long: one or two words in a title single nothing out (163 of 1,383 local titles contain "census"), and promoting
#: every such title would push every study only the semantic leg found off the first pages. On the golden queries
#: this costs hybrid nDCG@10 0.839 -> 0.832 ("population census" and "census 2011" lose the promotion), accepted.
#: No cap on how many are promoted: a cap (5 or 10) cost 0.839 -> 0.828/0.831.
_TITLE_PROMOTION_MIN_WORDS = 3


def _title_words(text: str) -> set[str]:
    """The words of a title or query, lowercased and without accents, as the index's ``nada_text`` analyzer reads
    them (no stemming there either -- good enough for a set-membership check; the analyzed fields do the real
    matching)."""
    folded = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    return set(_WORD.findall(folded))


def _title_is_complete_match(query: str, title: str) -> bool:
    """Whether every real word of ``query`` (stopwords aside) appears somewhere in ``title``, in any order -- for a
    query of at least ``_TITLE_PROMOTION_MIN_WORDS`` real words; a shorter one never matches.

    Used to keep a study whose title plainly says what was searched from being outranked, in hybrid mode, by
    studies the fusion happens to rank higher (see ``hybrid``): rank fusion only counts position, not how decisive
    a keyword match is, so a title that names every word of the query is a stronger signal than a fused rank.
    """
    needed = _title_words(query) - _QUERY_STOPWORDS
    return len(needed) >= _TITLE_PROMOTION_MIN_WORDS and needed <= _title_words(title)


def _dataset_type(hit: dict[str, Any]) -> str:
    values = (hit["_source"].get(FILTER_FACETS_KEY) or {}).get("dataset_type") or []
    return str(values[0]) if values else "unknown"


async def exact_idno_match(job: SearchJob) -> StudyPage | None:
    """A study whose idno exactly matches the query (case- and accent-insensitively, via the ``idno`` field's
    normalizer), within the active filters -- or ``None`` when the query is not a single token, or no study
    matches. A single token in quotes (``"ABC-2020"``) counts as that token.

    Checked before every relevance search: nothing a scored search does is as precise as this, and it is what the
    database's own idno/alias lookup already gives NADA on every other engine. ``idno`` matches are unique in NADA's
    catalog in practice, but this returns every match rather than assuming exactly one.
    """
    assert job.query is not None
    token = job.query.strip()
    parts = split_quoted(token)
    if parts.phrases:
        if parts.rest or len(parts.phrases) != 1:
            return None
        token = parts.phrases[0]
    if not token or len(token.split()) != 1:
        return None

    filters = list(filter_clauses(job.filters))
    if job.filters.types:
        filters.append({"terms": {f"{FILTER_FACETS_KEY}.dataset_type": job.filters.types}})
    match = {"term": {"idno": token}}
    body = {
        "size": max(1, job.offset + job.limit),
        "track_total_hits": True,
        "_source": ["sid", "idno", f"{FILTER_FACETS_KEY}.dataset_type"],
        "query": {"bool": {"filter": [*filters, match]}},
        "sort": [{"sid": {"order": "asc"}}],
    }
    response = await _search(job.client, job.index, body, job.timeout_seconds)
    hits = response["hits"]["hits"]
    if not hits:
        return None
    facets = await facet_counts(job, match)

    counts: dict[str, int] = {}
    for h in hits:
        t = _dataset_type(h)
        counts[t] = counts.get(t, 0) + 1
    page_hits = [
        {"sid": int(h["_source"]["sid"]), "idno": h["_source"]["idno"], "score": None, "matched_by": ["idno"]}
        for h in hits[job.offset : job.offset + job.limit]
    ]
    page = StudyPage(
        found=_total(response), counts_by_type=counts, hits=page_hits, took_ms=_took(response), request_bodies=[body]
    )
    return facets.into(page)


def _keyword_hit(hit: dict[str, Any]) -> dict[str, Any]:
    score = hit.get("_score")
    return {
        "sid": int(hit["_source"]["sid"]),
        "idno": hit["_source"]["idno"],
        "score": None if score is None else float(score),
        "matched_by": ["lexical"],
    }


async def lexical(job: SearchJob) -> StudyPage:
    """Keyword search: every matching study, best match first (or in the chosen sort), paged."""
    assert job.query is not None
    sort = (
        relevance_sort(job.sort_order)
        if job.sort_by is SortField.relevance
        else sort_clause(job.sort_by, job.sort_order)
    )
    body = keyword_body(job.query, job.filters, sort=sort, limit=job.limit, offset=job.offset)
    response, facets = await asyncio.gather(
        _search(job.client, job.index, body, job.timeout_seconds), facet_counts(job, lexical_query(job.query))
    )
    found = _total(response)
    if found == 0:
        await _raise_if_index_empty(job)
    page = StudyPage(
        found=found,
        counts_by_type=_by_type(response),
        hits=[_keyword_hit(h) for h in response["hits"]["hits"]],
        took_ms=_took(response),
        request_bodies=[body],
    )
    return facets.into(page)


# ---------------------------------------------------------------------------------------
# Semantic
# ---------------------------------------------------------------------------------------


def _hit(ranked: Ranked) -> dict[str, Any]:
    hit: dict[str, Any] = {
        "sid": ranked.sid,
        "idno": ranked.idno,
        "score": ranked.score,
        "matched_by": ranked.matched_by,
    }
    if ranked.passages:
        hit["passages"] = ranked.passages
    return hit


def _debug_body(body: dict[str, Any]) -> dict[str, Any]:
    """The vector request without the vector itself (hundreds of numbers no one reads)."""
    knn = next(iter(body["query"]["knn"].values()))
    shown = {**knn, "vector": f"<{len(knn['vector'])}-dimensional query vector>"}
    return {**body, "query": {"knn": {next(iter(body["query"]["knn"])): shown}}}


async def _embed(job: SearchJob) -> list[float]:
    assert job.query is not None
    if job.embed is None:
        raise EmbeddingUnavailable("semantic search needs the local embedding backend")
    return await job.embed(job.query)


async def _enrich(job: SearchJob, ranked: list[Ranked]) -> tuple[list[Ranked], float, list[dict[str, Any]]]:
    """Look up studies that only the vector search found in the study index, with the full filters.

    The study index is the source of truth: it supplies NADA's own idno (a chunk carries the record's schema idno,
    which can differ) and the type, and it drops studies that are no longer indexed or that a filter excludes.
    """
    missing = [r.sid for r in ranked if r.idno is None]
    if not missing:
        return ranked, 0.0, []
    body = {
        "size": len(missing),
        "_source": ["sid", "idno", f"{FILTER_FACETS_KEY}.dataset_type"],
        "query": {"bool": {"filter": [*filter_clauses(job.filters), {"terms": {"sid": missing}}]}},
    }
    response = await _search(job.client, job.index, body, job.timeout_seconds)
    known = {int(h["_source"]["sid"]): (h["_source"]["idno"], _dataset_type(h)) for h in response["hits"]["hits"]}
    kept: list[Ranked] = []
    for r in ranked:
        if r.idno is None:
            if r.sid not in known:
                continue
            r.idno, r.dataset_type = known[r.sid]
        kept.append(r)
    return kept, _took(response), [body]


async def _semantic_block(job: SearchJob) -> tuple[list[Ranked], float, list[dict[str, Any]]]:
    """The studies the vector search finds (at most ``semantic_window``, after the floor and the relative cutoff),
    best first, each with NADA's idno and type. Raises ``EmbeddingUnavailable`` before any search runs."""
    vector = await _embed(job)
    body = knn_body(vector, filter_clauses(job.filters, CHUNK_FIELDS), job.policy)
    response = await _search(job.client, job.chunk_index, body, job.timeout_seconds)
    hits = apply_semantic_policy(parse_semantic(response, job.policy), job.policy)
    hits, took_lookup, lookup = await _enrich(job, hits)
    return hits, _took(response) + took_lookup, [_debug_body(body), *lookup]


async def semantic(job: SearchJob) -> StudyPage:
    """Vector search only: the semantic block, and nothing else. Any sort but relevance re-sorts the block."""
    block, took, bodies = await _semantic_block(job)
    if not block:
        await _raise_if_index_empty(job)
        return (await facet_counts(job, _semantic_match([]))).into(StudyPage(0, {}, [], took, bodies))

    types = job.filters.types
    counts = dict(Counter(r.dataset_type or "unknown" for r in block))
    by_sid = {r.sid: r for r in block}
    if job.sort_by is SortField.relevance:
        facets = await facet_counts(job, _semantic_match(list(by_sid)))
        ordered = block if job.sort_order is SortOrder.desc else block[::-1]
        kept = [r for r in ordered if not types or (r.dataset_type or "unknown") in types]
        page = StudyPage(len(kept), counts, [_hit(r) for r in kept[job.offset : job.offset + job.limit]], took, bodies)
        return facets.into(page)

    # the sids pin the set; `types` narrows `found` only, exactly as in browse
    resorted, facets = await asyncio.gather(
        browse(replace(job, query=None, filters=StudyFilters(sids=list(by_sid), types=types), facets=None)),
        facet_counts(job, _semantic_match(list(by_sid))),
    )
    page = StudyPage(
        resorted.found,
        resorted.counts_by_type,
        [_hit(by_sid[h["sid"]]) for h in resorted.hits],
        took + (resorted.took_ms or 0.0),
        [*bodies, *resorted.request_bodies],
    )
    return facets.into(page)


# ---------------------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------------------


async def _keyword_matches_among(job: SearchJob, sids: list[int]) -> tuple[set[int], float, dict[str, Any]]:
    """Which of ``sids`` are also keyword matches (with the filters)."""
    assert job.query is not None
    body = {
        "size": len(sids),
        "track_total_hits": False,
        "_source": ["sid"],
        "query": {
            "bool": {
                "must": [lexical_query(job.query)],
                "filter": [*filter_clauses(job.filters), {"terms": {"sid": sids}}],
            }
        },
    }
    response = await _search(job.client, job.index, body, job.timeout_seconds)
    return {int(h["_source"]["sid"]) for h in response["hits"]["hits"]}, _took(response), body


def _keyword_ranked(hit: dict[str, Any]) -> Ranked:
    return Ranked(
        sid=int(hit["_source"]["sid"]),
        score=float(hit["_score"]),
        matched_by=["lexical"],
        idno=hit["_source"]["idno"],
        dataset_type=_dataset_type(hit),
    )


def _hybrid_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """A hybrid hit carries no score: the fused head and the keyword matches after it are ordered by different rules,
    so their scores could not be compared."""
    return {**hit, "score": None}


def _fused_hit(ranked: Ranked, keyword_matches: set[int]) -> dict[str, Any]:
    """A hit of the fused head; a semantic study that the keyword search also matches (beyond the fusion window)
    says so."""
    hit = _hit(ranked)
    if ranked.sid in keyword_matches and "lexical" not in hit["matched_by"]:
        hit["matched_by"] = ["lexical", *hit["matched_by"]]
    return _hybrid_hit(hit)


async def hybrid(job: SearchJob) -> StudyPage:
    """The semantic studies and the keyword matches, fused, then every other keyword match.

    With a relevance sort, the best ``fusion_window`` keyword matches and the semantic studies (at most
    ``semantic_window``) are fused by rank: a study in both comes first, and the two lists otherwise alternate, so an
    exact keyword match is never buried under related studies. The keyword matches beyond the window follow, best
    first and paged by OpenSearch. ``found`` is the fused head plus those matches, so nothing is cut, and the counts by
    type are the keyword counts plus the head's. With any other sort the union of the semantic studies and the keyword
    matches is ordered by that sort, so ``found`` is the same whatever the sort.
    """
    assert job.query is not None
    block, took, bodies = await _semantic_block(job)  # fail before any keyword search runs
    if not block:
        page = await lexical(job)
        page.took_ms = (page.took_ms or 0.0) + took
        page.request_bodies = [*bodies, *page.request_bodies]
        return page

    block_ids = [r.sid for r in block]
    by_sid = {r.sid: r for r in block}
    types = job.filters.types

    facet_task = facet_counts(job, _hybrid_match(job.query, block_ids))

    if job.sort_by is not SortField.relevance:
        body = keyword_body(
            job.query,
            job.filters,
            sort=sort_clause(job.sort_by, job.sort_order),
            limit=job.limit,
            offset=job.offset,
            union_sids=block_ids,
        )
        (agree, took_agree, body_agree), response, facets = await asyncio.gather(
            _keyword_matches_among(job, block_ids),
            _search(job.client, job.index, body, job.timeout_seconds),
            facet_task,
        )
        return facets.into(
            StudyPage(
                found=_total(response),
                counts_by_type=_by_type(response),
                hits=[
                    _fused_hit(by_sid[sid], agree)
                    if (sid := int(h["_source"]["sid"])) in by_sid
                    else _hybrid_hit(_keyword_hit(h))
                    for h in response["hits"]["hits"]
                ],
                took_ms=took + took_agree + _took(response),
                request_bodies=[*bodies, body_agree, body],
            )
        )

    # the head: the best keyword matches over ALL types (the tab never changes the order), fused with the semantic studies
    head_body = keyword_body(
        job.query,
        job.filters.model_copy(update={"types": None}),
        sort=relevance_sort(SortOrder.desc),
        limit=job.policy.fusion_window,
        offset=0,
    )
    (agree, took_agree, body_agree), head_response, facets = await asyncio.gather(
        _keyword_matches_among(job, block_ids),
        _search(job.client, job.index, head_body, job.timeout_seconds),
        facet_task,
    )
    keyword_head = [_keyword_ranked(h) for h in head_response["hits"]["hits"]]
    fused = rrf_fuse(keyword_head, block)

    # A study whose title names every word of the query is put first, ahead of the fused order: rank fusion only
    # counts position, and a related study the semantic leg also found can otherwise outrank a study that plainly
    # is the answer (a title match on all 3 words beat only 1 of the 4 studies above it in the fused order).
    titles = {int(h["_source"]["sid"]): h["_source"].get("title", "") for h in head_response["hits"]["hits"]}
    complete = [r for r in fused if _title_is_complete_match(job.query, titles.get(r.sid, ""))]
    if complete:
        complete_ids = {r.sid for r in complete}
        fused = complete + [r for r in fused if r.sid not in complete_ids]

    shown = [r for r in fused if not types or (r.dataset_type or "unknown") in types]
    descending = job.sort_order is SortOrder.desc

    # The fused head sits at the start (best first) or the end (reversed) of the ordered result; the keyword matches
    # without it fill the rest. One keyword request supplies its rows, its total and the per-type counts.
    if descending:
        from_head = max(0, min(job.limit, len(shown) - job.offset))
        tail_offset, tail_rows = max(0, job.offset - len(shown)), job.limit - from_head
    else:
        tail_offset, tail_rows = job.offset, job.limit
    body = keyword_body(
        job.query,
        job.filters,
        sort=relevance_sort(job.sort_order),
        limit=max(1, tail_rows),  # a page inside the head still needs the total and the counts
        offset=tail_offset,
        exclude_sids=[r.sid for r in fused],
    )
    response = await _search(job.client, job.index, body, job.timeout_seconds)
    tail_found = _total(response)
    tail_hits = [_hybrid_hit(_keyword_hit(h)) for h in response["hits"]["hits"]] if tail_rows > 0 else []

    if descending:
        hits = [_fused_hit(r, agree) for r in shown[job.offset : job.offset + from_head]] + tail_hits
    else:
        need = job.limit - len(tail_hits)
        start = max(0, job.offset - tail_found)
        hits = tail_hits + [_fused_hit(r, agree) for r in shown[::-1][start : start + need]]

    counts = _by_type(response)
    for r in fused:
        counts[r.dataset_type or "unknown"] = counts.get(r.dataset_type or "unknown", 0) + 1
    page = StudyPage(
        found=len(shown) + tail_found,
        counts_by_type=counts,
        hits=hits,
        took_ms=took + took_agree + _took(head_response) + _took(response),
        request_bodies=[*bodies, body_agree, head_body, body],
    )
    return facets.into(page)


#: One executor per effective mode this engine serves.
EXECUTORS: dict[EffectiveMode, Callable[[SearchJob], Awaitable[StudyPage]]] = {
    EffectiveMode.browse: browse,
    EffectiveMode.lexical: lexical,
    EffectiveMode.semantic: semantic,
    EffectiveMode.hybrid: hybrid,
}

OPENSEARCH_STUDY_MODES: frozenset[str] = frozenset(mode.value for mode in EXECUTORS)
