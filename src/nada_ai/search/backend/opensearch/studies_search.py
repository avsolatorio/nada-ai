"""Standard study search on OpenSearch: the query builders and one executor per search mode.

``EXECUTORS`` is the single list of modes this engine serves; ``GET /info`` derives its capabilities from it, so a
mode is advertised exactly when its executor exists.

* ``browse``: no query, filters and sort only, on the study index.
* ``lexical``: keyword search on the study index.
* ``semantic``: vector search on the chunk index, collapsed to one hit per study.
* ``hybrid``: both legs, fused by rank.

The study index (one document per study) is the source of truth for what a study is: its NADA idno, its type, the
counts. Filters are the flat ``filter_facets.<key>`` fields written at ingest on both indexes (see
``mapping.filter_facets_mapping``).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from opensearchpy.exceptions import NotFoundError

from nada_ai.app.studies_schemas import (
    EffectiveMode,
    SortField,
    SortOrder,
    StudyFilters,
)
from nada_ai.search.backend.opensearch.mapping import FILTER_FACETS_KEY, METADATA_OBJECT_KEY
from nada_ai.search.backend.opensearch.studies_semantic import (
    Ranked,
    StudyPolicy,
    apply_lexical_cutoff,
    apply_semantic_policy,
    knn_body,
    parse_semantic,
    rrf_fuse,
)

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


def filter_clauses(filters: StudyFilters, fields: FieldMap = STUDY_FIELDS) -> list[dict[str, Any]]:
    """``bool.filter`` clauses for every filter except ``types`` (see :func:`types_post_filter`).

    Filters combine with AND; values within one key are any-of. Only published studies are ever searched.
    ``fields`` says where the fields live, so the same filters serve the study index and the chunk index.
    """

    def facet(key: str) -> str:
        return f"{fields.facets}.{key}"

    clauses: list[dict[str, Any]] = [{"term": {facet("published"): 1}}]
    if filters.countries:
        clauses.append({"terms": {facet("countries"): filters.countries}})
    if filters.year_from is not None or filters.year_to is not None:
        bounds = {
            bound: value for bound, value in (("gte", filters.year_from), ("lte", filters.year_to)) if value is not None
        }
        clauses.append({"range": {facet("years"): bounds}})
    # repository (the active scope) and collections (the facet) are separate constraints, both against the
    # primary-or-secondary membership list
    if filters.repository:
        clauses.append({"term": {facet("repositories"): filters.repository}})
    if filters.collections:
        clauses.append({"terms": {facet("repositories"): filters.collections}})
    if filters.form_ids:
        clauses.append({"terms": {facet("formid"): filters.form_ids}})
    if filters.data_class_ids:
        clauses.append({"terms": {facet("data_class_id"): filters.data_class_ids}})
    if filters.tags:
        clauses.append({"terms": {facet("tags"): filters.tags}})
    for name, term_ids in (filters.facets or {}).items():
        clauses.append({"terms": {facet(f"fq_{name}"): term_ids}})
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


def types_post_filter(filters: StudyFilters) -> dict[str, Any] | None:
    """The ``types`` filter, applied as a ``post_filter`` so the per-type counts (tabs) ignore it.

    A post filter narrows the hits and ``hits.total`` but not the aggregations, which is exactly the contract:
    ``found`` honors ``types``; ``search_counts_by_type`` does not.
    """
    return {"terms": {f"{FILTER_FACETS_KEY}.dataset_type": filters.types}} if filters.types else None


def sort_clause(by: SortField, order: SortOrder) -> list[dict[str, Any]]:
    """OpenSearch ``sort``: the chosen key, then the deterministic tie-breakers (never the same field twice)."""
    field_name = SORT_FIELDS[by]
    keys: list[tuple[str, SortOrder]] = [(field_name, order)]
    keys += [(name, direction) for name, direction in _TIE_BREAKERS if name != field_name]
    return [{name: {"order": direction.value, "missing": "_last"}} for name, direction in keys]


def browse_body(filters: StudyFilters, by: SortField, order: SortOrder, limit: int, offset: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "from": offset,
        "size": limit,
        "track_total_hits": True,
        "_source": ["sid", "idno"],
        "query": {"bool": {"filter": filter_clauses(filters)}},
        "sort": sort_clause(by, order),
        "aggs": {"by_type": {"terms": {"field": f"{FILTER_FACETS_KEY}.dataset_type", "size": _TYPE_BUCKETS}}},
    }
    if (post := types_post_filter(filters)) is not None:
        body["post_filter"] = post
    return body


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
    #: The most studies a relevance search makes pageable (``settings.studies_result_cap``).
    result_cap: int
    policy: StudyPolicy
    #: Turns the query into a vector; ``None`` when no local embedding backend exists.
    embed: EmbedFn | None = None


@dataclass
class StudyPage:
    """What an executor returns; the route turns it into the response."""

    found: int
    counts_by_type: dict[str, int]
    hits: list[dict[str, Any]]  # each {"sid", "idno", "score", "matched_by", optional "passages"}
    took_ms: float | None = None
    request_bodies: list[dict[str, Any]] = field(default_factory=list)
    #: True when more studies matched than ``result_cap`` allows; ``result_cap`` is set for relevance searches only.
    truncated: bool = False
    result_cap: int | None = None


def _total(response: dict[str, Any]) -> int:
    total = response["hits"]["total"]
    return int(total["value"] if isinstance(total, dict) else total)


def _took(response: dict[str, Any]) -> float:
    return float(response.get("took") or 0)


async def _search(client: Any, index: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        return await client.search(index=index, body=body)
    except NotFoundError as e:
        raise IndexNotReady(index) from e


# ---------------------------------------------------------------------------------------
# browse
# ---------------------------------------------------------------------------------------


async def browse(job: SearchJob) -> StudyPage:
    """Filter-only listing: exact ``found``, exact per-type counts, deterministic order."""
    body = browse_body(job.filters, job.sort_by, job.sort_order, job.limit, job.offset)
    response = await _search(job.client, job.index, body)
    found = _total(response)
    # zero results is either "nothing matches" or "nothing is indexed"; only the second is an error
    if found == 0 and int((await job.client.count(index=job.index))["count"]) == 0:
        raise IndexNotReady(job.index)
    return StudyPage(
        found=found,
        counts_by_type={
            str(bucket["key"]): int(bucket["doc_count"]) for bucket in response["aggregations"]["by_type"]["buckets"]
        },
        hits=[
            {"sid": int(h["_source"]["sid"]), "idno": h["_source"]["idno"], "score": None, "matched_by": []}
            for h in response["hits"]["hits"]
        ],
        took_ms=_took(response),
        request_bodies=[body],
    )


# ---------------------------------------------------------------------------------------
# Relevance searches: the shared tail
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


async def _page(job: SearchJob, ranked: list[Ranked], *, truncated: bool, took: float, bodies: list[dict]) -> StudyPage:
    """Turn a best-first ranking into the page: the cut set, the counts, the types filter, the sort, the paging.

    The cut set is the best ``result_cap`` studies across ALL types, so choosing a tab (``types``) never changes
    which studies are relevant, and ``search_counts_by_type`` describes the cut set. Sorting by relevance orders that
    set by score; any other sort re-sorts the same set inside OpenSearch.
    """
    cut = ranked[: job.result_cap]
    truncated = truncated or len(ranked) > job.result_cap

    if not cut:
        if int((await job.client.count(index=job.index))["count"]) == 0:
            raise IndexNotReady(job.index)
        return StudyPage(0, {}, [], took, bodies, truncated=False, result_cap=job.result_cap)

    types = job.filters.types
    if job.sort_by is SortField.relevance:
        counts = dict(Counter(r.dataset_type or "unknown" for r in cut))
        ordered = cut if job.sort_order is SortOrder.desc else cut[::-1]
        kept = [r for r in ordered if not types or (r.dataset_type or "unknown") in types]
        page_hits = kept[job.offset : job.offset + job.limit]
        found = len(kept)
    else:
        # the sids pin the set; `types` narrows `found` only, exactly as in browse
        by_sid = {r.sid: r for r in cut}
        resorted = await browse(replace(job, query=None, filters=StudyFilters(sids=list(by_sid), types=types)))
        counts, found = resorted.counts_by_type, resorted.found
        took += resorted.took_ms or 0.0
        bodies = [*bodies, *resorted.request_bodies]
        page_hits = [by_sid[h["sid"]] for h in resorted.hits]

    return StudyPage(
        found, counts, [_hit(r) for r in page_hits], took, bodies, truncated=truncated, result_cap=job.result_cap
    )


async def _enrich(job: SearchJob, ranked: list[Ranked]) -> tuple[list[Ranked], float, list[dict[str, Any]]]:
    """Look up studies that only the vector leg found in the study index, with the full filters.

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
    response = await _search(job.client, job.index, body)
    known = {int(h["_source"]["sid"]): (h["_source"]["idno"], _dataset_type(h)) for h in response["hits"]["hits"]}
    kept: list[Ranked] = []
    for r in ranked:
        if r.idno is None:
            if r.sid not in known:
                continue
            r.idno, r.dataset_type = known[r.sid]
        kept.append(r)
    return kept, _took(response), [body]


# ---------------------------------------------------------------------------------------
# lexical
# ---------------------------------------------------------------------------------------

# Fields searched by keyword search and their boosts: the ones NADA's own search has always used.
LEXICAL_FIELDS = (
    "idno.text^60",
    "title^40",
    "nation^30",
    "authoring_entity^10",
    "keywords^10",
    "abstract",
    "methodology",
    "var_keywords^15",
)


def lexical_query(text: str) -> dict[str, Any]:
    """Keyword match over the study text fields.

    ``minimum_should_match: 2<75%``: one or two terms must all match, otherwise at least 75% of them (per field, as in
    NADA's own search). ``fuzziness: AUTO:5,9`` forgives one typo in words of 5-8 letters and two in longer ones,
    and none in shorter words (with ``AUTO``, 3-4 letter words and long variable-label fields matched unrelated
    words); ``prefix_length: 2`` keeps the first two letters exact.
    """
    return {
        "multi_match": {
            "query": text,
            "fields": list(LEXICAL_FIELDS),
            "type": "most_fields",
            "minimum_should_match": "2<75%",
            "fuzziness": "AUTO:5,9",
            "prefix_length": 2,
        }
    }


def lexical_body(text: str, filters: StudyFilters, size: int) -> dict[str, Any]:
    """The best ``size`` keyword matches across ALL types (the ``types`` filter is applied afterwards)."""
    return {
        "from": 0,
        "size": size,
        "track_total_hits": True,
        "_source": ["sid", "idno", f"{FILTER_FACETS_KEY}.dataset_type"],
        "query": {"bool": {"must": [lexical_query(text)], "filter": filter_clauses(filters)}},
        "sort": [{"_score": {"order": "desc"}}, {"sid": {"order": "asc"}}],
    }


def _dataset_type(hit: dict[str, Any]) -> str:
    values = (hit["_source"].get(FILTER_FACETS_KEY) or {}).get("dataset_type") or []
    return str(values[0]) if values else "unknown"


async def _lexical_hits(job: SearchJob, size: int) -> tuple[list[Ranked], bool, float, dict[str, Any]]:
    """``(best-first keyword matches within the relative cutoff, whether more studies matched than were returned,
    took, request)``."""
    assert job.query is not None
    body = lexical_body(job.query, job.filters, size)
    response = await _search(job.client, job.index, body)
    fetched = [
        Ranked(
            sid=int(h["_source"]["sid"]),
            score=float(h["_score"]),
            matched_by=["lexical"],
            idno=h["_source"]["idno"],
            dataset_type=_dataset_type(h),
        )
        for h in response["hits"]["hits"]
    ]
    hits = apply_lexical_cutoff(fetched, job.policy)
    # the cutoff drops the weak tail; only when it drops nothing can the studies beyond ``size`` still be strong ones
    more = _total(response) > size and len(hits) == len(fetched)
    return hits, more, _took(response), body


async def lexical(job: SearchJob) -> StudyPage:
    """Keyword search: the best ``result_cap`` matches; ``truncated`` says whether more studies matched."""
    hits, more, took, body = await _lexical_hits(job, job.result_cap)
    return await _page(job, hits, truncated=more, took=took, bodies=[body])


# ---------------------------------------------------------------------------------------
# semantic and hybrid
# ---------------------------------------------------------------------------------------


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


async def _semantic_hits(job: SearchJob, vector: list[float]) -> tuple[list[Ranked], float, dict[str, Any]]:
    """Vector matches that clear the floor and the relative cutoff, one per study, best first."""
    body = knn_body(vector, filter_clauses(job.filters, CHUNK_FIELDS), job.policy)
    response = await _search(job.client, job.chunk_index, body)
    hits = apply_semantic_policy(parse_semantic(response, job.policy), job.policy)
    return hits, _took(response), _debug_body(body)


async def semantic(job: SearchJob) -> StudyPage:
    """Vector search only: studies whose best chunk clears the floor and the cutoff."""
    vector = await _embed(job)
    hits, took, body = await _semantic_hits(job, vector)
    hits, took_lookup, lookup = await _enrich(job, hits)
    return await _page(job, hits, truncated=False, took=took + took_lookup, bodies=[body, *lookup])


async def hybrid(job: SearchJob) -> StudyPage:
    """Both legs, fused by rank: keyword matches lead, and the vector leg fills what keywords cannot reach."""
    vector = await _embed(job)  # fail before any search runs
    (lex, more, took_lex, body_lex), (sem, took_sem, body_sem) = await asyncio.gather(
        _lexical_hits(job, job.policy.window), _semantic_hits(job, vector)
    )
    fused, took_lookup, lookup = await _enrich(job, rrf_fuse(lex, sem, job.policy))
    return await _page(
        job,
        fused,
        truncated=more,
        took=took_lex + took_sem + took_lookup,
        bodies=[body_lex, body_sem, *lookup],
    )


#: One executor per effective mode this engine serves.
EXECUTORS: dict[EffectiveMode, Callable[[SearchJob], Awaitable[StudyPage]]] = {
    EffectiveMode.browse: browse,
    EffectiveMode.lexical: lexical,
    EffectiveMode.semantic: semantic,
    EffectiveMode.hybrid: hybrid,
}

OPENSEARCH_STUDY_MODES: frozenset[str] = frozenset(mode.value for mode in EXECUTORS)
