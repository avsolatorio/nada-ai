"""POST /studies/search, semantic and hybrid modes: the policy, the semantic block, and the executors."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from opensearchpy.exceptions import NotFoundError
from starlette.testclient import TestClient

from nada_ai.app.main import app, state
from nada_ai.app.studies_schemas import (
    EffectiveMode,
    ErrorResponse,
    SortField,
    SortOrder,
    StudyFilters,
    StudySearchResponse,
)
from nada_ai.search.backend.opensearch.studies_search import (
    CHUNK_FIELDS,
    EXECUTORS,
    EmbeddingUnavailable,
    IndexNotReady,
    SearchJob,
    filter_clauses,
    hybrid,
    semantic,
)
from nada_ai.search.backend.opensearch.studies_semantic import (
    Ranked,
    StudyPolicy,
    apply_semantic_policy,
    knn_body,
    parse_semantic,
    passages_from_inner_hits,
    rrf_fuse,
)
from nada_ai.settings import Settings

POLICY = StudyPolicy(
    semantic_window=50,
    fusion_window=50,
    semantic_k=1000,
    semantic_min_score=0.68,
    semantic_relative_cutoff=0.9,
)

# ---------------------------------------------------------------------------------------
# Policy and settings
# ---------------------------------------------------------------------------------------


def test_the_policy_comes_from_settings() -> None:
    settings = Settings(studies_semantic_min_score=0.7, studies_semantic_window=20, studies_fusion_window=30)
    policy = StudyPolicy.from_settings(settings)
    assert (policy.semantic_window, policy.fusion_window, policy.semantic_k) == (20, 30, 1000)
    assert (policy.semantic_min_score, policy.semantic_relative_cutoff) == (0.7, 0.94)


def test_the_semantic_side_is_bounded_and_the_keyword_side_is_not() -> None:
    """The semantic window bounds the related studies only. There is no cap on the keyword matches, no score cutoff,
    and the two rankings are fused with equal weights."""
    assert (Settings().studies_semantic_window, Settings().studies_fusion_window) == (50, 50)
    for gone in (
        "studies_result_cap",
        "studies_candidate_window",
        "studies_lexical_relative_cutoff",
        "studies_fusion_lexical_weight",
        "studies_fusion_semantic_weight",
        "studies_fusion_rank_constant",
    ):
        assert gone not in Settings.model_fields


def test_fusion_puts_a_study_found_by_both_lists_first_and_alternates_the_rest() -> None:
    keyword = [Ranked(sid=s, score=9.0, matched_by=["lexical"], idno=f"I{s}", dataset_type="survey") for s in (1, 2, 5)]
    semantic = [Ranked(sid=s, score=0.9, matched_by=["semantic"], dataset_type="table") for s in (3, 2, 7)]
    fused = rrf_fuse(keyword, semantic)
    assert [h.sid for h in fused] == [2, 1, 3, 5, 7]  # 2 is in both; then rank 1 of each list, rank 3 of each (by sid)
    assert fused[0].matched_by == ["lexical", "semantic"]
    assert all(h.score is None for h in fused)  # the fused score only orders; it is not carried on


def test_fusion_keeps_what_each_list_knows() -> None:
    keyword = [Ranked(sid=1, score=9.0, matched_by=["lexical"], idno="NADA_1", dataset_type="document")]
    semantic = [Ranked(sid=1, score=0.8, matched_by=["semantic"], passages=[{"page": 2}])]
    (fused,) = rrf_fuse(keyword, semantic)
    assert (fused.idno, fused.dataset_type, fused.passages) == ("NADA_1", "document", [{"page": 2}])


def test_fusing_nothing_is_nothing() -> None:
    assert rrf_fuse([], []) == []


def test_the_vector_request_asks_for_the_semantic_window() -> None:
    body = knn_body([0.1, 0.2], [], StudyPolicy(**{**POLICY.__dict__, "semantic_window": 37}))
    assert body["size"] == 37


# ---------------------------------------------------------------------------------------
# The vector request
# ---------------------------------------------------------------------------------------


def test_knn_body_filters_before_the_search_and_collapses_to_one_hit_per_study() -> None:
    clauses = filter_clauses(StudyFilters(countries=[16], created_from=5, sids=[2, 3]), CHUNK_FIELDS)
    body = knn_body([0.1, 0.2], clauses, POLICY)
    knn = body["query"]["knn"]["embedding"]
    assert (knn["k"], knn["vector"]) == (1000, [0.1, 0.2])
    assert knn["filter"] == {"bool": {"filter": clauses}}  # the top k come from the filtered set
    assert body["size"] == POLICY.semantic_window
    assert body["collapse"]["field"] == "metadata.sid"
    assert body["collapse"]["inner_hits"]["name"] == "passages"
    assert body["track_total_hits"] is False


def test_chunk_filters_point_at_the_chunk_fields() -> None:
    clauses = filter_clauses(
        StudyFilters(
            countries=[16],
            year_from=2010,
            repository="demo",
            form_ids=[1],
            facets={"author": [7]},
            sids=[2],
            created_from=5,
            types=["survey"],
        ),
        CHUNK_FIELDS,
    )
    assert clauses == [
        {"term": {"metadata.filter_facets.published": 1}},
        {"terms": {"metadata.filter_facets.countries": [16]}},
        {"range": {"metadata.filter_facets.years": {"gte": 2010}}},
        {"term": {"metadata.filter_facets.repositories": "demo"}},
        {"terms": {"metadata.filter_facets.formid": [1]}},
        {"terms": {"metadata.filter_facets.fq_author": [7]}},
        {"terms": {"metadata.sid": [2]}},
        {"range": {"metadata.created": {"gte": 5}}},
    ]  # `types` is applied after the cut, on every leg


def test_study_and_chunk_filters_are_the_same_filters_on_different_paths() -> None:
    filters = StudyFilters(countries=[1], tags=["x"], year_to=2020, data_class_ids=[3], collections=["a"])
    study = filter_clauses(filters)
    chunk = filter_clauses(filters, CHUNK_FIELDS)
    assert len(study) == len(chunk)
    assert [str(c).replace("metadata.", "") for c in chunk] == [str(c) for c in study]


# ---------------------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------------------


def _passage(page: int | None, score: float, text: str = "text", total: int | None = 12, qfield: str = "passages"):
    doc_meta: dict[str, Any] = {}
    if page is not None:
        doc_meta["page"] = page
    if total is not None:
        doc_meta["total_pages"] = total
    return {"_score": score, "_source": {"metadata": {"qfield": qfield, "doc_meta": doc_meta}, "page_content": text}}


def test_passages_are_pages_best_first_with_a_normalised_excerpt() -> None:
    inner = [
        _passage(2, 0.71, "  poverty\n fell   sharply "),
        _passage(0, 0.83, "first page"),
        _passage(2, 0.69, "worse copy of page three"),
        _passage(2, 0.75, "best copy of page three"),
    ]
    assert passages_from_inner_hits(inner, POLICY) == [
        {"page": 1, "score": 0.83, "total_pages": 12, "excerpt": "first page"},
        {"page": 3, "score": 0.75, "total_pages": 12, "excerpt": "best copy of page three"},
    ]


def test_only_passage_chunks_with_a_page_count() -> None:
    inner = [
        _passage(1, 0.9, qfield="abstract"),
        _passage(None, 0.9),
        _passage(-1, 0.9),
        _passage(4, 0.8, text="   ", total=None),
    ]
    assert passages_from_inner_hits(inner, POLICY) == [{"page": 5, "score": 0.8}]


def test_long_excerpts_are_cut() -> None:
    (passage,) = passages_from_inner_hits([_passage(0, 0.9, "x" * 1000)], POLICY)
    assert len(passage["excerpt"]) == 400


def test_no_inner_hits_means_no_passages() -> None:
    assert passages_from_inner_hits([], POLICY) == []


# ---------------------------------------------------------------------------------------
# Parsing and the floor / cutoff
# ---------------------------------------------------------------------------------------


def _knn_hit(sid: int, score: float, dataset_type: str = "survey", inner: list[dict] | None = None) -> dict[str, Any]:
    return {
        "_score": score,
        "_source": {"metadata": {"sid": sid, "filter_facets": {"dataset_type": [dataset_type]}}},
        "inner_hits": {"passages": {"hits": {"hits": inner or []}}},
    }


def _knn_response(*hits: dict[str, Any], took: int = 4) -> dict[str, Any]:
    return {"took": took, "hits": {"total": {"value": len(hits)}, "hits": list(hits)}}


def test_parse_semantic_reads_sid_score_type_and_passages() -> None:
    (hit,) = parse_semantic(_knn_response(_knn_hit(7, 0.81, "document", [_passage(1, 0.8)])), POLICY)
    assert (hit.sid, hit.score, hit.dataset_type, hit.matched_by) == (7, 0.81, "document", ["semantic"])
    assert hit.idno is None  # the study index supplies NADA's idno
    assert hit.passages[0]["page"] == 2


def _ranked(*scores: float) -> list[Ranked]:
    return [Ranked(sid=i + 1, score=s, matched_by=["semantic"]) for i, s in enumerate(scores)]


def test_a_query_whose_best_match_is_below_the_floor_has_no_semantic_matches() -> None:
    assert apply_semantic_policy(_ranked(0.67, 0.66, 0.6), POLICY) == []  # gibberish peaks here


def test_the_relative_cutoff_bounds_the_tail_and_the_floor_applies_too() -> None:
    kept = apply_semantic_policy(_ranked(0.85, 0.80, 0.78, 0.70, 0.60), POLICY)
    assert [h.sid for h in kept] == [1, 2, 3]  # >= max(0.68, 0.85 * 0.9 = 0.765)
    low_top = apply_semantic_policy(_ranked(0.70, 0.69, 0.68, 0.60), POLICY)
    assert [h.sid for h in low_top] == [1, 2, 3]  # here the floor (0.68) is the binding limit


def test_no_hits_no_matches() -> None:
    assert apply_semantic_policy([], POLICY) == []


# ---------------------------------------------------------------------------------------
# Executors, against a mocked cluster
# ---------------------------------------------------------------------------------------


def _lexical_response(
    rows: list[tuple[int, str, float | None, str]],
    total: int | None = None,
    counts: dict[str, int] | None = None,
    titles: dict[int, str] | None = None,
) -> dict[str, Any]:
    """A keyword page: the rows shown, the total of all keyword matches, and the per-type counts of all of them.
    ``titles``, if given, adds a title to the named sids (only the head response needs one)."""
    hits = [
        {
            "_score": score,
            "_source": {
                "sid": sid,
                "idno": idno,
                "filter_facets": {"dataset_type": [dtype]},
                **({"title": titles[sid]} if titles and sid in titles else {}),
            },
        }
        for sid, idno, score, dtype in rows
    ]
    by_type = counts
    if by_type is None:
        by_type = {}
        for _, _, _, dtype in rows:
            by_type[dtype] = by_type.get(dtype, 0) + 1
    return {
        "took": 3,
        "hits": {"total": {"value": len(hits) if total is None else total}, "hits": hits},
        "aggregations": {"by_type": {"buckets": [{"key": k, "doc_count": v} for k, v in by_type.items()]}},
    }


def _agree_response(sids: list[int]) -> dict[str, Any]:
    return {"took": 1, "hits": {"total": {"value": len(sids)}, "hits": [{"_source": {"sid": sid}} for sid in sids]}}


def _lookup_response(rows: list[tuple[int, str, str]]) -> dict[str, Any]:
    hits = [
        {"_source": {"sid": sid, "idno": idno, "filter_facets": {"dataset_type": [dtype]}}} for sid, idno, dtype in rows
    ]
    return {"took": 2, "hits": {"total": {"value": len(hits)}, "hits": hits}}


def _resort_response(rows: list[tuple[int, str]], found: int, counts: dict[str, int]) -> dict[str, Any]:
    return {
        "took": 1,
        "hits": {"total": {"value": found}, "hits": [{"_source": {"sid": s, "idno": i}} for s, i in rows]},
        "aggregations": {"by_type": {"buckets": [{"key": k, "doc_count": v} for k, v in counts.items()]}},
    }


def _is_union(clauses: dict[str, Any]) -> bool:
    """A keyword search that also takes in studies without the keyword (another sort): its match is a bare ``should``,
    unlike the phrase-scored keyword match of a query of several words, which has a ``must``."""
    match = clauses["must"][0]
    return "bool" in match and "must" not in match["bool"]


def _is_idno_probe(body: dict[str, Any]) -> bool:
    """The exact-idno check every relevance search makes first: a filter-only query with a ``term`` on ``idno``,
    no ``must`` and no ``aggs``. Recognized so it never gets mistaken for the semantic-only enrich lookup, which
    matches the same shape otherwise."""
    clauses = body["query"]["bool"] if "query" in body and "bool" in body["query"] else {}
    return "must" not in clauses and any("term" in f and "idno" in f["term"] for f in clauses.get("filter", []))


def _cluster(
    *,
    lexical: dict | None = None,
    head: dict | None = None,
    agree: list[int] | None = None,
    idno: dict | None = None,
    knn: dict | None = None,
    lookup: dict | None = None,
    resort: dict | None = None,
    indexed: int = 10,
    chunks_missing: bool = False,
) -> MagicMock:
    """A cluster stand-in that answers each kind of request the executors make."""
    client = MagicMock()
    client.requests = []

    async def search(index: str, body: dict[str, Any]) -> dict[str, Any]:
        client.requests.append((index, body))
        if _is_idno_probe(body):  # checked first: it must never be answered by an unrelated configured response
            return idno if idno is not None else _lookup_response([])
        if "collapse" in body:  # the vector search on the chunk index
            if chunks_missing:
                raise NotFoundError(404, "index_not_found_exception", {})
            return knn if knn is not None else _knn_response()
        clauses = body["query"]["bool"]
        if "must" in clauses:
            if "aggs" not in body:  # which of the semantic studies also match the keyword
                return _agree_response(agree or [])
            page = lexical if lexical is not None else _lexical_response([], counts={})
            if "must_not" not in clauses and not _is_union(clauses):  # the best keyword matches (the head)
                return head if head is not None else page
            return page  # the keyword matches after the head, or a keyword page of any other kind
        if "aggs" in body:  # a filter-only listing (browse, or the semantic block re-sorted)
            return resort if resort is not None else _resort_response([], 0, {})
        return lookup if lookup is not None else _lookup_response([])  # idno and type of the semantic studies

    client.search = search
    client.count = AsyncMock(return_value={"count": indexed})
    return client


def _bodies(client: MagicMock, kind: str) -> list[dict[str, Any]]:
    """The requests of one kind: ``vector``, ``head`` (the best keyword matches), ``tail`` (the keyword matches after
    them), ``union`` (the block and the keyword matches, for another sort), ``agree`` or ``lookup``."""
    chosen = []
    for _, body in client.requests:
        clauses = body["query"]["bool"] if "query" in body and "bool" in body["query"] else {}
        if "collapse" in body:
            found = "vector"
        elif "must" not in clauses:
            found = "lookup" if "aggs" not in body else "browse"
        elif "aggs" not in body:
            found = "agree"
        elif "must_not" in clauses:
            found = "tail"
        elif _is_union(clauses):
            found = "union"
        else:
            found = "head"
        if found == kind:
            chosen.append(body)
    return chosen


async def _embed(_text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def _job(client: Any, **overrides: Any) -> SearchJob:
    fields: dict[str, Any] = {
        "client": client,
        "index": "studies",
        "chunk_index": "chunks",
        "query": "poverty",
        "filters": StudyFilters(),
        "sort_by": SortField.relevance,
        "sort_order": SortOrder.desc,
        "limit": 15,
        "offset": 0,
        "policy": POLICY,
        "embed": _embed,
    }
    fields.update(overrides)
    return SearchJob(**fields)


def _run(executor: Any, job: SearchJob):
    return asyncio.run(executor(job))


def test_semantic_asks_the_chunk_index_then_the_study_index() -> None:
    client = _cluster(
        knn=_knn_response(_knn_hit(5, 0.82, "geospatial"), _knn_hit(9, 0.78, "document", [_passage(3, 0.77)])),
        lookup=_lookup_response([(5, "NADA_5", "geospatial"), (9, "NADA_9", "document")]),
    )
    page = _run(semantic, _job(client, filters=StudyFilters(countries=[16])))

    (chunk_index, knn), (study_index, lookup) = client.requests
    assert chunk_index == "chunks" and knn["query"]["knn"]["embedding"]["vector"] == [0.1, 0.2, 0.3]
    assert {"terms": {"metadata.filter_facets.countries": [16]}} in knn["query"]["knn"]["embedding"]["filter"]["bool"][
        "filter"
    ]
    # the lookup uses the full filters on the study index and asks only for the semantic studies
    assert study_index == "studies"
    assert {"terms": {"filter_facets.countries": [16]}} in lookup["query"]["bool"]["filter"]
    assert {"terms": {"sid": [5, 9]}} in lookup["query"]["bool"]["filter"]

    assert [(h["sid"], h["idno"], h["matched_by"]) for h in page.hits] == [
        (5, "NADA_5", ["semantic"]),
        (9, "NADA_9", ["semantic"]),
    ]
    assert page.hits[1]["passages"] == [{"page": 4, "score": 0.77, "total_pages": 12, "excerpt": "text"}]
    assert page.counts_by_type == {"geospatial": 1, "document": 1}
    assert page.found == 2


def test_semantic_drops_studies_the_study_index_does_not_have() -> None:
    """A stale chunk (its study was removed) or a study a filter excludes must not surface."""
    client = _cluster(
        knn=_knn_response(_knn_hit(5, 0.82), _knn_hit(6, 0.80)),
        lookup=_lookup_response([(5, "NADA_5", "survey")]),
    )
    page = _run(semantic, _job(client))
    assert [h["sid"] for h in page.hits] == [5]


def test_semantic_below_the_floor_finds_nothing_and_skips_the_lookup() -> None:
    client = _cluster(knn=_knn_response(_knn_hit(5, 0.66), _knn_hit(6, 0.65)))
    page = _run(semantic, _job(client, query="xyzzy qwerty flurbo"))
    assert (page.found, page.hits) == (0, [])
    assert [index for index, _ in client.requests] == ["chunks"]


def test_semantic_needs_an_embedding() -> None:
    with pytest.raises(EmbeddingUnavailable):
        _run(semantic, _job(_cluster(), embed=None))


def test_a_failing_embedding_stops_hybrid_before_any_search() -> None:
    async def broken(_text: str) -> list[float]:
        raise EmbeddingUnavailable("model failed")

    client = _cluster()
    with pytest.raises(EmbeddingUnavailable):
        _run(hybrid, _job(client, embed=broken))
    assert client.requests == []


def test_a_missing_chunk_index_is_not_ready() -> None:
    with pytest.raises(IndexNotReady):
        _run(semantic, _job(_cluster(chunks_missing=True)))


def test_semantic_with_another_sort_re_sorts_the_block_in_opensearch() -> None:
    client = _cluster(
        knn=_knn_response(_knn_hit(5, 0.82, "survey"), _knn_hit(9, 0.78, "document")),
        lookup=_lookup_response([(5, "NADA_5", "survey"), (9, "NADA_9", "document")]),
        resort=_resort_response([(9, "NADA_9"), (5, "NADA_5")], 2, {"survey": 1, "document": 1}),
    )
    page = _run(semantic, _job(client, sort_by=SortField.title, sort_order=SortOrder.asc))
    assert [h["sid"] for h in page.hits] == [9, 5]
    assert {"terms": {"sid": [5, 9]}} in client.requests[-1][1]["query"]["bool"]["filter"]


# -- hybrid: the semantic studies fused with the best keyword matches, then every other keyword match

#: The vector search finds 3 (a table), 2 (a document, which the keyword search also matches) and 7 (a survey).
BLOCK = dict(
    knn=_knn_response(
        _knn_hit(3, 0.90, "table"), _knn_hit(2, 0.88, "document", [_passage(0, 0.87)]), _knn_hit(7, 0.86)
    ),
    lookup=_lookup_response([(3, "NADA_3", "table"), (2, "NADA_2", "document"), (7, "NADA_7", "survey")]),
    agree=[2],
)

#: The best keyword matches are 1, 2 and 5. Fused with the semantic studies 3, 2, 7 by rank, 2 (in both) leads and the
#: others follow in the order 1, 3, 5, 7 (ties break by sid): the two lists alternate.
HEAD = _lexical_response(
    [(1, "NADA_1", 12.0, "survey"), (2, "NADA_2", 9.0, "document"), (5, "NADA_5", 4.0, "survey")],
    total=43,
    counts={"survey": 31, "document": 11, "table": 1},
)
FUSED = [2, 1, 3, 5, 7]

#: 40 more keyword matches after the head and the block: the two best are shown on the first page.
TAIL = _lexical_response(
    [(9, "NADA_9", 3.0, "survey"), (11, "NADA_11", 2.0, "survey")],
    total=40,
    counts={"survey": 30, "document": 9, "table": 1},
)


def _sids(page: Any) -> list[int]:
    return [h["sid"] for h in page.hits]


def test_a_title_naming_every_query_word_is_promoted_ahead_of_the_fused_order() -> None:
    """The real case this exists for: a study whose title says exactly what was searched (here, three words) ranks
    behind four studies the fusion happens to rank higher, because the semantic leg also found them and rank fusion
    only counts position. The title match belongs first regardless."""
    head = _lexical_response(
        [
            (5, "NADA_5", 9.0, "geospatial"),  # ranked ahead by score, but not a title match
            (234, "AGO_2020_HRPM_GEO_v01_M", 8.0, "geospatial"),
        ],
        titles={234: "High Resolution Poverty Map (Geospatial Data), Angola, 2020", 5: "Surface water extent, 2020"},
    )
    client = _cluster(
        head=head,
        lexical=_lexical_response([], counts={}),
        knn=_knn_response(_knn_hit(5, 0.9, "geospatial"), _knn_hit(6, 0.89), _knn_hit(7, 0.88), _knn_hit(8, 0.87)),
        lookup=_lookup_response(
            [(5, "NADA_5", "geospatial"), (6, "NADA_6", "survey"), (7, "NADA_7", "survey"), (8, "NADA_8", "survey")]
        ),
    )
    page = _run(hybrid, _job(client, query="high resolution angola"))
    assert [h["sid"] for h in page.hits][0] == 234
    assert page.hits[0]["matched_by"] == ["lexical"]  # promoted, but still correctly attributed


def test_a_partial_title_match_is_not_promoted() -> None:
    """Two of the three words is not enough: fusion order is unchanged."""
    head = _lexical_response(
        [(1, "NADA_1", 9.0, "survey"), (2, "NADA_2", 8.0, "survey")],
        titles={1: "High Resolution Imagery", 2: "Angola Household Survey"},  # neither has all three words
    )
    client = _cluster(
        head=head,
        lexical=_lexical_response([], counts={}),
        knn=_knn_response(_knn_hit(2, 0.9)),
        lookup=_lookup_response([(2, "NADA_2", "survey")]),
    )
    page = _run(hybrid, _job(client, query="high resolution angola"))
    assert [h["sid"] for h in page.hits][0] == 2  # the normal fused order (found by both legs), unchanged


def test_more_than_one_title_match_keeps_their_relative_keyword_rank() -> None:
    head = _lexical_response(
        [(1, "NADA_1", 9.0, "survey"), (2, "NADA_2", 8.0, "survey"), (3, "NADA_3", 7.0, "survey")],
        titles={1: "Poverty Survey 2020", 2: "Rwanda Poverty Survey 2020", 3: "unrelated title here"},
    )
    client = _cluster(head=head, lexical=_lexical_response([], counts={}))
    page = _run(hybrid, _job(client, query="poverty survey 2020"))
    assert [h["sid"] for h in page.hits][:2] == [1, 2]  # both match; 1 keeps its lead over 2


def test_the_promotion_only_applies_to_a_relevance_sort() -> None:
    """A non-relevance sort orders the union of the semantic studies and the keyword matches by that sort; the
    title-match promotion (a relevance-only idea) plays no part."""
    head = _lexical_response([(1, "NADA_1", 9.0, "survey")], titles={1: "Poverty Survey 2020"})
    union = _lexical_response(
        [(9, "NADA_9", None, "table"), (1, "NADA_1", None, "survey")], total=2, counts={"survey": 1, "table": 1}
    )
    client = _cluster(
        head=head,
        lexical=union,
        knn=_knn_response(_knn_hit(9, 0.9, "table")),
        lookup=_lookup_response([(9, "NADA_9", "table")]),
    )
    page = _run(hybrid, _job(client, query="poverty survey 2020", sort_by=SortField.title, sort_order=SortOrder.asc))
    assert [h["sid"] for h in page.hits] == [9, 1]  # the union query's own order, not the title-match promotion


def test_hybrid_fuses_the_head_then_pages_every_other_keyword_match() -> None:
    client = _cluster(head=HEAD, lexical=TAIL, **BLOCK)
    page = _run(hybrid, _job(client))
    assert [(h["sid"], h["matched_by"]) for h in page.hits] == [
        (2, ["lexical", "semantic"]),
        (1, ["lexical"]),
        (3, ["semantic"]),
        (5, ["lexical"]),
        (7, ["semantic"]),
        (9, ["lexical"]),
        (11, ["lexical"]),
    ]
    assert page.hits[0]["passages"][0]["page"] == 1 and page.hits[2]["idno"] == "NADA_3"
    assert all(h["score"] is None for h in page.hits)  # the two orderings' scores could not be compared
    assert page.found == 5 + 40  # the fused head plus every other keyword match: nothing is cut
    # the keyword counts after the head plus the head's own
    assert page.counts_by_type == {"survey": 33, "document": 10, "table": 2}
    assert sum(page.counts_by_type.values()) == page.found


def test_an_exact_keyword_match_is_not_buried_under_semantic_only_studies() -> None:
    """The reason for fusing: with fifty related studies, a keyword match at rank 1 still comes near the top."""
    semantic_only = [_knn_hit(100 + i, 0.90 - i * 0.001) for i in range(50)]
    client = _cluster(
        head=_lexical_response([(1, "NADA_1", 40.0, "survey")], total=1, counts={"survey": 1}),
        knn=_knn_response(*semantic_only),
        lookup=_lookup_response([(100 + i, f"NADA_{100 + i}", "survey") for i in range(50)]),
    )
    page = _run(hybrid, _job(client, policy=replace(POLICY, semantic_window=50)))
    assert _sids(page)[:2] == [1, 100]  # the keyword match ties with the best semantic one; both are on top


def test_the_head_is_the_best_keyword_matches_over_all_types() -> None:
    client = _cluster(head=HEAD, lexical=TAIL, **BLOCK)
    _run(hybrid, _job(client, filters=StudyFilters(types=["survey"], countries=[16])))
    (head,) = _bodies(client, "head")
    assert (head["from"], head["size"]) == (0, POLICY.fusion_window)
    assert head["sort"] == [{"_score": {"order": "desc"}}, {"sid": {"order": "asc"}}]
    assert "post_filter" not in head and "dataset_type" not in str(head["query"])  # the tab never changes the order
    assert {"terms": {"filter_facets.countries": [16]}} in head["query"]["bool"]["filter"]
    assert _bodies(client, "vector")[0]["size"] == POLICY.semantic_window


def test_the_keyword_tail_leaves_out_the_fused_head_and_fills_the_page() -> None:
    client = _cluster(head=HEAD, lexical=TAIL, **BLOCK)
    _run(hybrid, _job(client))
    (tail,) = _bodies(client, "tail")
    assert tail["query"]["bool"]["must_not"] == [{"terms": {"sid": FUSED}}]
    assert (tail["from"], tail["size"]) == (0, 10)  # 15 rows, 5 of them from the fused head
    assert tail["sort"] == [{"_score": {"order": "desc"}}, {"sid": {"order": "asc"}}]


def test_a_page_inside_the_head_still_asks_for_the_total_and_the_counts() -> None:
    client = _cluster(head=HEAD, lexical=TAIL, **BLOCK)
    page = _run(hybrid, _job(client, limit=2, offset=0))
    assert _sids(page) == [2, 1]
    (tail,) = _bodies(client, "tail")
    assert (tail["from"], tail["size"]) == (0, 1)
    assert page.found == 45


def test_a_page_after_the_head_pages_the_keyword_matches() -> None:
    client = _cluster(head=HEAD, lexical=TAIL, **BLOCK)
    page = _run(hybrid, _job(client, limit=10, offset=25))
    (tail,) = _bodies(client, "tail")
    assert (tail["from"], tail["size"]) == (20, 10)  # positions 25.. are keyword matches 20.. after the head
    assert _sids(page) == [9, 11]  # whatever the mock returns, and no head rows


def test_a_page_across_the_end_of_the_head_takes_the_rest_from_the_keyword_matches() -> None:
    client = _cluster(head=HEAD, lexical=TAIL, **BLOCK)
    page = _run(hybrid, _job(client, limit=4, offset=3))
    (tail,) = _bodies(client, "tail")
    assert (tail["from"], tail["size"]) == (0, 2)
    assert _sids(page) == [5, 7, 9, 11]


def test_ascending_relevance_lists_the_weakest_keyword_matches_first_and_the_head_last() -> None:
    tail = _lexical_response(
        [(11, "NADA_11", 1.0, "survey"), (9, "NADA_9", 2.0, "survey")], total=40, counts={"survey": 40}
    )
    client = _cluster(head=HEAD, lexical=tail, **BLOCK)
    page = _run(hybrid, _job(client, sort_order=SortOrder.asc, limit=4))
    (body,) = _bodies(client, "tail")
    assert body["sort"][0] == {"_score": {"order": "asc"}}
    assert (body["from"], body["size"]) == (0, 4)
    assert _sids(page) == [11, 9, 7, 5]  # the fused head, reversed, follows the keyword matches


def test_a_query_of_several_words_scores_the_phrase_in_the_head_and_in_the_tail() -> None:
    client = _cluster(head=HEAD, lexical=TAIL, **BLOCK)
    _run(hybrid, _job(client, query="foreign direct investment"))
    for kind in ("head", "tail"):
        (body,) = _bodies(client, kind)
        keyword = body["query"]["bool"]["must"][0]["bool"]
        assert keyword["should"][0]["multi_match"]["type"] == "phrase"  # an optional score bonus, not a requirement
    assert _bodies(client, "agree")  # and which semantic studies match the keyword is still asked


def test_the_types_filter_narrows_found_and_the_hits_but_not_the_counts() -> None:
    tail = _lexical_response([(9, "NADA_9", 3.0, "survey")], total=31, counts={"survey": 30, "document": 9, "table": 1})
    client = _cluster(head=HEAD, lexical=tail, **BLOCK)
    page = _run(hybrid, _job(client, filters=StudyFilters(types=["survey"])))
    assert _sids(page) == [1, 5, 7, 9]  # the surveys of the fused head, then the surveys among the other matches
    assert page.found == 3 + 31
    assert page.counts_by_type == {"survey": 33, "document": 10, "table": 2}
    (body,) = _bodies(client, "tail")
    assert body["post_filter"] == {"terms": {"filter_facets.dataset_type": ["survey"]}}
    assert (body["from"], body["size"]) == (0, 12)


def test_another_sort_orders_the_semantic_studies_and_the_keyword_matches_together() -> None:
    union = _lexical_response(
        [(3, "NADA_3", None, "table"), (1, "NADA_1", None, "survey"), (2, "NADA_2", None, "document")],
        total=45,
        counts={"survey": 33, "document": 10, "table": 2},
    )
    client = _cluster(lexical=union, **BLOCK)
    page = _run(hybrid, _job(client, sort_by=SortField.title, sort_order=SortOrder.asc, limit=3))
    (body,) = _bodies(client, "union")
    should = body["query"]["bool"]["must"][0]["bool"]
    assert should["minimum_should_match"] == 1 and {"terms": {"sid": [3, 2, 7]}} in should["should"]
    assert body["sort"][0] == {"title_sort": {"order": "asc", "missing": "_last"}}
    assert "must_not" not in body["query"]["bool"]
    assert [(h["sid"], h["matched_by"]) for h in page.hits] == [
        (3, ["semantic"]),
        (1, ["lexical"]),
        (2, ["lexical", "semantic"]),
    ]
    # the same set as the relevance sort finds: the semantic studies plus every keyword match
    assert page.found == 45 and page.counts_by_type == {"survey": 33, "document": 10, "table": 2}


def test_hybrid_without_semantic_matches_is_the_keyword_search() -> None:
    client = _cluster(knn=_knn_response(_knn_hit(5, 0.60)), lexical=TAIL)
    page = _run(hybrid, _job(client))
    assert _sids(page) == [9, 11]
    assert all(h["matched_by"] == ["lexical"] for h in page.hits)
    assert page.found == 40
    assert "collapse" in page.request_bodies[0]  # the vector search is still listed in the debug output


def test_semantic_and_hybrid_are_registered() -> None:
    assert EXECUTORS[EffectiveMode.semantic] is semantic
    assert EXECUTORS[EffectiveMode.hybrid] is hybrid


def test_the_debug_request_hides_the_query_vector() -> None:
    client = _cluster(knn=_knn_response(_knn_hit(5, 0.9)), lookup=_lookup_response([(5, "N", "survey")]))
    page = _run(semantic, _job(client))
    shown = page.request_bodies[0]["query"]["knn"]["embedding"]["vector"]
    assert shown == "<3-dimensional query vector>"


# ---------------------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------------------


class _Model:
    """Stand-in for the embedding service: ``encode_query`` returns a vector, or fails."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def encode_query(self, text: str, **_kwargs: Any) -> np.ndarray:
        if self.fail:
            raise RuntimeError("model out of memory")
        return np.array([0.1, 0.2, 0.3])


@contextmanager
def _running(monkeypatch: pytest.MonkeyPatch, cluster: Any, model: Any) -> Iterator[TestClient]:
    monkeypatch.setenv("NADA_SEARCH_BACKEND", "opensearch")
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        previous = (state.client, state.embedding)
        state.client, state.embedding = cluster, model
        try:
            yield client
        finally:
            state.client, state.embedding = previous


def _post(client: TestClient, body: dict[str, Any]):
    return client.post("/studies/search", json=body)


def _full_cluster() -> MagicMock:
    return _cluster(
        # the best keyword matches (also what a keyword-only search returns), and the one match after them
        head=_lexical_response([(1, "NADA_1", 12.0, "survey"), (4, "NADA_4", 7.0, "survey")]),
        lexical=_lexical_response([(6, "NADA_6", 3.0, "survey")]),
        knn=_knn_response(_knn_hit(2, 0.85, "document", [_passage(0, 0.84)]), _knn_hit(3, 0.8, "table")),
        lookup=_lookup_response([(2, "NADA_2", "document"), (3, "NADA_3", "table")]),
        agree=[2],
    )


def test_a_query_defaults_to_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _full_cluster(), _Model()) as client:
        response = _post(client, {"query": "poverty"})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert body.invariant_violations() == []
    assert body.applied.mode is EffectiveMode.hybrid
    assert (body.applied.sort.by, body.applied.sort.order) == (SortField.relevance, SortOrder.desc)
    # keyword 1, 4 fused with semantic 2, 3 (alternating; 2 is also a keyword match), then the other keyword match 6
    assert [h.sid for h in body.hits] == [1, 2, 3, 4, 6]
    assert [[m.value for m in h.matched_by] for h in body.hits] == [
        ["lexical"],
        ["lexical", "semantic"],
        ["semantic"],
        ["lexical"],
        ["lexical"],
    ]
    assert body.hits[1].passages is not None and body.hits[1].passages[0].page == 1
    assert body.warnings == []
    assert (body.found, body.truncated) == (5, False)
    assert body.search_counts_by_type == {"survey": 3, "document": 1, "table": 1}


def test_explicit_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _full_cluster(), _Model()) as client:
        body = StudySearchResponse.model_validate(_post(client, {"query": "poverty", "mode": "semantic"}).json())
    assert body.applied.mode is EffectiveMode.semantic
    assert body.invariant_violations() == []
    assert all([m.value for m in h.matched_by] == ["semantic"] for h in body.hits)


def test_auto_degrades_to_keyword_search_when_the_model_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _full_cluster(), _Model(fail=True)) as client:
        response = _post(client, {"query": "poverty"})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert body.applied.mode is EffectiveMode.lexical
    assert [w.code.value for w in body.warnings] == ["semantic_unavailable"]
    assert [h.sid for h in body.hits] == [1, 4]
    assert all([m.value for m in h.matched_by] == ["lexical"] for h in body.hits)
    assert body.invariant_violations() == []


def test_degradation_keeps_the_requested_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = _cluster(lexical=_lexical_response([(2, "NADA_2", None, "document"), (1, "NADA_1", None, "survey")]))
    with _running(monkeypatch, cluster, _Model(fail=True)) as client:
        body = StudySearchResponse.model_validate(
            _post(client, {"query": "poverty", "sort": {"by": "title", "order": "asc"}}).json()
        )
    assert body.applied.mode is EffectiveMode.lexical
    assert (body.applied.sort.by, body.applied.sort.order) == (SortField.title, SortOrder.asc)
    assert [h.sid for h in body.hits] == [2, 1]


@pytest.mark.parametrize("mode", ["hybrid", "semantic"])
def test_an_explicit_semantic_mode_fails_rather_than_answering_something_else(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    with _running(monkeypatch, _full_cluster(), _Model(fail=True)) as client:
        response = _post(client, {"query": "poverty", "mode": mode})
    assert response.status_code == 503
    error = ErrorResponse.model_validate(response.json()).error
    assert error.code.value == "embedding_unavailable"
    assert error.details == {"mode": mode}


def test_no_query_never_needs_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    browse_cluster = _cluster(resort=_resort_response([(1, "NADA_1")], 1, {"survey": 1}))
    with _running(monkeypatch, browse_cluster, _Model(fail=True)) as client:
        response = _post(client, {})
    assert response.status_code == 200
    assert response.json()["warnings"] == []


def test_a_missing_chunk_index_is_index_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = _cluster(chunks_missing=True, lexical=_lexical_response([(1, "NADA_1", 5.0, "survey")]))
    with _running(monkeypatch, cluster, _Model()) as client:
        response = _post(client, {"query": "poverty"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "index_not_ready"


def test_the_semantic_window_reaches_the_vector_search_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NADA_STUDIES_SEMANTIC_WINDOW", "7")
    cluster = _full_cluster()
    with _running(monkeypatch, cluster, _Model()) as client:
        _post(client, {"query": "poverty"})
    assert _bodies(cluster, "vector")[0]["size"] == 7
