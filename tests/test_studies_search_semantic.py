"""POST /studies/search, semantic and hybrid modes (step 7 of the OpenSearch plan): the policy, fusion, executors."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
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
    apply_lexical_cutoff,
    apply_semantic_policy,
    knn_body,
    parse_semantic,
    passages_from_inner_hits,
    rrf_fuse,
)
from nada_ai.settings import Settings

POLICY = StudyPolicy(
    window=200,
    semantic_k=1000,
    semantic_min_score=0.68,
    semantic_relative_cutoff=0.9,
    lexical_relative_cutoff=0.0,
    lexical_weight=1.0,
    semantic_weight=0.5,
    rrf_k=60,
)

# ---------------------------------------------------------------------------------------
# Policy and settings
# ---------------------------------------------------------------------------------------


def test_the_policy_comes_from_settings() -> None:
    settings = Settings(studies_semantic_min_score=0.7, studies_fusion_semantic_weight=0.25)
    policy = StudyPolicy.from_settings(settings)
    assert (policy.window, policy.semantic_k) == (200, 1000)
    assert (policy.semantic_min_score, policy.semantic_relative_cutoff) == (0.7, 0.94)
    assert policy.lexical_relative_cutoff == 0.4
    assert (policy.lexical_weight, policy.semantic_weight, policy.rrf_k) == (1.0, 0.25, 60)


def test_the_legs_weigh_the_same_by_default() -> None:
    """Chosen on the golden queries: a heavier keyword leg buried paraphrase matches, a heavier vector leg
    demoted exact and misspelled ones."""
    policy = StudyPolicy.from_settings(Settings())
    assert policy.lexical_weight == policy.semantic_weight


def test_the_lexical_cutoff_drops_the_weak_tail_of_keyword_matches() -> None:
    policy = StudyPolicy(**{**POLICY.__dict__, "lexical_relative_cutoff": 0.4})
    hits = [Ranked(sid=i, score=score, matched_by=["lexical"]) for i, score in enumerate((10.0, 6.0, 4.0, 3.9, 0.5), 1)]
    assert [h.sid for h in apply_lexical_cutoff(hits, policy)] == [1, 2, 3]
    assert apply_lexical_cutoff([], policy) == []


def test_a_zero_lexical_cutoff_keeps_every_keyword_match() -> None:
    hits = [Ranked(sid=i, score=score, matched_by=["lexical"]) for i, score in enumerate((10.0, 0.1), 1)]
    assert len(apply_lexical_cutoff(hits, POLICY)) == 2


# ---------------------------------------------------------------------------------------
# The vector request
# ---------------------------------------------------------------------------------------


def test_knn_body_filters_before_the_search_and_collapses_to_one_hit_per_study() -> None:
    clauses = filter_clauses(StudyFilters(countries=[16], created_from=5, sids=[2, 3]), CHUNK_FIELDS)
    body = knn_body([0.1, 0.2], clauses, POLICY)
    knn = body["query"]["knn"]["embedding"]
    assert (knn["k"], knn["vector"]) == (1000, [0.1, 0.2])
    assert knn["filter"] == {"bool": {"filter": clauses}}  # the top k come from the filtered set
    assert body["size"] == 200
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
# Fusion
# ---------------------------------------------------------------------------------------


def _lex(*sids: int) -> list[Ranked]:
    return [
        Ranked(sid=s, score=10.0 - i, matched_by=["lexical"], idno=f"I{s}", dataset_type="survey")
        for i, s in enumerate(sids)
    ]


def _sem(*sids: int) -> list[Ranked]:
    return [
        Ranked(sid=s, score=0.9 - i / 100, matched_by=["semantic"], dataset_type="survey") for i, s in enumerate(sids)
    ]


def test_a_study_found_by_both_legs_outranks_one_found_by_either_alone() -> None:
    fused = rrf_fuse(_lex(1, 2), _sem(2, 3), POLICY)
    assert [h.sid for h in fused] == [2, 1, 3]
    assert fused[0].matched_by == ["lexical", "semantic"]


def test_keyword_matches_lead_semantic_only_ones() -> None:
    fused = rrf_fuse(_lex(1, 2, 3), _sem(9, 8), POLICY)
    assert [h.sid for h in fused] == [1, 2, 3, 9, 8]  # semantic weight 0.5: its rank 1 is below lexical rank 3


def test_weights_and_the_rank_constant_are_the_policy() -> None:
    equal = StudyPolicy(**{**POLICY.__dict__, "semantic_weight": 1.0})
    assert [h.sid for h in rrf_fuse(_lex(1), _sem(9), equal)] == [1, 9]  # a tie breaks by sid
    semantic_first = StudyPolicy(**{**POLICY.__dict__, "lexical_weight": 0.5, "semantic_weight": 1.0})
    assert [h.sid for h in rrf_fuse(_lex(1), _sem(9), semantic_first)] == [9, 1]


def test_fused_scores_are_reciprocal_rank_sums() -> None:
    fused = rrf_fuse(_lex(1), _sem(1), POLICY)
    assert fused[0].score == pytest.approx(1.0 / 61 + 0.5 / 61)


def test_fusion_keeps_what_each_leg_knows() -> None:
    semantic_side = Ranked(sid=1, score=0.8, matched_by=["semantic"], dataset_type="document", passages=[{"page": 2}])
    (fused,) = rrf_fuse(_lex(1), [semantic_side], POLICY)
    assert (fused.idno, fused.passages) == ("I1", [{"page": 2}])


def test_fusing_nothing_is_nothing() -> None:
    assert rrf_fuse([], [], POLICY) == []


# ---------------------------------------------------------------------------------------
# Executors, against a mocked cluster
# ---------------------------------------------------------------------------------------


def _lexical_response(rows: list[tuple[int, str, float, str]], total: int | None = None) -> dict[str, Any]:
    hits = [
        {
            "_score": score,
            "_source": {"sid": sid, "idno": idno, "filter_facets": {"dataset_type": [dtype]}},
        }
        for sid, idno, score, dtype in rows
    ]
    return {"took": 3, "hits": {"total": {"value": len(hits) if total is None else total}, "hits": hits}}


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


def _cluster(
    *,
    lexical: dict | None = None,
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
        if "collapse" in body:
            if chunks_missing:
                raise NotFoundError(404, "index_not_found_exception", {})
            return knn if knn is not None else _knn_response()
        if "must" in body["query"]["bool"]:
            return lexical if lexical is not None else _lexical_response([])
        if "aggs" in body:
            return resort if resort is not None else _resort_response([], 0, {})
        return lookup if lookup is not None else _lookup_response([])

    client.search = search
    client.count = AsyncMock(return_value={"count": indexed})
    return client


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
        "result_cap": 100,
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
    # the lookup uses the full filters on the study index and asks only for the semantic-only studies
    assert study_index == "studies"
    assert {"terms": {"filter_facets.countries": [16]}} in lookup["query"]["bool"]["filter"]
    assert {"terms": {"sid": [5, 9]}} in lookup["query"]["bool"]["filter"]

    assert [(h["sid"], h["idno"], h["matched_by"]) for h in page.hits] == [
        (5, "NADA_5", ["semantic"]),
        (9, "NADA_9", ["semantic"]),
    ]
    assert page.hits[1]["passages"] == [{"page": 4, "score": 0.77, "total_pages": 12, "excerpt": "text"}]
    assert page.counts_by_type == {"geospatial": 1, "document": 1}
    assert (page.found, page.truncated, page.result_cap) == (2, False, 100)


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


def test_hybrid_fuses_keyword_and_semantic_matches() -> None:
    client = _cluster(
        lexical=_lexical_response([(1, "NADA_1", 12.0, "survey"), (2, "NADA_2", 7.0, "table")]),
        knn=_knn_response(_knn_hit(2, 0.85, "table"), _knn_hit(3, 0.80, "document", [_passage(0, 0.8)])),
        lookup=_lookup_response([(3, "NADA_3", "document")]),
    )
    page = _run(hybrid, _job(client))
    assert [(h["sid"], h["matched_by"]) for h in page.hits] == [
        (2, ["lexical", "semantic"]),  # found by both legs
        (1, ["lexical"]),
        (3, ["semantic"]),
    ]
    assert page.hits[2]["idno"] == "NADA_3" and "passages" in page.hits[2]
    assert page.counts_by_type == {"table": 1, "survey": 1, "document": 1}
    assert page.found == 3

    indexes = [index for index, _ in client.requests]
    assert sorted(indexes) == ["chunks", "studies", "studies"]  # keyword leg, vector leg, lookup of the semantic-only
    # only the semantic-only study is looked up: the keyword leg already came from the study index
    lookup_body = client.requests[-1][1]
    assert {"terms": {"sid": [3]}} in lookup_body["query"]["bool"]["filter"]


def test_hybrid_fetches_a_window_from_each_leg() -> None:
    client = _cluster(lexical=_lexical_response([(1, "NADA_1", 1.0, "survey")]))
    _run(hybrid, _job(client, policy=StudyPolicy(**{**POLICY.__dict__, "window": 37})))
    sizes = {("semantic" if "collapse" in body else "lexical"): body["size"] for _, body in client.requests}
    assert sizes == {"lexical": 37, "semantic": 37}


def test_hybrid_is_truncated_when_more_keyword_matches_exist_than_the_window() -> None:
    client = _cluster(lexical=_lexical_response([(1, "NADA_1", 5.0, "survey")], total=900))
    assert _run(hybrid, _job(client)).truncated is True
    assert _run(hybrid, _job(_cluster(lexical=_lexical_response([(1, "NADA_1", 5.0, "survey")])))).truncated is False


def test_hybrid_cuts_at_the_result_cap_across_all_types() -> None:
    rows = [(i, f"NADA_{i}", 100.0 - i, "survey" if i % 2 else "table") for i in range(1, 8)]
    client = _cluster(lexical=_lexical_response(rows))
    page = _run(hybrid, _job(client, result_cap=4, filters=StudyFilters(types=["table"])))
    assert page.truncated is True
    assert page.counts_by_type == {"survey": 2, "table": 2}  # the cut set (top 4), ignoring the types filter
    assert [h["sid"] for h in page.hits] == [2, 4]  # `types` narrows found and the hits only
    assert page.found == 2


def test_hybrid_with_another_sort_re_sorts_the_cut_set_in_opensearch() -> None:
    client = _cluster(
        lexical=_lexical_response([(1, "NADA_1", 9.0, "survey"), (2, "NADA_2", 8.0, "survey")]),
        knn=_knn_response(_knn_hit(3, 0.9, "document")),
        lookup=_lookup_response([(3, "NADA_3", "document")]),
        resort=_resort_response([(3, "NADA_3"), (1, "NADA_1"), (2, "NADA_2")], 3, {"survey": 2, "document": 1}),
    )
    page = _run(hybrid, _job(client, sort_by=SortField.title, sort_order=SortOrder.asc))
    assert [h["sid"] for h in page.hits] == [3, 1, 2]
    assert page.hits[0]["matched_by"] == ["semantic"]  # each hit keeps what the relevance search knew
    resort_body = client.requests[-1][1]
    assert {"terms": {"sid": [1, 2, 3]}} in resort_body["query"]["bool"]["filter"]


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
        lexical=_lexical_response([(1, "NADA_1", 12.0, "survey"), (2, "NADA_2", 7.0, "document")]),
        knn=_knn_response(_knn_hit(2, 0.85, "document", [_passage(0, 0.84)]), _knn_hit(3, 0.8, "table")),
        lookup=_lookup_response([(3, "NADA_3", "table")]),
    )


def test_a_query_defaults_to_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _full_cluster(), _Model()) as client:
        response = _post(client, {"query": "poverty"})
    assert response.status_code == 200
    body = StudySearchResponse.model_validate(response.json())
    assert body.invariant_violations() == []
    assert body.applied.mode is EffectiveMode.hybrid
    assert (body.applied.sort.by, body.applied.sort.order) == (SortField.relevance, SortOrder.desc)
    assert [h.sid for h in body.hits] == [2, 1, 3]
    assert [[m.value for m in h.matched_by] for h in body.hits] == [["lexical", "semantic"], ["lexical"], ["semantic"]]
    assert body.hits[0].passages is not None and body.hits[0].passages[0].page == 1
    assert body.warnings == []
    assert body.search_counts_by_type == {"document": 1, "survey": 1, "table": 1}


def test_explicit_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    with _running(monkeypatch, _full_cluster(), _Model()) as client:
        body = StudySearchResponse.model_validate(_post(client, {"query": "x", "mode": "semantic"}).json())
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
    assert [h.sid for h in body.hits] == [1, 2]
    assert all([m.value for m in h.matched_by] == ["lexical"] for h in body.hits)
    assert body.invariant_violations() == []


def test_degradation_keeps_the_requested_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = _cluster(
        lexical=_lexical_response([(1, "NADA_1", 12.0, "survey"), (2, "NADA_2", 7.0, "document")]),
        resort=_resort_response([(2, "NADA_2"), (1, "NADA_1")], 2, {"survey": 1, "document": 1}),
    )
    with _running(monkeypatch, cluster, _Model(fail=True)) as client:
        body = StudySearchResponse.model_validate(
            _post(client, {"query": "x", "sort": {"by": "title", "order": "asc"}}).json()
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
        response = _post(client, {"query": "x"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "index_not_ready"


def test_the_policy_reaches_the_executors_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NADA_STUDIES_CANDIDATE_WINDOW", "50")
    cluster = _full_cluster()
    with _running(monkeypatch, cluster, _Model()) as client:
        _post(client, {"query": "x"})
    sizes = sorted(body["size"] for _, body in cluster.requests[:2])  # the two legs
    assert sizes == [50, 50]
