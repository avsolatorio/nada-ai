"""Pure helpers of the semantic leg of the study search: the vector request, passages, the relevance policy, fusion.

Nothing here talks to OpenSearch. The executors in ``studies_search.py`` call these, which keeps the ranking rules
(floor, cutoff, fusion) testable without a cluster.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD, FILTER_FACETS_KEY, METADATA_OBJECT_KEY
from nada_ai.settings import Settings

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class StudyPolicy:
    """The knobs that decide what a relevance search returns. Built from settings; tuned in the evaluation step."""

    #: Studies each leg contributes before fusion.
    window: int
    #: Chunk candidates the vector search considers before they are collapsed to one hit per study.
    semantic_k: int
    #: Absolute floor on a chunk's vector score; below it a study is never a semantic match (rejects gibberish).
    semantic_min_score: float
    #: Keep semantic matches scoring at least this fraction of the best one (bounds the tail).
    semantic_relative_cutoff: float
    #: Keep keyword matches scoring at least this fraction of the best keyword match (0 keeps them all).
    lexical_relative_cutoff: float
    lexical_weight: float
    semantic_weight: float
    rrf_k: int
    #: Passages returned per document study, and the length of each excerpt.
    passages_per_study: int = 5
    excerpt_chars: int = 400

    @classmethod
    def from_settings(cls, settings: Settings) -> StudyPolicy:
        return cls(
            window=settings.studies_candidate_window,
            semantic_k=settings.studies_semantic_k,
            semantic_min_score=settings.studies_semantic_min_score,
            semantic_relative_cutoff=settings.studies_semantic_relative_cutoff,
            lexical_relative_cutoff=settings.studies_lexical_relative_cutoff,
            lexical_weight=settings.studies_fusion_lexical_weight,
            semantic_weight=settings.studies_fusion_semantic_weight,
            rrf_k=settings.studies_fusion_rank_constant,
        )


@dataclass
class Ranked:
    """One study in a ranked list. ``idno`` and ``dataset_type`` are filled from the study index when missing."""

    sid: int
    score: float
    matched_by: list[str]
    idno: str | None = None
    dataset_type: str | None = None
    passages: list[dict[str, Any]] = field(default_factory=list)


def knn_body(vector: list[float], filter_clauses: list[dict[str, Any]], policy: StudyPolicy) -> dict[str, Any]:
    """Vector search on the chunk index: filters apply *before* the search (the top ``k`` come from the filtered set),
    and chunks collapse to one hit per study, keeping that study's best passages."""
    return {
        "size": policy.window,
        "track_total_hits": False,
        "_source": [f"{METADATA_OBJECT_KEY}.sid", f"{METADATA_OBJECT_KEY}.{FILTER_FACETS_KEY}.dataset_type"],
        "query": {
            "knn": {
                EMBEDDING_FIELD: {
                    "vector": vector,
                    "k": policy.semantic_k,
                    "filter": {"bool": {"filter": filter_clauses}},
                }
            }
        },
        "collapse": {
            "field": f"{METADATA_OBJECT_KEY}.sid",
            "inner_hits": {
                "name": "passages",
                "size": policy.passages_per_study,
                "_source": [f"{METADATA_OBJECT_KEY}.qfield", f"{METADATA_OBJECT_KEY}.doc_meta", "page_content"],
            },
        },
    }


def passages_from_inner_hits(inner_hits: list[dict[str, Any]], policy: StudyPolicy) -> list[dict[str, Any]]:
    """Matching pages of a document study, best first: ``{page (1-based), total_pages?, score, excerpt?}``.

    Only passage chunks carry a page; the best score per page wins.
    """
    best: dict[int, dict[str, Any]] = {}
    for hit in inner_hits:
        source = hit.get("_source") or {}
        meta = source.get(METADATA_OBJECT_KEY) or {}
        doc_meta = meta.get("doc_meta") or {}
        page_index = doc_meta.get("page")
        if meta.get("qfield") != "passages" or not isinstance(page_index, int) or page_index < 0:
            continue
        score = float(hit.get("_score") or 0.0)
        if page_index in best and best[page_index]["score"] >= score:
            continue
        passage: dict[str, Any] = {"page": page_index + 1, "score": round(score, 4)}
        if isinstance(doc_meta.get("total_pages"), int) and doc_meta["total_pages"] > 0:
            passage["total_pages"] = doc_meta["total_pages"]
        excerpt = _WHITESPACE.sub(" ", str(source.get("page_content") or "")).strip()
        if excerpt:
            passage["excerpt"] = excerpt[: policy.excerpt_chars]
        best[page_index] = passage
    return sorted(best.values(), key=lambda p: (-p["score"], p["page"]))


def parse_semantic(response: dict[str, Any], policy: StudyPolicy) -> list[Ranked]:
    """Study hits of a collapsed vector search, best first."""
    ranked = []
    for hit in response["hits"]["hits"]:
        meta = hit["_source"][METADATA_OBJECT_KEY]
        types = (meta.get(FILTER_FACETS_KEY) or {}).get("dataset_type") or []
        inner = ((hit.get("inner_hits") or {}).get("passages") or {}).get("hits", {}).get("hits") or []
        ranked.append(
            Ranked(
                sid=int(meta["sid"]),
                score=float(hit["_score"]),
                matched_by=["semantic"],
                dataset_type=str(types[0]) if types else None,
                passages=passages_from_inner_hits(inner, policy),
            )
        )
    return ranked


def apply_semantic_policy(hits: list[Ranked], policy: StudyPolicy) -> list[Ranked]:
    """Keep semantic matches that clear the absolute floor and are close enough to the best one.

    The floor decides whether there are any semantic matches at all (nothing meaningful is close to a gibberish
    query); the relative cutoff bounds how many. ``hits`` are best first.
    """
    if not hits:
        return []
    threshold = max(policy.semantic_min_score, hits[0].score * policy.semantic_relative_cutoff)
    return [hit for hit in hits if hit.score >= threshold]


def apply_lexical_cutoff(hits: list[Ranked], policy: StudyPolicy) -> list[Ranked]:
    """Keep keyword matches close enough to the best one. Keyword scores are unbounded, so a fraction of the best
    score separates the strong matches from the long tail of partial ones. ``hits`` are best first."""
    if not hits:
        return []
    threshold = hits[0].score * policy.lexical_relative_cutoff
    return [hit for hit in hits if hit.score >= threshold]


def rrf_fuse(lexical: list[Ranked], semantic: list[Ranked], policy: StudyPolicy) -> list[Ranked]:
    """Reciprocal rank fusion of the two legs. Scores of different engines are not comparable, ranks are.

    A study's fused score is the sum over the legs of ``weight / (rrf_k + rank)``, so a study found by both legs
    outranks one found by either alone, and the lexical weight (higher by default) keeps exact matches on top.
    Ties break by ``sid``. ``matched_by`` lists lexical before semantic.
    """
    scores: dict[int, float] = {}
    merged: dict[int, Ranked] = {}
    for weight, ranked in ((policy.lexical_weight, lexical), (policy.semantic_weight, semantic)):
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.sid] = scores.get(hit.sid, 0.0) + weight / (policy.rrf_k + rank)
            known = merged.get(hit.sid)
            if known is None:
                merged[hit.sid] = replace(hit, matched_by=list(hit.matched_by), passages=list(hit.passages))
                continue
            known.matched_by += [leg for leg in hit.matched_by if leg not in known.matched_by]
            known.idno = known.idno or hit.idno
            known.dataset_type = known.dataset_type or hit.dataset_type
            known.passages = known.passages or list(hit.passages)
    order = sorted(scores, key=lambda sid: (-scores[sid], sid))
    return [replace(merged[sid], score=scores[sid]) for sid in order]
