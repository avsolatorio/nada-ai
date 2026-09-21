"""Run the golden queries against the study search executors and report ranking quality.

Talks to OpenSearch directly (no API server), so the ranking policy can be varied per run:

    uv run python eval/run_golden.py                      # lexical / semantic / hybrid with the configured defaults
    uv run python eval/run_golden.py --sweep              # also sweep the hybrid policy
    uv run python eval/run_golden.py --json out.json      # machine-readable results

Reads the same ``NADA_*`` environment as the server (OpenSearch URL, index names, embedding model). Relevance
comes from ``golden_queries.json``; it never depends on the engine under test.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nada_ai.app.studies_schemas import EffectiveMode, SortField, SortOrder, StudyFilters
from nada_ai.search.backend.opensearch.client import build_async_client
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.studies_search import EXECUTORS, SearchJob
from nada_ai.search.backend.opensearch.studies_semantic import StudyPolicy
from nada_ai.settings import Settings

GOLDEN = Path(__file__).with_name("golden_queries.json")
TOP = 10


@dataclass(frozen=True)
class Variant:
    name: str
    mode: EffectiveMode
    policy: StudyPolicy


# ---------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------


def ndcg(grades: list[int], ideal: list[int], k: int = TOP) -> float:
    def dcg(gs: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))

    best = dcg(sorted(ideal, reverse=True))
    return dcg(grades) / best if best else 0.0


def score_query(query: dict[str, Any], sids: list[int]) -> dict[str, float | int | bool]:
    """Metrics of one query. ``sids`` is the ranked result (all pages, best first)."""
    relevant = {int(sid): g for sid, g in query["relevant"].items()}
    if query["expect"] == "empty":
        return {"returned": len(sids), "pass": not sids}
    grades = [relevant.get(sid, 0) for sid in sids]
    top = grades[:TOP]
    top_grade = max(relevant.values(), default=0)
    first = next((i for i, g in enumerate(grades, 1) if g >= min(2, top_grade)), None)
    core = {sid for sid, g in relevant.items() if g >= 2} or set(relevant)
    out: dict[str, float | int | bool] = {
        "returned": len(sids),
        "ndcg": ndcg(grades, list(relevant.values())),
        "p10": sum(1 for g in top if g >= 1) / max(1, len(top)),
        "mrr": 1 / first if first else 0.0,
        "empty": not sids,
    }
    if not query["broad"]:
        out["recall"] = len(core & set(sids)) / len(core)
        # how much of the whole result set is on topic: the measure of "returns everything"
        out["p_all"] = sum(1 for g in grades if g >= 1) / len(grades) if grades else 0.0
    return out


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query metrics: overall, per category, and the negative-query pass rate."""
    positive = [r for r in rows if "pass" not in r["m"]]
    negative = [r for r in rows if "pass" in r["m"]]

    def agg(part: list[dict[str, Any]]) -> dict[str, float | None]:
        return {
            "ndcg@10": mean([r["m"]["ndcg"] for r in part]),
            "p@10": mean([r["m"]["p10"] for r in part]),
            "mrr": mean([r["m"]["mrr"] for r in part]),
            "recall": mean([r["m"]["recall"] for r in part if "recall" in r["m"]]),
            "p_all": mean([r["m"]["p_all"] for r in part if "p_all" in r["m"]]),
            "size": statistics.median([r["m"]["returned"] for r in part]) if part else None,
            "empty": sum(1 for r in part if r["m"]["empty"]),
            "n": len(part),
        }

    by_category = {
        c: agg([r for r in positive if r["category"] == c]) for c in sorted({r["category"] for r in positive})
    }
    return {
        "overall": agg(positive),
        "by_category": by_category,
        "negative_pass": sum(1 for r in negative if r["m"]["pass"]),
        "negative_n": len(negative),
        "negative_returned": [r["m"]["returned"] for r in negative],
    }


# ---------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------


async def run_variant(
    variant: Variant, queries: list[dict[str, Any]], settings: Settings, client: Any, embed: Any
) -> list[dict[str, Any]]:
    rows = []
    for q in queries:
        job = SearchJob(
            client=client,
            index=settings.studies_index,
            chunk_index=settings.index_name,
            query=q["query"],
            filters=StudyFilters(**q["filters"]),
            sort_by=SortField.relevance,
            sort_order=SortOrder.desc,
            limit=100,
            offset=0,
            result_cap=settings.studies_result_cap,
            policy=variant.policy,
            embed=embed,
        )
        page = await EXECUTORS[variant.mode](job)
        sids = [h["sid"] for h in page.hits]
        rows.append({"id": q["id"], "category": q["category"], "sids": sids, "m": score_query(q, sids)})
    return rows


def caching_embedder(service: EmbeddingService):
    cache: dict[str, list[float]] = {}

    async def embed(text: str) -> list[float]:
        if text not in cache:
            cache[text] = [float(x) for x in await asyncio.to_thread(service.encode_query, text)]
        return cache[text]

    return embed


def variants(base: StudyPolicy, sweep: bool) -> list[Variant]:
    out = [
        Variant("lexical", EffectiveMode.lexical, base),
        Variant("semantic", EffectiveMode.semantic, base),
        Variant("hybrid", EffectiveMode.hybrid, base),
    ]
    if sweep:
        grid = itertools.product((0.68, 0.70, 0.72), (0.9, 0.94, 0.98), (0.0, 0.4, 0.7), (0.5, 1.0, 2.0))
        for floor, cutoff, lex_cutoff, weight in grid:
            policy = replace(
                base,
                semantic_min_score=floor,
                semantic_relative_cutoff=cutoff,
                lexical_relative_cutoff=lex_cutoff,
                semantic_weight=weight,
            )
            name = f"hybrid floor={floor} sem-cutoff={cutoff} lex-cutoff={lex_cutoff} w={weight}"
            out.append(Variant(name, EffectiveMode.hybrid, policy))
    return out


def fmt(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def report(results: dict[str, dict[str, Any]], full: bool) -> str:
    lines = [
        "| variant | ndcg@10 | p@10 | mrr | recall | precision (whole result) | median size | empty (positive) | negatives passed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in results.items():
        o = s["overall"]
        lines.append(
            f"| {name} | {fmt(o['ndcg@10'])} | {fmt(o['p@10'])} | {fmt(o['mrr'])} | {fmt(o['recall'])} "
            f"| {fmt(o['p_all'])} | {o['size']:.0f} | {o['empty']}/{o['n']} | {s['negative_pass']}/{s['negative_n']} |"
        )
    if full:
        for name, s in results.items():
            lines += [
                "",
                f"### {name}",
                "",
                "| category | n | ndcg@10 | p@10 | mrr | recall | empty |",
                "|---|---|---|---|---|---|---|",
            ]
            for cat, a in s["by_category"].items():
                lines.append(
                    f"| {cat} | {a['n']} | {fmt(a['ndcg@10'])} | {fmt(a['p@10'])} | {fmt(a['mrr'])} | {fmt(a['recall'])} | {a['empty']} |"
                )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="also run the hybrid policy grid")
    parser.add_argument("--json", type=Path, help="write per-query results and summaries here")
    args = parser.parse_args()

    settings = Settings()
    golden = json.loads(GOLDEN.read_text())
    queries = golden["queries"]
    client = build_async_client(settings)
    embed = caching_embedder(EmbeddingService(settings))
    try:
        results: dict[str, dict[str, Any]] = {}
        per_query: dict[str, list[dict[str, Any]]] = {}
        for variant in variants(StudyPolicy.from_settings(settings), args.sweep):
            rows = await run_variant(variant, queries, settings, client, embed)
            per_query[variant.name] = rows
            results[variant.name] = summarize(rows)
    finally:
        await client.close()

    print(report(results, full=not args.sweep))
    if args.json:
        args.json.write_text(json.dumps({"summary": results, "queries": per_query}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
