"""Ranking-quality metrics and the comparison table, shared by the evaluation scripts.

Pure functions over a golden query (see ``golden_queries.json``) and a ranked list of study ids.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

TOP = 10


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
