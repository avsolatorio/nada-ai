"""Run the golden queries against the study search executors and report ranking quality.

Talks to OpenSearch directly (no API server), so the ranking policy can be varied per run:

    uv run python eval/run_golden.py                      # lexical / semantic / hybrid with the configured defaults
    uv run python eval/run_golden.py --sweep              # also sweep the semantic floor, cutoff and window
    uv run python eval/run_golden.py --json out.json      # machine-readable results

Reads the same ``NADA_*`` environment as the server (OpenSearch URL, index names, embedding model). Relevance
comes from ``golden_queries.json``; it never depends on the engine under test.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
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

sys.path.insert(0, str(Path(__file__).parent))
from metrics import report, score_query, summarize  # noqa: E402

GOLDEN = Path(__file__).with_name("golden_queries.json")


@dataclass(frozen=True)
class Variant:
    name: str
    mode: EffectiveMode
    policy: StudyPolicy


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
            policy=variant.policy,
            embed=embed,
        )
        page = await EXECUTORS[variant.mode](job)
        sids = [h["sid"] for h in page.hits]
        metrics = score_query(q, sids)
        metrics["returned"] = page.found  # the size of the whole result, not of the 100 studies fetched
        if q["expect"] == "empty":
            metrics["pass"] = page.found == 0
        rows.append({"id": q["id"], "category": q["category"], "sids": sids, "found": page.found, "m": metrics})
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
        grid = itertools.product((0.68, 0.70, 0.72), (0.90, 0.94, 0.98), (25, 50, 100))
        for floor, cutoff, window in grid:
            policy = replace(base, semantic_min_score=floor, semantic_relative_cutoff=cutoff, semantic_window=window)
            out.append(Variant(f"hybrid floor={floor} cutoff={cutoff} window={window}", EffectiveMode.hybrid, policy))
    return out


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
