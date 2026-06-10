"""
End-to-end demo against the live Data Compass catalog + local OpenSearch.

1. Fetches the first page of catalog search results (real API).
2. Creates the index if needed and bulk-indexes those records (embeddings + text).
3. Runs sample keyword, vector, and hybrid searches and prints hit summaries.

Requires:
  - OpenSearch reachable at ``NADA_OPENSEARCH_URL`` (e.g. docker-compose in repo).
  - Network access to the configured catalog URL (see ai4data discovery ``AI4DATA_METADATA_CATALOG_URL``).
  - A SentenceTransformer model (first run may download weights). For a lighter run::

      export NADA_EMBEDDING_MODEL_ID=avsolatorio/GIST-small-Embedding-v0
      export NADA_QUERY_PROMPT_NAME=

Usage::

    cd nada-ai && uv sync --all-groups && uv run python -m nada_ai.demo_integration --max_items=5

    uv run python -m nada_ai.demo_integration --recreate_index
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from nada_ai.ingest.pipeline import run_bulk_index
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.queries import SearchMode, build_search_query
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def fetch_catalog_pairs_first_page(
    catalog_api_type: str,
    max_items: int,
) -> list[tuple[str, str]]:
    """Return ``(idno, metadata_type)`` using one search page from Data Compass."""
    from ai4data.discovery.catalog import get_ids_type, search_metadata

    ps = max(10, min(max_items, 100))
    params: dict[str, Any] = {
        "sk": "",
        "ps": ps,
        "type": catalog_api_type,
        "sort_by": "year",
        "sort_order": "asc",
        "page": 1,
    }
    data = search_metadata(params)
    rows = data.get("rows", [])[:max_items]
    pairs: list[tuple[str, str]] = []
    for row in rows:
        ids = get_ids_type(row)
        idno, mtype = ids.get("idno"), ids.get("type")
        if idno and mtype:
            pairs.append((str(idno), str(mtype)))
    return pairs


def _print_hits(label: str, resp: dict[str, Any], max_show: int = 3) -> None:
    hits = resp.get("hits", {}).get("hits", [])
    print(f"\n--- {label} ({len(hits)} hits) ---")
    for h in hits[:max_show]:
        src = h.get("_source", {})
        meta = src.get("metadata") or {}
        snippet = (src.get("page_content") or "")[:200].replace("\n", " ")
        idno = meta.get("idno", src.get("idno"))
        mtype = meta.get("type", src.get("type"))
        qfield = meta.get("qfield", src.get("qfield"))
        print(f"  score={h.get('_score')} idno={idno} type={mtype} qfield={qfield}")
        print(f"  text: {snippet}...")
    if len(hits) > max_show:
        print(f"  ... and {len(hits) - max_show} more")


def run_demo(
    max_items: int = 5,
    catalog_type: str = "timeseries",
    recreate_index: bool = False,
    force_fetch: bool = True,
    index_name_suffix: str | None = None,
) -> None:
    """
    :param catalog_type: Data Compass API type: ``timeseries``, ``survey``, ``document``, ``geospatial``.
    :param recreate_index: If True, delete and recreate the OpenSearch index (destructive).
    :param force_fetch: Passed through to metadata loader (refresh catalog JSON).
    :param index_name_suffix: If set, ``NADA_INDEX_NAME`` is ignored and name becomes ``nada-demo-{suffix}``.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = Settings()
    if index_name_suffix:
        settings = settings.model_copy(update={"index_name": f"nada-demo-{index_name_suffix}"})

    print(
        "Settings:",
        json.dumps(
            {
                "opensearch_url": settings.opensearch_url,
                "index": settings.index_name,
                "embedding_backend": settings.embedding_backend,
                "embedding_model_id": settings.embedding_model_id,
                "opensearch_ml_model_id": settings.opensearch_ml_model_id,
                "query_encoding": settings.describe_query_encoding(),
            },
            indent=2,
        ),
    )
    if settings.embedding_backend != "local":
        print(
            "(query_encoding applies when embedding_backend=local; "
            "opensearch_ml uses the cluster ingest model, not these prompts.)",
        )

    pairs = fetch_catalog_pairs_first_page(catalog_type, max_items)
    if not pairs:
        print("No catalog rows returned; check network or catalog_type.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetched {len(pairs)} id(s) from catalog (first page): {pairs[:3]}{'...' if len(pairs) > 3 else ''}")

    client = build_client(settings)
    try:
        health = client.cluster.health()
        print("Cluster health:", health.get("status"), "cluster_name=", health.get("cluster_name"))
    except Exception as e:
        print(f"Cannot reach OpenSearch at {settings.opensearch_url}: {e}", file=sys.stderr)
        print(
            "1) Start the dev cluster and wait until it is healthy (first start can take ~30–60s):\n"
            "     docker compose -f docker-compose.opensearch.yml up -d\n"
            "     curl -s http://localhost:9200/ | head\n"
            "2) If something else uses port 9200, stop it or set NADA_OPENSEARCH_URL to your cluster URL.",
            file=sys.stderr,
        )
        sys.exit(2)

    n, err = run_bulk_index(settings, pairs, force=force_fetch, recreate_index=recreate_index)
    print(f"Ingest complete: indexed={n} bulk_errors={len(err) if err else 0}")
    if err:
        print("First bulk errors:", err[:2])

    embedding = EmbeddingService(settings) if settings.embedding_backend == "local" else None
    sample_queries: list[tuple[str, SearchMode, str]] = [
        ("keyword", "keyword", "economic growth"),
        ("vector", "vector", "population health survey"),
        ("hybrid", "hybrid", "development indicator"),
    ]

    for label, mode, qtext in sample_queries:
        qvec = None
        if mode in ("vector", "hybrid") and settings.embedding_backend == "local":
            assert embedding is not None
            qvec = embedding.encode_query(qtext).tolist()
        body = build_search_query(
            settings,
            query_text=qtext,
            mode=mode,
            query_vector=qvec,
            filters=None,
            size=5,
            from_=0,
            knn_k=20,
        )
        resp = client.search(index=settings.index_name, body=body)
        _print_hits(f"{label}: {qtext!r}", resp)

    try:
        client.transport.close()
    except Exception:
        pass
    print("\nDone.")


def main() -> None:
    import fire

    fire.Fire(run_demo)


if __name__ == "__main__":
    main()
