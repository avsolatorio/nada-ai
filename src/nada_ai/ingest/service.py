"""Reusable ingest operations shared by the CLI and the FastAPI admin router.

Each ``*_op`` returns a small dict suitable for HTTP responses or job results, so
callers (CLI, API) just stringify or store the dict instead of duplicating the
logic. The CLI in :mod:`nada_ai.ingest.cli` is a thin ``print`` wrapper.
"""

from __future__ import annotations

import logging
from typing import Any

from nada_ai.ingest.pipeline import run_bulk_index
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.ml.setup import ensure_text_embedding_ingest_pipeline
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def _close_quiet(client: Any) -> None:
    try:
        client.transport.close()
    except Exception:
        pass


def create_index_op(settings: Settings, recreate: bool = False) -> dict[str, Any]:
    """Create the search index or Qdrant collection (drop first if ``recreate``).

    Returns ``{"index", "dim", "recreated", "embedding_backend"}``.
    """
    from nada_ai.ingest.factory import create_ingest_writer

    if settings.embedding_backend == "opensearch_ml":
        dim = int(settings.opensearch_ml_embedding_dimension or 0)
    else:
        dim = EmbeddingService(settings).embedding_dimension()

    writer = create_ingest_writer(settings)
    writer.ensure_target(dim, recreate=recreate)

    index_name = settings.qdrant_collection if settings.search_backend == "qdrant" else settings.index_name
    return {
        "index": index_name,
        "dim": dim,
        "recreated": recreate,
        "embedding_backend": settings.embedding_backend,
    }


def setup_ingest_pipeline_op(settings: Settings) -> dict[str, Any]:
    """Create or replace the ``text_embedding`` ingest pipeline.

    Returns ``{"pipeline", "embedding_backend", "skipped"}``.
    """
    if settings.search_backend == "qdrant":
        return {
            "pipeline": None,
            "embedding_backend": settings.embedding_backend,
            "skipped": True,
            "detail": "OpenSearch ingest pipelines do not apply when search_backend=qdrant.",
        }
    client = build_client(settings)
    try:
        skipped = settings.opensearch_ml_skip_ingest_pipeline_setup
        ensure_text_embedding_ingest_pipeline(client, settings)
    finally:
        _close_quiet(client)
    return {
        "pipeline": settings.opensearch_ml_ingest_pipeline_name,
        "embedding_backend": settings.embedding_backend,
        "skipped": skipped,
    }


def index_ids_op(
    settings: Settings,
    idnos: list[str],
    metadata_type: str = "indicator",
    force: bool = False,
    recreate_index: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 1000,
) -> dict[str, Any]:
    """Bulk-index the given idnos for a single metadata_type.

    Returns ``{"indexed", "errors", "requested", "metadata_type", "index"}``.
    """
    pairs = [(i, metadata_type) for i in idnos]
    n, err = run_bulk_index(
        settings,
        pairs,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
    )
    idx = settings.qdrant_collection if settings.search_backend == "qdrant" else settings.index_name
    return {
        "indexed": int(n),
        "errors": err or [],
        "requested": len(idnos),
        "metadata_type": metadata_type,
        "index": idx,
    }


def index_from_catalog_op(
    settings: Settings,
    catalog_type: str = "timeseries",
    ps: int = 100,
    limit: int | None = None,
    force: bool = False,
    recreate_index: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 1000,
) -> dict[str, Any]:
    """Fetch ids from Data Compass search API and bulk-index them.

    Returns ``{"indexed", "errors", "rows", "catalog_type", "index"}``.
    """
    from ai4data.discovery.catalog import get_metadata_ids

    params: dict[str, Any] = {"sk": "", "ps": ps, "type": catalog_type, "sort_by": "year", "sort_order": "asc"}
    if catalog_type == "indicator":
        params["type"] = "timeseries"
    elif catalog_type == "microdata":
        params["type"] = "survey"

    rows = get_metadata_ids(params, max_items=limit)
    pairs: list[tuple[str, str]] = []
    for row in rows:
        idno = row.get("idno")
        t = row.get("type")
        if not idno or not t:
            continue
        pairs.append((idno, t))

    n, err = run_bulk_index(
        settings,
        pairs,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
    )
    idx = settings.qdrant_collection if settings.search_backend == "qdrant" else settings.index_name
    return {
        "indexed": int(n),
        "errors": err or [],
        "rows": len(pairs),
        "catalog_type": catalog_type,
        "index": idx,
    }
