"""Reusable ingest operations shared by the CLI and the FastAPI admin router.

Each ``*_op`` returns a small dict suitable for HTTP responses or job results, so
callers (CLI, API) just stringify or store the dict instead of duplicating the
logic. The CLI in :mod:`nada_ai.ingest.cli` is a thin ``print`` wrapper.
"""

from __future__ import annotations

import logging
from typing import Any

from nada_ai.ingest.pipeline import run_bulk_index
from nada_ai.ingest.quality import QualityReport
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.index_template import (
    put_cluster_auto_create_index,
    put_composable_index_template,
)
from nada_ai.search.backend.opensearch.ml.setup import ensure_text_embedding_ingest_pipeline
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def _close_quiet(client: Any) -> None:
    try:
        client.transport.close()
    except Exception:
        pass


def delete_by_idno_op(settings: Settings, idno: str) -> dict[str, Any]:
    """Delete all indexed documents/points for an idno. Works with both backends."""
    if settings.search_backend == "qdrant":
        return _delete_qdrant(settings, idno)
    return _delete_opensearch(settings, idno)


def _delete_qdrant(settings: Settings, idno: str) -> dict[str, Any]:
    from qdrant_client.http import models as qm
    from nada_ai.ingest.qdrant_writer import _client as make_client
    from nada_ai.search.backend.opensearch.mapping import metadata_field

    client = make_client(settings)
    coll = settings.qdrant_collection
    try:
        result = client.delete(
            collection_name=coll,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[
                    qm.FieldCondition(key=metadata_field("idno"), match=qm.MatchValue(value=idno))
                ])
            ),
        )
        return {
            "backend": "qdrant",
            "collection": coll,
            "idno": idno,
            "operation": result.status.value if result else "unknown",
        }
    finally:
        client.close()


def _delete_opensearch(settings: Settings, idno: str) -> dict[str, Any]:
    from nada_ai.search.backend.opensearch.mapping import metadata_field

    client = build_client(settings)
    try:
        body = {"query": {"term": {metadata_field("idno"): idno}}}
        resp = client.delete_by_query(index=settings.index_name, body=body, refresh=True)
        return {
            "backend": "opensearch",
            "index": settings.index_name,
            "idno": idno,
            "deleted": int(resp.get("deleted") or 0),
            "total": resp.get("total"),
        }
    finally:
        _close_quiet(client)


def delete_by_idnos_op(settings: Settings, idnos: list[str]) -> dict[str, Any]:
    """Delete all indexed documents/points for a batch of idnos in one call. Works with both backends."""
    if settings.search_backend == "qdrant":
        return _delete_qdrant_batch(settings, idnos)
    return _delete_opensearch_batch(settings, idnos)


def _delete_qdrant_batch(settings: Settings, idnos: list[str]) -> dict[str, Any]:
    from qdrant_client.http import models as qm
    from nada_ai.ingest.qdrant_writer import _client as make_client
    from nada_ai.search.backend.opensearch.mapping import metadata_field

    client = make_client(settings)
    coll = settings.qdrant_collection
    try:
        result = client.delete(
            collection_name=coll,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[
                    qm.FieldCondition(key=metadata_field("idno"), match=qm.MatchAny(any=idnos))
                ])
            ),
        )
        return {
            "backend": "qdrant",
            "collection": coll,
            "idnos": idnos,
            "operation": result.status.value if result else "unknown",
        }
    finally:
        client.close()


def _delete_opensearch_batch(settings: Settings, idnos: list[str]) -> dict[str, Any]:
    from nada_ai.search.backend.opensearch.mapping import metadata_field

    client = build_client(settings)
    try:
        body = {"query": {"terms": {metadata_field("idno"): idnos}}}
        resp = client.delete_by_query(index=settings.index_name, body=body, refresh=True)
        return {
            "backend": "opensearch",
            "index": settings.index_name,
            "idnos": idnos,
            "deleted": int(resp.get("deleted") or 0),
            "total": resp.get("total"),
        }
    finally:
        _close_quiet(client)


def put_index_template_op(settings: Settings) -> dict[str, Any]:
    """Install composable index template (and optional cluster auto-create setting) for OpenSearch only."""
    if settings.search_backend == "qdrant":
        return {
            "skipped": True,
            "detail": "Index templates apply to OpenSearch only (search_backend=qdrant).",
        }
    if settings.embedding_backend == "opensearch_ml":
        dim = int(settings.opensearch_ml_embedding_dimension or 0)
    else:
        dim = EmbeddingService(settings).embedding_dimension()

    client = build_client(settings)
    try:
        out: dict[str, Any] = {"dim": dim}
        if settings.opensearch_put_composable_index_template:
            out["template"] = put_composable_index_template(client, settings, dim)
        else:
            out["template"] = {"skipped": True, "reason": "opensearch_put_composable_index_template is false"}
        if settings.opensearch_cluster_auto_create_index:
            out["cluster_auto_create_index"] = put_cluster_auto_create_index(
                client, settings.opensearch_cluster_auto_create_index
            )
        return out
    finally:
        _close_quiet(client)


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
    embedding: EmbeddingService | None = None,
) -> dict[str, Any]:
    """Bulk-index the given idnos for a single metadata_type.

    ``embedding`` — pass the app's shared :class:`EmbeddingService` to avoid
    reloading the model for every job.  ``None`` (default) self-loads.

    Returns ``{"indexed", "errors", "requested", "metadata_type", "index", "quality"}``.
    ``quality`` is a non-blocking report of thin/malformed source documents
    (empty content, missing idno/type) observed during this run — see
    ``ingest/quality.py``. It never affects what gets indexed.
    """
    pairs = [(i, metadata_type) for i in idnos]
    report = QualityReport()
    n, err = run_bulk_index(
        settings,
        pairs,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
        embedding=embedding,
        quality_report=report,
    )
    idx = settings.qdrant_collection if settings.search_backend == "qdrant" else settings.index_name
    return {
        "indexed": int(n),
        "errors": err or [],
        "requested": len(idnos),
        "metadata_type": metadata_type,
        "index": idx,
        "quality": report.to_dict(),
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
    embedding: EmbeddingService | None = None,
) -> dict[str, Any]:
    """Fetch ids from Data Compass search API and bulk-index them.

    Returns ``{"indexed", "errors", "rows", "catalog_type", "index", "quality"}``.
    ``quality`` is a non-blocking report of thin/malformed source documents
    observed during this run — see ``ingest/quality.py``. It never affects
    what gets indexed.
    """
    from ai4data.discovery.catalog import get_metadata_ids, is_extract_mode

    params: dict[str, Any] = {"sk": "", "ps": ps, "type": catalog_type, "sort_by": "year", "sort_order": "asc"}
    if catalog_type == "indicator":
        params["type"] = "timeseries"
    elif catalog_type == "microdata":
        params["type"] = "survey"

    rows = get_metadata_ids(
        params,
        max_items=limit,
        cache_metadata=is_extract_mode(),
        include_resources=True,
    )
    pairs: list[tuple[str, str]] = []
    for row in rows:
        idno = row.get("idno")
        t = row.get("type")
        if not idno or not t:
            continue
        pairs.append((idno, t))

    report = QualityReport()
    n, err = run_bulk_index(
        settings,
        pairs,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
        embedding=embedding,
        quality_report=report,
    )
    idx = settings.qdrant_collection if settings.search_backend == "qdrant" else settings.index_name
    return {
        "indexed": int(n),
        "errors": err or [],
        "rows": len(pairs),
        "catalog_type": catalog_type,
        "index": idx,
        "quality": report.to_dict(),
    }
