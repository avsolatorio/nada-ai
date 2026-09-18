"""Reusable ingest operations shared by the CLI and the FastAPI admin router.

Each ``*_op`` returns a small dict suitable for HTTP responses or job results, so
callers (CLI, API) just stringify or store the dict instead of duplicating the
logic. The CLI in :mod:`nada_ai.ingest.cli` is a thin ``print`` wrapper.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from nada_ai.ingest.pipeline import run_bulk_index
from nada_ai.ingest.progress import CancelToken, IngestProgressTracker, load_checkpoint
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


def _error_by_idno(load_errors: list[dict[str, Any]], empty_docs: list[dict[str, Any]]) -> dict[str, str]:
    """Map failed idno -> a human-readable reason, for ``last_error`` reporting.

    ``load_errors`` entries carry the real exception text; ``empty_docs`` ones
    don't (they didn't raise), so fall back to their ``reason`` code.
    """
    errors = {e["idno"]: str(e["error"]) for e in load_errors}
    for e in empty_docs:
        errors.setdefault(e["idno"], f"no documents produced ({e.get('reason', 'empty')})")
    return errors


def _attribute_write_error(err: Any) -> tuple[str | None, str]:
    """Best-effort ``(idno, message)`` for one backend write-error entry.

    Qdrant writer entries carry an ``idno`` key directly. OpenSearch ``bulk``
    entries are ``{"index": {"_id", "error", "data": <source doc>}}`` — the idno
    is read from the echoed source's ``metadata.idno``. ``idno`` is ``None``
    when the entry can't be tied to one idno (e.g. an error that aborted the
    whole run partway through).
    """
    if not isinstance(err, dict):
        return None, str(err)
    idno = err.get("idno")
    message = str(err.get("error") or err)
    if not idno:
        for op in ("index", "create", "update"):
            body = err.get(op)
            if not isinstance(body, dict):
                continue
            message = str(body.get("error") or message)
            data = body.get("data")
            meta = data.get("metadata") if isinstance(data, dict) else None
            idno = meta.get("idno") if isinstance(meta, dict) else None
            break
    return (str(idno) if idno else None), message


def _state_report_items(
    candidate_idnos: Iterable[str],
    load_errors: list[dict[str, Any]],
    empty_docs: list[dict[str, Any]],
    write_errors: list[Any] | None,
) -> list[dict[str, Any]]:
    """Build the ``search_index_state`` items for one indexing call.

    ``load_errors``/``empty_docs`` and backend ``write_errors`` are all failures
    — an idno whose documents never made it into the backend must not be
    reported ``indexed``, or NADA drops it from its "missing" diff and
    reconcile never retries it.

    Write errors that can't be tied to an idno (see :func:`_attribute_write_error`)
    mean we can't tell which idnos are affected, so in that case nothing is
    reported ``indexed`` for this call — those idnos stay "missing" in NADA and
    a later reconcile retries them, rather than being falsely marked done.
    """
    failed: dict[str, str] = {}
    fully_attributed = True
    for err in write_errors or []:
        idno, message = _attribute_write_error(err)
        if idno is None:
            fully_attributed = False
        else:
            failed.setdefault(idno, f"write failed: {message}")
    failed.update(_error_by_idno(load_errors, empty_docs))

    items: list[dict[str, Any]] = []
    if fully_attributed:
        items += [
            {"object_type": "survey", "object_key": i, "status": "indexed"} for i in candidate_idnos if i not in failed
        ]
    else:
        logger.warning(
            "Backend reported write errors not attributable to an idno; not reporting any idno as 'indexed' "
            "to search_index_state for this call (they stay 'missing' so a reconcile retries them)"
        )
    items += [{"object_type": "survey", "object_key": i, "status": "failed", "error": m} for i, m in failed.items()]
    return items


def _report_state_bulk_best_effort(settings: Settings, items: list[dict[str, Any]]) -> None:
    """Best-effort: tell NADA's search_index_state about an indexing/deletion
    outcome, for content this function indexed/deleted directly rather than
    via the queue/ack flow (which has its own reporting — see
    ``ingest/search_index_sync.py``).

    Deferred import: ``search_index_sync`` already imports several ``*_op``
    functions from this module, so a top-level import here would be circular.

    Never raises. By the time this is called, the actual index/delete already
    happened — a reporting failure only means NADA's own bookkeeping falls
    behind, not that anything already indexed/deleted needs to be undone.
    """
    if not settings.report_search_index_state_enabled or not items:
        return
    try:
        from nada_ai.ingest.search_index_sync import report_state_bulk

        report_state_bulk(settings, items)
    except Exception as e:  # noqa: BLE001 - see docstring
        logger.warning("search_index_state report failed for %d item(s): %s", len(items), e)


def delete_by_idno_op(settings: Settings, idno: str) -> dict[str, Any]:
    """Delete all indexed documents/points for an idno. Works with both backends."""
    if settings.search_backend == "qdrant":
        result = _delete_qdrant(settings, idno)
    else:
        result = _delete_opensearch(settings, idno)
    _report_state_bulk_best_effort(settings, [{"object_type": "survey", "object_key": idno, "status": "deleted"}])
    return result


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
        result = _delete_qdrant_batch(settings, idnos)
    else:
        result = _delete_opensearch_batch(settings, idnos)
    _report_state_bulk_best_effort(
        settings, [{"object_type": "survey", "object_key": i, "status": "deleted"} for i in idnos]
    )
    return result


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
    buffer_size: int = 200,
    embedding: EmbeddingService | None = None,
) -> dict[str, Any]:
    """Bulk-index the given idnos for a single metadata_type.

    ``embedding`` — pass the app's shared :class:`EmbeddingService` to avoid
    reloading the model for every job.  ``None`` (default) self-loads.

    Returns ``{"indexed", "errors", "load_errors", "empty_docs", "requested",
    "metadata_type", "index", "quality"}``. ``quality`` is a non-blocking report
    of thin/malformed source documents (empty content, missing idno/type)
    *that were built* — see ``ingest/quality.py``. ``empty_docs`` is idnos that
    loaded without error but produced zero documents to even check (no
    langdocs, or all-empty content) — distinct from ``quality``, which never
    sees these since no source document ever existed to observe. Neither
    affects what gets indexed.
    """
    pairs = [(i, metadata_type) for i in idnos]
    report = QualityReport()
    load_errors: list[dict[str, Any]] = []
    empty_docs: list[dict[str, Any]] = []
    n, err = run_bulk_index(
        settings,
        pairs,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
        embedding=embedding,
        quality_report=report,
        load_errors=load_errors,
        empty_docs=empty_docs,
    )
    _report_state_bulk_best_effort(settings, _state_report_items(idnos, load_errors, empty_docs, err))

    idx = settings.qdrant_collection if settings.search_backend == "qdrant" else settings.index_name
    return {
        "indexed": int(n),
        "errors": err or [],
        "load_errors": load_errors,
        "empty_docs": empty_docs,
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
    buffer_size: int = 200,
    embedding: EmbeddingService | None = None,
    resume: bool = False,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Fetch ids from Data Compass search API and bulk-index them.

    Returns ``{"indexed", "errors", "load_errors", "empty_docs", "rows",
    "resumed_skipped", "cancelled", "catalog_type", "index", "quality"}``.
    ``errors`` is write-time failures against the search backend; ``load_errors``
    is per-idno failures fetching/parsing metadata *before* a document was even
    built (previously silently logged and dropped — see ``ingest/pipeline.py``).
    ``empty_docs`` is idnos that loaded without error but produced zero
    documents (no langdocs, or all-empty content) — without this, ``indexed``
    could be well below ``rows`` with both ``errors`` and ``load_errors`` empty
    and no explanation anywhere for the gap. ``quality`` is a non-blocking
    report of thin/malformed source documents *that were built* — see
    ``ingest/quality.py``. None of these three ever affect what gets indexed.

    ``resume=True`` loads any existing checkpoint for this ``catalog_type``
    (see ``ingest/progress.py``) and skips idnos it already recorded as done —
    use this to continue a run that was cancelled or crashed partway through
    instead of reindexing everything again. ``progress_cb``, if given, is
    called after every idno with a live snapshot (processed/total/failed);
    ``cancel_token``, checked once per idno, is what makes cancelling this job
    actually stop promptly instead of running the remaining catalog anyway.
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
    all_pairs: list[tuple[str, str]] = []
    for row in rows:
        idno = row.get("idno")
        t = row.get("type")
        if not idno or not t:
            continue
        all_pairs.append((idno, t))

    checkpoint = load_checkpoint(settings, catalog_type) if resume else None
    pairs = all_pairs
    if checkpoint is not None:
        pairs = [(idno, t) for idno, t in all_pairs if idno not in checkpoint.completed_idnos]

    tracker = IngestProgressTracker(
        settings,
        catalog_type,
        total=len(all_pairs),
        checkpoint=checkpoint,
        on_update=progress_cb,
    )

    report = QualityReport()
    load_errors: list[dict[str, Any]] = []
    empty_docs: list[dict[str, Any]] = []
    try:
        n, err = run_bulk_index(
            settings,
            pairs,
            force=force,
            recreate_index=recreate_index,
            show_progress_bar=show_progress_bar,
            buffer_size=buffer_size,
            embedding=embedding,
            quality_report=report,
            progress=tracker,
            cancel_token=cancel_token,
            load_errors=load_errors,
            empty_docs=empty_docs,
        )
    except Exception:
        # Didn't run to completion (unexpected error, not a per-idno one
        # already handled inside the pipeline) — keep the checkpoint so a
        # follow-up resume=True run doesn't lose whatever *did* complete.
        tracker.finalize(completed=False)
        raise

    cancelled = cancel_token is not None and cancel_token.is_set()
    # Ran through the whole (possibly resumed) list, cancellation aside — even
    # if individual idnos failed, that's already captured in errors/load_errors
    # above, so there's nothing left worth resuming; clear the checkpoint.
    tracker.finalize(completed=not cancelled)

    _report_state_bulk_best_effort(
        settings, _state_report_items(tracker.checkpoint.completed_idnos, load_errors, empty_docs, err)
    )

    idx = settings.qdrant_collection if settings.search_backend == "qdrant" else settings.index_name
    return {
        "indexed": int(n),
        "errors": err or [],
        "load_errors": load_errors,
        "empty_docs": empty_docs,
        "rows": len(all_pairs),
        "resumed_skipped": len(all_pairs) - len(pairs),
        "cancelled": cancelled,
        "catalog_type": catalog_type,
        "index": idx,
        "quality": report.to_dict(),
    }
