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
from nada_ai.search.backend.opensearch.studies import sids_for_idnos
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
    variables_failed: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build the ``search_index_state`` items for one indexing call.

    ``load_errors``/``empty_docs``, backend ``write_errors`` and ``variables_failed`` (idno -> reason, see
    :func:`_sync_variables_best_effort`) are all failures — an idno whose documents never made it into the backend
    must not be reported ``indexed``, or NADA drops it from its "missing" diff and reconcile never retries it. A full
    index of a study is study + chunks + variables, so a study whose variables failed is not indexed either.

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
    failed.update({idno: f"variables: {reason}" for idno, reason in (variables_failed or {}).items()})
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
                filter=qm.Filter(must=[qm.FieldCondition(key=metadata_field("idno"), match=qm.MatchValue(value=idno))])
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
    out = _delete_opensearch_by(settings, idnos=[idno])
    return {"backend": "opensearch", "index": settings.index_name, "idno": idno, **out}


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
                filter=qm.Filter(must=[qm.FieldCondition(key=metadata_field("idno"), match=qm.MatchAny(any=idnos))])
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
    out = _delete_opensearch_by(settings, idnos=idnos)
    return {"backend": "opensearch", "index": settings.index_name, "idnos": idnos, **out}


def delete_by_sid_op(settings: Settings, sid: int) -> dict[str, Any]:
    """Delete every indexed document/point of one study, keyed by the NADA internal id ``sid``."""
    return delete_by_sids_op(settings, [sid])


def delete_by_sids_op(settings: Settings, sids: list[int]) -> dict[str, Any]:
    """Delete every indexed document/point of the given NADA internal study ids. Works with both backends.

    The ``sid`` is the key nada-ai stores in ``metadata.sid`` on each document (see ``search.documents``).

    Unlike the idno operations this does not report to NADA's ``search_index_state``, which is keyed by idno.
    """
    clean = list(dict.fromkeys(int(s) for s in sids))
    if not clean or any(s <= 0 for s in clean):
        raise ValueError("sids must be a non-empty list of positive integers")
    if settings.search_backend == "qdrant":
        return _delete_qdrant_by_sids(settings, clean)
    return _delete_opensearch_by_sids(settings, clean)


def _delete_qdrant_by_sids(settings: Settings, sids: list[int]) -> dict[str, Any]:
    from qdrant_client.http import models as qm

    from nada_ai.ingest.qdrant_writer import _client as make_client
    from nada_ai.search.backend.opensearch.mapping import metadata_field

    client = make_client(settings)
    coll = settings.qdrant_collection
    try:
        result = client.delete(
            collection_name=coll,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key=metadata_field("sid"), match=qm.MatchAny(any=sids))])
            ),
        )
        return {
            "backend": "qdrant",
            "collection": coll,
            "sids": sids,
            "operation": result.status.value if result else "unknown",
        }
    finally:
        client.close()


def _delete_opensearch_by_sids(settings: Settings, sids: list[int]) -> dict[str, Any]:
    out = _delete_opensearch_by(settings, sids=sids)
    return {"backend": "opensearch", "index": settings.index_name, "sids": sids, **out}


def _delete_opensearch_by(
    settings: Settings, *, idnos: list[str] | None = None, sids: list[int] | None = None
) -> dict[str, Any]:
    """Delete studies from all three OpenSearch indexes (chunks, studies, variables), by NADA idno and/or ``sid``.

    The study index stores NADA's own ``idno``, so a delete by idno first looks up the study's ``sid`` there;
    chunks are then removed by ``sid`` as well as by their stored idno (which comes from the record's schema and
    can differ from NADA's), so none are left behind. Variables carry NADA's own idno and sid, like studies.
    """
    from nada_ai.search.backend.opensearch.mapping import metadata_field

    idnos = idnos or []
    sid_set = set(sids or [])
    client = build_client(settings)
    try:
        sid_set.update(sids_for_idnos(client, settings.studies_index, idnos).values())

        chunk_match: list[dict[str, Any]] = []
        study_match: list[dict[str, Any]] = []
        if idnos:
            chunk_match.append({"terms": {metadata_field("idno"): idnos}})
            study_match.append({"terms": {"idno": idnos}})
        if sid_set:
            chunk_match.append({"terms": {metadata_field("sid"): sorted(sid_set)}})
            study_match.append({"terms": {"sid": sorted(sid_set)}})

        def delete(index: str, should: list[dict[str, Any]]) -> dict[str, Any]:
            body = {"query": {"bool": {"should": should, "minimum_should_match": 1}}}
            return client.delete_by_query(index=index, body=body, refresh=True, ignore_unavailable=True)

        chunks = delete(settings.index_name, chunk_match)
        studies = delete(settings.studies_index, study_match)
        # A deleted (or unpublished-and-removed) study's variables must not stay searchable: variable documents
        # carry the same NADA ``idno`` and ``sid`` as the study, at the document root.
        variables = delete(settings.variables_index, study_match)
        return {
            "deleted": int(chunks.get("deleted") or 0),
            "total": chunks.get("total"),
            "studies_index": settings.studies_index,
            "studies_deleted": int(studies.get("deleted") or 0),
            "variables_index": settings.variables_index,
            "variables_deleted": int(variables.get("deleted") or 0),
        }
    finally:
        _close_quiet(client)


def apply_study_options_op(settings: Settings, idno: str) -> dict[str, Any]:
    """Apply a change of a study's options to the index without reading its metadata or embedding anything.

    NADA sends ``upsert_partial`` for changes to a study's options (publish state, license, data class, links, DOI,
    featured, collections): fields around the metadata, never its text, so the study's chunks keep their vectors.
    This rewrites the study document, replaces the flat ``filter_facets`` on the study's chunks, and sets
    ``published`` on its variables, all from NADA's extract record. OpenSearch only.

    A study with no document in the index has nothing to update: ``{"updated": False, "reason": "not_indexed"}``, and
    the caller indexes it in full.
    """
    from nada_ai.filters.metadata_extract import fetch_study_extract
    from nada_ai.search.backend.opensearch.mapping import FILTER_FACETS_KEY, metadata_field
    from nada_ai.search.backend.opensearch.studies import study_to_source

    sid, core_fields, filters = fetch_study_extract(settings, idno)
    client = build_client(settings)
    try:
        if not sids_for_idnos(client, settings.studies_index, [idno]):
            return {"idno": idno, "updated": False, "reason": "not_indexed"}

        source = study_to_source(sid, core_fields, filters)
        client.index(index=settings.studies_index, id=str(sid), body=source, refresh=True)

        chunks = client.update_by_query(
            index=settings.index_name,
            body={
                "query": {"term": {metadata_field("sid"): sid}},
                "script": {
                    "lang": "painless",
                    "source": f"ctx._source.metadata.{FILTER_FACETS_KEY} = params.facets",
                    "params": {"facets": source[FILTER_FACETS_KEY]},
                },
            },
            refresh=True,
            conflicts="proceed",
        )
        published = source[FILTER_FACETS_KEY].get("published") or []
        variables = client.update_by_query(
            index=settings.variables_index,
            body={
                "query": {"term": {"sid": sid}},
                "script": {
                    "lang": "painless",
                    "source": "ctx._source.published = params.published",
                    "params": {"published": int(published[0]) if published else 0},
                },
            },
            refresh=True,
            conflicts="proceed",
            ignore_unavailable=True,
        )
        return {
            "idno": idno,
            "updated": True,
            "sid": sid,
            "chunks_updated": int(chunks.get("updated") or 0),
            "variables_updated": int(variables.get("updated") or 0),
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
    dim: int | None
    if settings.embedding_backend == "opensearch_ml":
        dim = int(settings.opensearch_ml_embedding_dimension or 0)
    elif settings.embedding_backend == "none":
        dim = None  # no model to load: this deployment never computes or stores a vector
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

    dim: int | None
    if settings.embedding_backend == "opensearch_ml":
        dim = int(settings.opensearch_ml_embedding_dimension or 0)
    elif settings.embedding_backend == "none":
        dim = None  # no model to load: this deployment never computes or stores a vector
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


def _sync_variables_best_effort(
    settings: Settings,
    idnos: Iterable[str],
    *,
    cancel_token: CancelToken | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Sync each idno's variables (best-effort: one idno's failure doesn't stop the rest, and never raises —
    a full study index must not stop just because the separate variable index couldn't be reached).

    Returns ``{"indexed", "errors", "failed"}``; ``failed`` maps each idno whose variables were not fully synced to
    the reason — an error, write errors, or a cancellation (the idnos not reached count too) — so the caller reports
    that study failed rather than indexed.

    Only called for ``metadata_type=microdata`` (variables only exist on that dataset type — see
    ``_metadata_type`` in NADA's ``Semantic.php`` for the same map). By design, a full index of a study is
    study + chunks + variables together; see ``docs/variables-search-contract.md``. ``progress_cb`` receives the
    job-registry progress shape after every study.
    """
    from nada_ai.ingest.variables_index import sync_survey_variables_op

    todo = list(idnos)
    indexed = 0
    errors: list[Any] = []
    failed: dict[str, str] = {}
    for done, idno in enumerate(todo, start=1):
        if cancel_token is not None and cancel_token.is_set():
            failed.update({i: "cancelled before its variables were synced" for i in todo[done - 1 :]})
            break
        try:
            result = sync_survey_variables_op(settings, idno, cancel_token=cancel_token)
            indexed += int(result.get("indexed") or 0)
            errors.extend(result.get("errors") or [])
            if result.get("errors"):
                failed[idno] = f"{len(result['errors'])} write error(s)"
            elif result.get("cancelled"):
                failed[idno] = "cancelled while its variables were being synced"
        except Exception as e:  # noqa: BLE001 - reported in the result, not raised
            logger.warning("variable sync failed for idno=%s: %s", idno, e)
            errors.append({"idno": idno, "error": str(e)})
            failed[idno] = str(e)
        if progress_cb is not None:
            progress_cb(
                {
                    "processed": done,
                    "total": len(todo),
                    "failed": len(failed),
                    "percent": round(100 * done / len(todo), 1),
                    "current_idno": idno,
                }
            )
    return {"indexed": indexed, "errors": errors, "failed": failed}


#: Errors kept in a variables-only job's result: a bulk failure echoes whole documents, and a job's result is stored
#: in memory and sent to the dashboard.
_JOB_ERROR_LIMIT = 50


def sync_variables_op(
    settings: Settings,
    idnos: list[str] | None = None,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Index variables only — no study document, no chunks, so no embedding.

    ``idnos`` replaces those studies' variables (one NADA page-walk each). ``None`` walks the whole catalog's
    variables instead, which adds and replaces but does not remove variables that no longer exist (see
    ``backfill_variables_op``). Returns ``{"scope", "indexed", "errors", "error_count", "cancelled", ...}``.
    """
    if idnos is None:
        from nada_ai.ingest.variables_index import backfill_variables_op

        result = backfill_variables_op(
            settings, show_progress_bar=False, progress_cb=progress_cb, cancel_token=cancel_token
        )
        errors = result.pop("errors")
        return {"scope": "all", **result, "errors": errors[:_JOB_ERROR_LIMIT], "error_count": len(errors)}

    result = _sync_variables_best_effort(settings, idnos, cancel_token=cancel_token, progress_cb=progress_cb)
    errors = result["errors"]
    return {
        "scope": "idnos",
        "requested": len(idnos),
        "indexed": result["indexed"],
        "errors": errors[:_JOB_ERROR_LIMIT],
        "error_count": len(errors),
        "cancelled": cancel_token is not None and cancel_token.is_set(),
    }


def sync_citations_op(
    settings: Settings,
    citation_ids: list[int] | None = None,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Index citations only (lexical: no chunks, no embedding).

    ``citation_ids`` (NADA citation ids) syncs those citations one by one: each is rewritten, or removed when NADA no
    longer has it. ``None`` walks the whole catalog's citations, which adds and replaces but does not remove (see
    ``backfill_citations_op``). Returns ``{"scope", "indexed", "deleted", "errors", "error_count", "cancelled", ...}``.
    """
    from nada_ai.ingest.citations_index import backfill_citations_op, sync_citation_op

    if citation_ids is None:
        result = backfill_citations_op(
            settings, show_progress_bar=False, progress_cb=progress_cb, cancel_token=cancel_token
        )
        errors = result.pop("errors")
        return {"scope": "all", **result, "errors": errors[:_JOB_ERROR_LIMIT], "error_count": len(errors)}

    indexed = deleted = failed = 0
    errors: list[Any] = []
    for done, citation_id in enumerate(citation_ids, start=1):
        if cancel_token is not None and cancel_token.is_set():
            break
        try:
            result = sync_citation_op(settings, citation_id)
            indexed += int(result.get("indexed") or 0)
            deleted += int(result.get("deleted") or 0)
            errors.extend(result.get("errors") or [])
            failed += 1 if result.get("errors") else 0
        except Exception as e:  # noqa: BLE001 - reported in the result, not raised
            logger.warning("citation sync failed for id=%s: %s", citation_id, e)
            errors.append({"citation_id": citation_id, "error": str(e)})
            failed += 1
        if progress_cb is not None:
            progress_cb(
                {
                    "processed": done,
                    "total": len(citation_ids),
                    "failed": failed,
                    "percent": round(100 * done / len(citation_ids), 1),
                    "current_idno": str(citation_id),
                }
            )
    return {
        "scope": "ids",
        "requested": len(citation_ids),
        "indexed": indexed,
        "deleted": deleted,
        "errors": errors[:_JOB_ERROR_LIMIT],
        "error_count": len(errors),
        "cancelled": cancel_token is not None and cancel_token.is_set(),
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
    "metadata_type", "index", "quality", "variables"}``. ``quality`` is a non-blocking report
    of thin/malformed source documents (empty content, missing idno/type)
    *that were built* — see ``ingest/quality.py``. ``empty_docs`` is idnos that
    loaded without error but produced zero documents to even check (no
    langdocs, or all-empty content) — distinct from ``quality``, which never
    sees these since no source document ever existed to observe. Neither
    affects what gets indexed. ``variables`` is ``{"indexed", "errors"}`` for
    the OpenSearch variable index (see ``_sync_variables_best_effort``); ``None``
    for any metadata_type other than ``microdata``, which never has variables.
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
    variables_result: dict[str, Any] | None = None
    if metadata_type == "microdata" and settings.search_backend == "opensearch":
        failed_idnos = {e["idno"] for e in load_errors if e.get("idno")}
        variables_result = _sync_variables_best_effort(settings, (i for i in idnos if i not in failed_idnos))
    # Reported after the variables: a study whose variables failed is not indexed (see _state_report_items).
    _report_state_bulk_best_effort(
        settings,
        _state_report_items(
            idnos, load_errors, empty_docs, err, variables_result["failed"] if variables_result else None
        ),
    )

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
        "variables": variables_result,
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
    "resumed_skipped", "cancelled", "catalog_type", "index", "quality", "variables"}``. ``variables`` is
    ``{"indexed", "errors"}`` for the microdata studies of this run (see ``_sync_variables_best_effort``), else ``None``.
    ``errors`` contains backend write failures; ``load_errors`` contains per-idno metadata fetch or parsing failures;
    ``empty_docs`` records studies that produced no indexable content. ``quality`` reports thin or malformed source
    documents without blocking indexing. These fields explain gaps between requested and indexed records.

    ``resume=True`` loads any existing checkpoint for this ``catalog_type``
    (see ``ingest/progress.py``) and skips idnos it already recorded as done —
    use this to continue a run that was cancelled or crashed partway through
    instead of reindexing everything again. ``progress_cb``, if given, is
    called after every idno with a live snapshot (processed/total/failed);
    ``cancel_token``, checked once per idno, is what makes cancelling this job
    actually stop promptly instead of running the remaining catalog anyway.
    """
    if resume and recreate_index:
        raise ValueError("resume cannot be combined with recreate_index")

    from ai4data.discovery.catalog import get_metadata_ids, is_extract_mode

    params: dict[str, Any] = {"sk": "", "ps": ps, "type": catalog_type, "sort_by": "year", "sort_order": "asc"}
    if catalog_type == "indicator":
        params["type"] = "timeseries"
    elif catalog_type == "microdata":
        params["type"] = "survey"
    elif catalog_type in ("indicator-db", "timeseries-db"):
        params["type"] = "timeseriesdb"

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

    # A full index of a study is study + chunks + variables (see index_ids_op). Variables only exist on microdata,
    # so this is one NADA call per microdata study of this run, not per catalog row — a run over thousands of
    # documents makes none. A cancelled run stops here rather than starting a long sync nobody is waiting for; its
    # microdata studies are then reported failed (their variables were never synced), so a reconcile retries them.
    variables_result: dict[str, Any] | None = None
    variables_failed: dict[str, str] = {}
    if settings.search_backend == "opensearch":
        failed_idnos = {e["idno"] for e in load_errors if e.get("idno")}
        microdata_idnos = [idno for idno, t in pairs if t == "microdata" and idno not in failed_idnos]
        if cancelled:
            variables_failed = {i: "cancelled before its variables were synced" for i in microdata_idnos}
        elif microdata_idnos:
            variables_result = _sync_variables_best_effort(settings, microdata_idnos, cancel_token=cancel_token)
            variables_failed = variables_result["failed"]

    # Reported after the variables: a study whose variables failed is not indexed (see _state_report_items).
    _report_state_bulk_best_effort(
        settings,
        _state_report_items(tracker.checkpoint.completed_idnos, load_errors, empty_docs, err, variables_failed),
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
        "variables": variables_result,
    }
