"""Reusable ingest operations shared by the CLI and the FastAPI admin router.

Each ``*_op`` returns a small dict suitable for HTTP responses or job results, so
callers (CLI, API) just stringify or store the dict instead of duplicating the
logic. The CLI in :mod:`nada_ai.ingest.cli` is a thin ``print`` wrapper.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from nada_ai.ingest.pipeline import run_bulk_index
from nada_ai.ingest.progress import CancelToken, IngestProgressTracker, load_checkpoint
from nada_ai.ingest.quality import QualityReport
from nada_ai.ingest.study_sources import StudySource
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


def _error_by_sid(load_errors: list[dict[str, Any]], empty_docs: list[dict[str, Any]]) -> dict[int, str]:
    """Map failed sid -> a human-readable reason, for ``last_error`` reporting.

    ``load_errors`` entries carry the real exception text; ``empty_docs`` ones
    don't (they didn't raise), so fall back to their ``reason`` code.
    """
    errors = {int(e["sid"]): str(e["error"]) for e in load_errors if e.get("sid")}
    for e in empty_docs:
        errors.setdefault(int(e["sid"]), f"no documents produced ({e.get('reason', 'empty')})")
    return errors


def _attribute_write_error(err: Any) -> tuple[int | None, str]:
    """Best-effort ``(sid, message)`` for one backend write-error entry.

    Qdrant writer entries carry a ``sid`` key directly. OpenSearch ``bulk``
    entries are ``{"index": {"_id", "error", "data": <source doc>}}`` — the sid
    is read from the echoed source: ``metadata.sid`` on a chunk, ``sid`` on a
    study document. ``sid`` is ``None`` when the entry can't be tied to one
    study (e.g. an error that aborted the whole run partway through).
    """
    if not isinstance(err, dict):
        return None, str(err)
    sid = err.get("sid")
    message = str(err.get("error") or err)
    if not sid:
        for op in ("index", "create", "update"):
            body = err.get(op)
            if not isinstance(body, dict):
                continue
            message = str(body.get("error") or message)
            data = body.get("data")
            if isinstance(data, dict):
                meta = data.get("metadata")
                sid = meta.get("sid") if isinstance(meta, dict) else data.get("sid")
            break
    try:
        return (int(sid) if sid else None), message
    except (TypeError, ValueError):
        return None, message


def _state_item(sid: int, idno: str | None, status: str, error: str | None = None) -> dict[str, Any]:
    """One ``search_index_state`` report: keyed by ``object_id`` (the sid); ``object_key`` (the idno) is informative."""
    item: dict[str, Any] = {"object_type": "survey", "object_id": int(sid), "status": status}
    if idno:
        item["object_key"] = idno
    if error is not None:
        item["error"] = error
    return item


def _state_report_items(
    idno_by_sid: dict[int, str | None],
    load_errors: list[dict[str, Any]],
    empty_docs: list[dict[str, Any]],
    write_errors: list[Any] | None,
    variables_failed: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Build the ``search_index_state`` items for one indexing call: one per study of ``idno_by_sid`` (sid -> idno,
    the idno when known).

    ``load_errors``/``empty_docs``, backend ``write_errors`` and ``variables_failed`` (sid -> reason, see
    :func:`_sync_variables_best_effort`) are all failures — a study whose documents never made it into the backend
    must not be reported ``indexed``, or NADA drops it from its "missing" diff and reconcile never retries it. A full
    index of a study is study + chunks + variables, so a study whose variables failed is not indexed either.

    Write errors that can't be tied to a study (see :func:`_attribute_write_error`)
    mean we can't tell which studies are affected, so in that case nothing is
    reported ``indexed`` for this call — those studies stay "missing" in NADA and
    a later reconcile retries them, rather than being falsely marked done.
    """
    failed: dict[int, str] = {}
    fully_attributed = True
    for err in write_errors or []:
        sid, message = _attribute_write_error(err)
        if sid is None:
            fully_attributed = False
        else:
            failed.setdefault(sid, f"write failed: {message}")
    failed.update({sid: f"variables: {reason}" for sid, reason in (variables_failed or {}).items()})
    failed.update(_error_by_sid(load_errors, empty_docs))

    items: list[dict[str, Any]] = []
    if fully_attributed:
        items += [_state_item(sid, idno, "indexed") for sid, idno in idno_by_sid.items() if sid not in failed]
    else:
        logger.warning(
            "Backend reported write errors not attributable to a study; not reporting any study as 'indexed' "
            "to search_index_state for this call (they stay 'missing' so a reconcile retries them)"
        )
    items += [_state_item(sid, idno_by_sid.get(sid), "failed", m) for sid, m in failed.items()]
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


def delete_by_sid_op(settings: Settings, sid: int) -> dict[str, Any]:
    """Delete every indexed document/point of one study, keyed by the NADA internal id ``sid``."""
    return delete_by_sids_op(settings, [sid])


def delete_by_sids_op(settings: Settings, sids: list[int]) -> dict[str, Any]:
    """Delete every indexed document/point of the given NADA internal study ids, and report them ``deleted`` to
    NADA's ``search_index_state``. Works with both backends.

    The ``sid`` is the key nada-ai stores in ``metadata.sid`` on each document (see ``search.documents``), ``sid`` on
    the study and variable documents.
    """
    clean = list(dict.fromkeys(int(s) for s in sids))
    if not clean or any(s <= 0 for s in clean):
        raise ValueError("sids must be a non-empty list of positive integers")
    if settings.search_backend == "qdrant":
        result = _delete_qdrant_by_sids(settings, clean)
    else:
        result = _delete_opensearch_by_sids(settings, clean)
    _report_state_bulk_best_effort(settings, [_state_item(sid, None, "deleted") for sid in clean])
    return result


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
    """Delete studies from all three OpenSearch indexes (chunks, studies, variables) by ``sid``: every document of a
    study carries it (``metadata.sid`` on chunks, ``sid`` on study and variable documents)."""
    from nada_ai.search.backend.opensearch.mapping import metadata_field

    client = build_client(settings)
    try:

        def delete(index: str, query: dict[str, Any]) -> dict[str, Any]:
            return client.delete_by_query(index=index, body={"query": query}, refresh=True, ignore_unavailable=True)

        chunks = delete(settings.index_name, {"terms": {metadata_field("sid"): sids}})
        studies = delete(settings.studies_index, {"terms": {"sid": sids}})
        # A deleted (or unpublished-and-removed) study's variables must not stay searchable.
        variables = delete(settings.variables_index, {"terms": {"sid": sids}})
        return {
            "backend": "opensearch",
            "index": settings.index_name,
            "sids": sids,
            "deleted": int(chunks.get("deleted") or 0),
            "total": chunks.get("total"),
            "studies_index": settings.studies_index,
            "studies_deleted": int(studies.get("deleted") or 0),
            "variables_index": settings.variables_index,
            "variables_deleted": int(variables.get("deleted") or 0),
        }
    finally:
        _close_quiet(client)


def apply_study_options_op(settings: Settings, sid: int) -> dict[str, Any]:
    """Apply a change of a study's options to the index without reading its metadata or embedding anything.

    NADA sends ``upsert_partial`` for changes to a study's options (publish state, license, data class, links, DOI,
    featured, collections): fields around the metadata, never its text, so the study's chunks keep their vectors.
    This rewrites the study document and replaces the flat ``filter_facets`` on the study's chunks, both from NADA's
    extract record. Its variables are not touched: variable search reads the study's state from the study index.
    OpenSearch only.

    A study with no document in the index has nothing to update: ``{"updated": False, "reason": "not_indexed"}``, and
    the caller indexes it in full.
    """
    from nada_ai.ingest.study_sources import fetch_studies_by_sids

    batch = fetch_studies_by_sids(settings, [sid])
    if batch["errors"]:
        raise ValueError(f"NADA could not build the extract of study {sid}: {batch['errors'][0].get('message')}")
    if not batch["studies"]:
        raise ValueError(f"Study {sid} is not in NADA's catalog")
    record = batch["studies"][0]
    return apply_study_options_from_extract(settings, sid, record.get("core_fields") or {}, record.get("filters") or {})


def apply_study_options_from_extract(
    settings: Settings, sid: int, core_fields: dict[str, Any], filters: dict[str, Any]
) -> dict[str, Any]:
    """:func:`apply_study_options_op` for an extract record already fetched (the scheduler fetches a page of them in
    one request).

    NADA sends ``all_text`` (every metadata value) with full updates only, so the study document keeps the one it
    has: it is read back from the stored field and written again unchanged.
    """
    from nada_ai.search.backend.opensearch.mapping import FILTER_FACETS_KEY, STUDY_ALL_TEXT_FIELD, metadata_field
    from nada_ai.search.backend.opensearch.studies import study_to_source

    client = build_client(settings)
    try:
        if not client.exists(index=settings.studies_index, id=str(sid)):
            return {"sid": sid, "updated": False, "reason": "not_indexed"}

        source = study_to_source(sid, core_fields, filters)
        if STUDY_ALL_TEXT_FIELD not in core_fields:
            kept = _stored_all_text(client, settings.studies_index, sid)
            if kept:
                source[STUDY_ALL_TEXT_FIELD] = kept
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
        return {"sid": sid, "updated": True, "chunks_updated": int(chunks.get("updated") or 0)}
    finally:
        _close_quiet(client)


def _stored_all_text(client: Any, index: str, sid: int) -> list[str]:
    """The ``all_text`` the study document holds (a stored field: it is not in ``_source``)."""
    from nada_ai.search.backend.opensearch.mapping import STUDY_ALL_TEXT_FIELD

    doc = client.get(index=index, id=str(sid), stored_fields=STUDY_ALL_TEXT_FIELD, _source=False)
    return list((doc.get("fields") or {}).get(STUDY_ALL_TEXT_FIELD) or [])


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
    sids: Iterable[int],
    *,
    cancel_token: CancelToken | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Sync each study's variables (best-effort: one study's failure doesn't stop the rest, and never raises —
    a full study index must not stop just because the separate variable index couldn't be reached).

    Returns ``{"indexed", "errors", "failed"}``; ``failed`` maps each sid whose variables were not fully synced to
    the reason — an error, write errors, or a cancellation (the studies not reached count too) — so the caller
    reports that study failed rather than indexed.

    Only called for ``metadata_type=microdata`` (variables only exist on that dataset type — see
    ``_metadata_type`` in NADA's ``Semantic.php`` for the same map). By design, a full index of a study is
    study + chunks + variables together; see ``docs/variables-search-contract.md``. ``progress_cb`` receives the
    job-registry progress shape after every study.
    """
    from nada_ai.ingest.variables_index import sync_survey_variables_op

    todo = list(sids)
    indexed = 0
    errors: list[Any] = []
    failed: dict[int, str] = {}
    for done, sid in enumerate(todo, start=1):
        if cancel_token is not None and cancel_token.is_set():
            failed.update({i: "cancelled before its variables were synced" for i in todo[done - 1 :]})
            break
        try:
            result = sync_survey_variables_op(settings, sid, cancel_token=cancel_token)
            indexed += int(result.get("indexed") or 0)
            errors.extend(result.get("errors") or [])
            if result.get("errors"):
                failed[sid] = f"{len(result['errors'])} write error(s)"
            elif result.get("cancelled"):
                failed[sid] = "cancelled while its variables were being synced"
        except Exception as e:  # noqa: BLE001 - reported in the result, not raised
            logger.warning("variable sync failed for study %s: %s", sid, e)
            errors.append({"sid": sid, "error": str(e)})
            failed[sid] = str(e)
        if progress_cb is not None:
            progress_cb(
                {
                    "processed": done,
                    "total": len(todo),
                    "failed": len(failed),
                    "percent": round(100 * done / len(todo), 1),
                    "current_sid": sid,
                }
            )
    return {"indexed": indexed, "errors": errors, "failed": failed}


#: Errors kept in a variables-only job's result: a bulk failure echoes whole documents, and a job's result is stored
#: in memory and sent to the dashboard.
_JOB_ERROR_LIMIT = 50


def sync_variables_op(
    settings: Settings,
    sids: list[int] | None = None,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Index variables only — no study document, no chunks, so no embedding.

    ``sids`` replaces those studies' variables (one NADA page-walk each). ``None`` walks the whole catalog's
    variables instead, which adds and replaces but does not remove variables that no longer exist (see
    ``backfill_variables_op``). Returns ``{"scope", "indexed", "errors", "error_count", "cancelled", ...}``.
    """
    if sids is None:
        from nada_ai.ingest.variables_index import backfill_variables_op

        result = backfill_variables_op(
            settings, show_progress_bar=False, progress_cb=progress_cb, cancel_token=cancel_token
        )
        errors = result.pop("errors")
        return {"scope": "all", **result, "errors": errors[:_JOB_ERROR_LIMIT], "error_count": len(errors)}

    result = _sync_variables_best_effort(settings, sids, cancel_token=cancel_token, progress_cb=progress_cb)
    errors = result["errors"]
    return {
        "scope": "sids",
        "requested": len(sids),
        "indexed": result["indexed"],
        "errors": errors[:_JOB_ERROR_LIMIT],
        "error_count": len(errors),
        "cancelled": cancel_token is not None and cancel_token.is_set(),
    }


def _recreate(settings: Settings, recreate: Callable[[Any, Settings], str]) -> str:
    from nada_ai.search.backend.opensearch.client import build_client

    client = build_client(settings)
    try:
        return recreate(client, settings)
    finally:
        client.close()


def recreate_variables_op(
    settings: Settings,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Drop the variable index, create it empty with the current mapping, and index every variable again.

    Variable search has little or nothing to find until the walk is done. Returns ``sync_variables_op``'s result for
    the whole catalog, plus ``recreated`` (the index) and ``generation``.
    """
    from nada_ai.ingest.variables_index import recreate_variables_index

    generation = _recreate(settings, recreate_variables_index)
    logger.info(
        "variable index %s recreated (generation %s); indexing every variable", settings.variables_index, generation
    )
    result = sync_variables_op(settings, None, progress_cb=progress_cb, cancel_token=cancel_token)
    return {"recreated": settings.variables_index, "generation": generation, **result}


def recreate_citations_op(
    settings: Settings,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Drop the citation index, create it empty with the current mapping, and index every citation again.

    Citation search has little or nothing to find until the walk is done. Returns ``sync_citations_op``'s result for
    the whole catalog, plus ``recreated`` (the index) and ``generation``.
    """
    from nada_ai.ingest.citations_index import recreate_citations_index

    generation = _recreate(settings, recreate_citations_index)
    logger.info(
        "citation index %s recreated (generation %s); indexing every citation", settings.citations_index, generation
    )
    result = sync_citations_op(settings, None, progress_cb=progress_cb, cancel_token=cancel_token)
    return {"recreated": settings.citations_index, "generation": generation, **result}


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
                    "current_citation_id": citation_id,
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


class _Seen:
    """Passes studies through, noting each one's idno by sid (for the state report) once the pipeline is done with it.

    A study is noted when the pipeline asks for the next one (or the sources run out), not when it is handed over: a
    cancelled pipeline takes the next study, sees the cancel and stops without loading it, and that study must not be
    reported indexed. What the pipeline is done with is written before it returns (its last buffer is flushed)."""

    def __init__(self, sources: Iterable[StudySource]) -> None:
        self._sources = sources
        self.idno_by_sid: dict[int, str | None] = {}
        self.microdata: list[int] = []

    def __iter__(self) -> Iterator[StudySource]:
        for source in self._sources:
            yield source
            self.idno_by_sid[source.sid] = source.idno
            if source.metadata_type == "microdata":
                self.microdata.append(source.sid)


def _index_sources(
    settings: Settings,
    sources: Iterable[StudySource],
    requested: Iterable[int],
    *,
    force: bool,
    recreate_index: bool,
    show_progress_bar: bool,
    buffer_size: int,
    embedding: EmbeddingService | None,
    load_errors: list[dict[str, Any]],
    progress: IngestProgressTracker | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Index ``sources`` in full (study, chunks and, for microdata, variables) and report every study of
    ``requested`` (sids) to NADA's ``search_index_state``. ``load_errors`` may already hold the studies that could
    not be fetched."""
    report = QualityReport()
    empty_docs: list[dict[str, Any]] = []
    seen = _Seen(sources)
    n, err = run_bulk_index(
        settings,
        seen,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
        embedding=embedding,
        quality_report=report,
        progress=progress,
        cancel_token=cancel_token,
        load_errors=load_errors,
        empty_docs=empty_docs,
    )
    cancelled = cancel_token is not None and cancel_token.is_set()

    # A full index of a study is study + chunks + variables. Variables only exist on microdata, so this is one NADA
    # call per microdata study of this run. A cancelled run stops here rather than starting a long sync nobody is
    # waiting for; its microdata studies are then reported failed, so a reconcile retries them.
    variables_result: dict[str, Any] | None = None
    variables_failed: dict[int, str] = {}
    if settings.search_backend == "opensearch":
        load_failed = {int(e["sid"]) for e in load_errors if e.get("sid")}
        microdata = [sid for sid in seen.microdata if sid not in load_failed]
        if cancelled:
            variables_failed = {sid: "cancelled before its variables were synced" for sid in microdata}
        elif microdata:
            variables_result = _sync_variables_best_effort(settings, microdata, cancel_token=cancel_token)
            variables_failed = variables_result["failed"]

    # Reported after the variables: a study whose variables failed is not indexed (see _state_report_items).
    idno_by_sid = {int(sid): None for sid in requested}
    idno_by_sid.update(seen.idno_by_sid)
    _report_state_bulk_best_effort(
        settings, _state_report_items(idno_by_sid, load_errors, empty_docs, err, variables_failed)
    )

    idx = settings.qdrant_collection if settings.search_backend == "qdrant" else settings.index_name
    return {
        "indexed": int(n),
        "errors": err or [],
        "load_errors": load_errors,
        "empty_docs": empty_docs,
        "cancelled": cancelled,
        "index": idx,
        "quality": report.to_dict(),
        "variables": variables_result,
    }


def index_sids_op(
    settings: Settings,
    sids: list[int],
    force: bool = False,
    recreate_index: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 200,
    embedding: EmbeddingService | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Index these studies (NADA internal ids) in full: their records are fetched from NADA by id.

    ``records``: their full extract records when the caller already has them (nothing is fetched then).
    ``embedding`` — pass the app's shared :class:`EmbeddingService` to avoid reloading the model for every job.
    ``force`` embeds every chunk again (no stored vector is reused).

    Returns ``{"indexed", "errors", "load_errors", "empty_docs", "requested", "index", "quality", "variables"}``.
    ``load_errors`` are the studies that could not be fetched or loaded (``sid``, ``idno``, ``stage``, ``error``);
    ``empty_docs`` those that loaded but produced no documents; ``quality`` is a non-blocking report of thin or
    malformed source documents (see ``ingest/quality.py``); ``variables`` is ``{"indexed", "errors", "failed"}`` for
    the microdata studies, else ``None``.
    """
    from nada_ai.ingest.study_sources import sources_by_sids, sources_from_records

    sids = list(dict.fromkeys(int(s) for s in sids))
    load_errors: list[dict[str, Any]] = []
    sources = (
        sources_from_records(records, load_errors)
        if records is not None
        else sources_by_sids(settings, sids, load_errors)
    )
    result = _index_sources(
        settings,
        sources,
        sids,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
        embedding=embedding,
        load_errors=load_errors,
    )
    result.pop("cancelled")
    return {**result, "requested": len(sids)}


#: Catalog types of a full run -> NADA's ``surveys.type``.
_CATALOG_TYPE_TO_NADA_TYPE: dict[str, str] = {
    "timeseries": "timeseries",
    "indicator": "timeseries",
    "microdata": "survey",
    "survey": "survey",
    "indicator-db": "timeseriesdb",
    "timeseries-db": "timeseriesdb",
    "timeseriesdb": "timeseriesdb",
    "document": "document",
    "geospatial": "geospatial",
    "table": "table",
    "script": "script",
    "image": "image",
    "video": "video",
}


def index_from_catalog_op(
    settings: Settings,
    catalog_type: str = "timeseries",
    ps: int = 20,
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
    """Index every study of ``catalog_type``, paged from NADA's metadata-extract API in id order, ``ps`` full records
    per request; ``limit`` stops after that many studies.

    Returns ``{"indexed", "errors", "load_errors", "empty_docs", "rows", "resumed_skipped", "cancelled",
    "catalog_type", "index", "quality", "variables"}``. ``errors`` contains backend write failures; ``load_errors``
    the studies that could not be loaded; ``empty_docs`` those that produced no indexable content. ``quality`` reports
    thin or malformed source documents without blocking indexing.

    ``resume=True`` loads any existing checkpoint for this ``catalog_type`` (see ``ingest/progress.py``) and skips the
    studies (by sid) it already recorded as done — use this to continue a run that was cancelled or crashed partway
    through. ``progress_cb``, if given, is called after every study with a live snapshot (processed/total/failed);
    ``cancel_token``, checked once per study, makes cancelling this job stop promptly.
    """
    from nada_ai.ingest.study_sources import fetch_study_page, sources_from_records

    if resume and recreate_index:
        raise ValueError("resume cannot be combined with recreate_index")
    if catalog_type not in _CATALOG_TYPE_TO_NADA_TYPE:
        raise ValueError(f"Unknown catalog_type {catalog_type!r}; one of {sorted(_CATALOG_TYPE_TO_NADA_TYPE)}")
    nada_type = _CATALOG_TYPE_TO_NADA_TYPE[catalog_type]

    first = fetch_study_page(settings, offset=0, limit=ps, nada_type=nada_type)
    total = first["total"] if limit is None else min(first["total"], limit)

    checkpoint = load_checkpoint(settings, catalog_type) if resume else None
    done = checkpoint.completed_sids if checkpoint is not None else set()
    tracker = IngestProgressTracker(settings, catalog_type, total=total, checkpoint=checkpoint, on_update=progress_cb)

    rows = skipped = 0
    load_errors: list[dict[str, Any]] = []

    def records() -> Iterator[dict[str, Any]]:
        nonlocal rows, skipped
        page, offset = first, 0
        while True:
            for record in page["studies"]:
                if rows >= total:
                    return
                rows += 1
                if int((record.get("core_fields") or {}).get("catalog_id") or 0) in done:
                    skipped += 1
                    continue
                yield record
            offset += len(page["studies"])
            if not page["has_more"] or not page["studies"] or rows >= total:
                return
            if cancel_token is not None and cancel_token.is_set():
                return
            page = fetch_study_page(settings, offset=offset, limit=ps, nada_type=nada_type)

    try:
        result = _index_sources(
            settings,
            sources_from_records(records(), load_errors, on_skip=lambda sid, error: tracker.mark(sid, False, error)),
            [],
            force=force,
            recreate_index=recreate_index,
            show_progress_bar=show_progress_bar,
            buffer_size=buffer_size,
            embedding=embedding,
            load_errors=load_errors,
            progress=tracker,
            cancel_token=cancel_token,
        )
    except Exception:
        # Didn't run to completion (an unexpected error, not a per-study one already handled inside the pipeline) —
        # keep the checkpoint so a follow-up resume=True run doesn't lose whatever *did* complete.
        tracker.finalize(completed=False)
        raise
    # Ran through the whole (possibly resumed) list, cancellation aside: per-study failures are already in the
    # result, so there is nothing left worth resuming.
    tracker.finalize(completed=not result["cancelled"])
    return {**result, "rows": rows, "resumed_skipped": skipped, "catalog_type": catalog_type}
