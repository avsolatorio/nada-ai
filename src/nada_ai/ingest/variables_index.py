"""Indexing for the variable search (lexical only — see ``docs/variables-search-contract.md``).

Deliberately a standalone module, not part of the chunk/embedding ingestion pipeline (``pipeline.py``,
``opensearch_writer.py``): a variable document is a flat denormalization of one DB row, with no chunking and no
embedding, so none of that machinery applies.

Two operations: ``sync_survey_variables_op`` replaces one study's variables (called by a full study index, by NADA's
``change_class=variables`` queue signal, and by the variables-only job) and ``backfill_variables_op`` walks the whole
catalog. Both fetch from NADA a page at a time, so neither needs a study's or the catalog's variables in memory at
once. ``POST /admin/variables/sync`` runs either as a background job; the CLI runs them directly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import chain
from typing import Any

import ai4data.discovery.catalog.extract as catalog_extract
from opensearchpy.helpers import bulk

from nada_ai.ingest.batching import batched
from nada_ai.ingest.extract_access import ExtractError, request_kwargs
from nada_ai.ingest.progress import CancelToken
from nada_ai.nada.admin_auth import scrub_admin_credentials
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.mapping import new_index_generation, variables_index_body
from nada_ai.search.backend.opensearch.variables import variable_bulk_action
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def ensure_variables_index(client: Any, settings: Settings) -> None:
    """Create the variable index if missing (stamped with a generation), else leave it as is."""
    name = settings.variables_index
    if client.indices.exists(index=name):
        return
    body = variables_index_body()
    body["mappings"]["_meta"] = {"generation": new_index_generation()}
    client.indices.create(index=name, body=body)


def _bulk_actions(index: str, variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = []
    for variable in variables:
        core_fields = variable.get("core_fields")
        filters = variable.get("filters")
        if not isinstance(core_fields, dict) or not isinstance(filters, dict):
            continue
        uid = core_fields.get("uid")
        if uid is None:
            continue
        actions.append(variable_bulk_action(index, int(uid), core_fields, filters))
    return actions


def _write_batch(client: Any, index: str, batch: list[dict[str, Any]]) -> tuple[int, list[Any]]:
    """Bulk-write one page of variable documents; ``(indexed, errors)``. Refreshes once at the end of a run, not here."""
    actions = _bulk_actions(index, batch)
    if not actions:
        return 0, []
    success, errors = bulk(client, actions, raise_on_error=False, refresh=False)
    return int(success), list(errors) if isinstance(errors, list) else []


def sync_survey_variables_op(
    settings: Settings,
    idno: str,
    *,
    page_size: int = 1000,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Replace one study's variables in the index (by its NADA idno): delete them, then write every page of them.

    Best fit after that study's own document is (re)indexed, so a search never shows a variable of an unpublished or
    deleted study. The study's variables are fetched a page at a time (a study can have far more than fit in one
    response), and the first page is fetched **before** anything is deleted: a study that cannot be read (unknown
    idno, NADA unreachable) leaves its existing variables alone instead of emptying them. A failure after that first
    page leaves the study partially indexed and raises; running the sync again repairs it, since each variable's
    ``_id`` is its ``uid``.

    Returns ``{"idno", "indexed", "errors", "cancelled"}``; ``cancelled`` is true when ``cancel_token`` stopped the
    pages early.
    """
    client = build_client(settings)
    try:
        ensure_variables_index(client, settings)

        try:
            pages = batched(
                catalog_extract.iter_extract_survey_variables(idno, page_size=page_size, **request_kwargs(settings)),
                page_size,
            )
            first = next(pages, ())
        except Exception as e:
            raise ExtractError(scrub_admin_credentials(str(e))) from e

        if cancel_token is not None and cancel_token.is_set():
            return {"idno": idno, "indexed": 0, "errors": [], "cancelled": True}
        client.delete_by_query(index=settings.variables_index, body={"query": {"term": {"idno": idno}}}, refresh=True)

        indexed = 0
        errors: list[Any] = []
        cancelled = False
        try:
            for batch in chain([first] if first else [], pages):
                if cancel_token is not None and cancel_token.is_set():
                    cancelled = True
                    break
                written, batch_errors = _write_batch(client, settings.variables_index, list(batch))
                indexed += written
                errors.extend(batch_errors)
        except Exception as e:
            raise ExtractError(scrub_admin_credentials(str(e))) from e

        client.indices.refresh(index=settings.variables_index)
        return {"idno": idno, "indexed": indexed, "errors": errors, "cancelled": cancelled}
    finally:
        client.close()


def backfill_variables_op(
    settings: Settings,
    *,
    batch_size: int = 1000,
    max_records: int | None = None,
    recreate_index: bool = False,
    show_progress_bar: bool = True,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Page through every variable in the catalog and (re)index it. Run once to populate the index, or again after
    a catalog-wide change: each document's ``_id`` is its ``uid``, so re-running is idempotent.

    The catalog is walked with a keyset cursor (``after_uid``), so a page costs the same however deep it is, and
    NADA counts the catalog once, on the first page. ``progress_cb`` receives the job-registry progress shape
    (``processed``/``total``/``failed``/``percent``) after every page. This adds and replaces variables; it does not
    remove variables that no longer exist in NADA — deleting a study does that (see ``delete_by_idno_op``), and
    ``sync_survey_variables_op`` replaces one study's variables outright.

    Returns ``{"seen", "indexed", "errors", "total", "cancelled"}``.
    """
    client = build_client(settings)
    try:
        if cancel_token is not None and cancel_token.is_set():
            return {"seen": 0, "indexed": 0, "errors": [], "total": None, "cancelled": True}
        if recreate_index and client.indices.exists(index=settings.variables_index):
            client.indices.delete(index=settings.variables_index)
        ensure_variables_index(client, settings)

        indexed = 0
        errors: list[Any] = []
        seen = 0
        cancelled = False
        total: int | None = None

        def on_page(data: dict[str, Any]) -> None:
            nonlocal total
            if total is None and isinstance(data.get("total"), int):
                total = data["total"]

        pbar: Any = None
        if show_progress_bar:
            from tqdm.auto import tqdm

            pbar = tqdm(unit="variable", desc="Index variables")
        try:
            variables = catalog_extract.iter_extract_variables(
                page_size=batch_size, max_items=max_records, on_page=on_page, **request_kwargs(settings)
            )
            for batch in batched(variables, batch_size):
                if cancel_token is not None and cancel_token.is_set():
                    cancelled = True
                    break
                written, batch_errors = _write_batch(client, settings.variables_index, list(batch))
                indexed += written
                errors.extend(batch_errors)
                seen += len(batch)
                if pbar is not None:
                    pbar.update(len(batch))
                if progress_cb is not None:
                    progress_cb(
                        {
                            "processed": seen,
                            "total": total,
                            "failed": len(errors),
                            "percent": round(100 * seen / total, 1) if total else None,
                        }
                    )
        except Exception as e:
            raise ExtractError(scrub_admin_credentials(str(e))) from e
        finally:
            if pbar is not None:
                pbar.close()

        client.indices.refresh(index=settings.variables_index)
        return {"seen": seen, "indexed": indexed, "errors": errors, "total": total, "cancelled": cancelled}
    finally:
        client.close()
