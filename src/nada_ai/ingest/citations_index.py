"""Indexing for the citation search (lexical only — see ``docs/citations-search-contract.md``).

A standalone module like ``variables_index.py``, for the same reason: a citation document is a flat denormalization of
one DB row, with no chunking and no embedding. Three operations: ``sync_citation_op`` (re)writes one citation, or
removes it when NADA no longer has it (NADA's change queue calls this for both an edit and a delete of a citation),
``delete_citation_op`` removes one without asking NADA, and ``backfill_citations_op`` walks the whole catalog.
``POST /admin/citations/sync`` runs the first and the last as background jobs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import batched
from typing import Any

import ai4data.discovery.catalog.extract as catalog_extract
import httpx
from opensearchpy.helpers import bulk

from nada_ai.ingest.extract_access import ExtractError, request_kwargs
from nada_ai.ingest.progress import CancelToken
from nada_ai.nada.admin_auth import scrub_admin_credentials
from nada_ai.search.backend.opensearch.citations import citation_bulk_action
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.mapping import citations_index_body, new_index_generation
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def ensure_citations_index(client: Any, settings: Settings) -> None:
    """Create the citation index if missing (stamped with a generation), else leave it as is."""
    name = settings.citations_index
    if client.indices.exists(index=name):
        return
    body = citations_index_body()
    body["mappings"]["_meta"] = {"generation": new_index_generation()}
    client.indices.create(index=name, body=body)


def _write_batch(client: Any, index: str, batch: list[dict[str, Any]]) -> tuple[int, list[Any]]:
    """Bulk-write one page of citation documents; ``(indexed, errors)``. Refreshes once at the end of a run."""
    actions = [
        citation_bulk_action(index, citation) for citation in batch if isinstance(citation.get("core_fields"), dict)
    ]
    if not actions:
        return 0, []
    success, errors = bulk(client, actions, raise_on_error=False, refresh=False)
    return int(success), list(errors) if isinstance(errors, list) else []


def delete_citation_op(settings: Settings, citation_id: int) -> dict[str, Any]:
    """Remove one citation from the index (a missing document is not an error)."""
    client = build_client(settings)
    try:
        resp = client.delete_by_query(
            index=settings.citations_index,
            body={"query": {"ids": {"values": [str(int(citation_id))]}}},
            refresh=True,
            ignore_unavailable=True,
        )
        return {"citation_id": int(citation_id), "deleted": int(resp.get("deleted") or 0)}
    finally:
        client.close()


def sync_citation_op(settings: Settings, citation_id: int) -> dict[str, Any]:
    """Make the index match NADA for one citation: write it, or remove it when NADA no longer has it.

    Returns ``{"citation_id", "indexed", "deleted", "errors"}``. Any other failure to read it from NADA raises, so the
    existing document is never dropped just because NADA was unreachable.
    """
    try:
        citation = catalog_extract.fetch_extract_citation(int(citation_id), **request_kwargs(settings))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            removed = delete_citation_op(settings, citation_id)
            return {"citation_id": int(citation_id), "indexed": 0, "deleted": removed["deleted"], "errors": []}
        raise ExtractError(scrub_admin_credentials(str(e))) from e
    except Exception as e:
        raise ExtractError(scrub_admin_credentials(str(e))) from e

    client = build_client(settings)
    try:
        ensure_citations_index(client, settings)
        written, errors = _write_batch(client, settings.citations_index, [citation])
        client.indices.refresh(index=settings.citations_index)
        return {"citation_id": int(citation_id), "indexed": written, "deleted": 0, "errors": errors}
    finally:
        client.close()


def backfill_citations_op(
    settings: Settings,
    *,
    batch_size: int = 500,
    max_records: int | None = None,
    recreate_index: bool = False,
    show_progress_bar: bool = True,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Page through every citation in the catalog and (re)index it. Idempotent: ``_id`` is the citation id.

    Adds and replaces citations; it does not remove citations that no longer exist in NADA — a delete reaches the index
    through NADA's change queue (``sync_citation_op``). ``progress_cb`` receives the job-registry progress shape after
    every page. Returns ``{"seen", "indexed", "errors", "total", "cancelled"}``.
    """
    client = build_client(settings)
    try:
        if recreate_index and client.indices.exists(index=settings.citations_index):
            client.indices.delete(index=settings.citations_index)
        ensure_citations_index(client, settings)

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

            pbar = tqdm(unit="citation", desc="Index citations")
        try:
            citations = catalog_extract.iter_extract_citations(
                page_size=batch_size, max_items=max_records, on_page=on_page, **request_kwargs(settings)
            )
            for batch in batched(citations, batch_size):
                if cancel_token is not None and cancel_token.is_set():
                    cancelled = True
                    break
                written, batch_errors = _write_batch(client, settings.citations_index, list(batch))
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

        client.indices.refresh(index=settings.citations_index)
        return {"seen": seen, "indexed": indexed, "errors": errors, "total": total, "cancelled": cancelled}
    finally:
        client.close()
