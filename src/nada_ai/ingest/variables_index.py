"""Indexing for the variable search (lexical only — see ``docs/variables-search-contract.md``).

Deliberately a standalone module, not part of the chunk/embedding ingestion pipeline (``pipeline.py``,
``opensearch_writer.py``): a variable document is a flat denormalization of one DB row, with no chunking and no
embedding, so none of that machinery applies. It is also not wired into the live catalog-change queue
(``search_index_sync.py``) yet — that queue only carries ``object_type=survey`` today, so keeping a study's
variables fresh currently means running ``sync_survey_variables_op`` yourself (e.g. from the same place that
already reindexes the study) or scheduling ``backfill_variables_op``.
"""

from __future__ import annotations

import logging
from typing import Any

import ai4data.discovery.catalog.extract as catalog_extract
from opensearchpy.helpers import bulk

from nada_ai.nada.admin_auth import resolve_admin_cookies, resolve_admin_headers, scrub_admin_credentials
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.mapping import new_index_generation, variables_index_body
from nada_ai.search.backend.opensearch.variables import variable_bulk_action
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

_USER_AGENT = "nada-ai-variables-index-cli/1.0"


class VariablesExtractError(RuntimeError):
    """Raised when NADA's search-metadata-extract variables endpoint returns an error payload."""


def _base_url(settings: Settings) -> str:
    if settings.metadata_extract_base_url:
        return settings.metadata_extract_base_url.rstrip("/")
    if url := catalog_extract.extract_base_url():
        return url
    raise VariablesExtractError(
        "No metadata-extract base URL configured. Set NADA_METADATA_EXTRACT_BASE_URL "
        "(or AI4DATA_METADATA_CATALOG_EXTRACT_PATH) to the metadata-extract API for your NADA instance."
    )


def _request_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "base_url": _base_url(settings),
        "headers": resolve_admin_headers(user_agent=_USER_AGENT),
        "cookies": resolve_admin_cookies(),
    }


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


def sync_survey_variables_op(settings: Settings, idno: str) -> dict[str, Any]:
    """Delete then re-index every variable of one study (by its NADA idno). Best fit after that study's own
    document is (re)indexed, so a search never shows a variable of an unpublished or deleted study."""
    client = build_client(settings)
    try:
        ensure_variables_index(client, settings)

        client.delete_by_query(
            index=settings.variables_index,
            body={"query": {"term": {"idno": idno}}},
            refresh=True,
        )

        try:
            data = catalog_extract.fetch_extract_survey_variables(idno, **_request_kwargs(settings))
        except Exception as e:
            raise VariablesExtractError(scrub_admin_credentials(str(e))) from e

        variables = data.get("variables")
        actions = _bulk_actions(settings.variables_index, variables if isinstance(variables, list) else [])
        if not actions:
            return {"idno": idno, "indexed": 0, "errors": []}

        success, errors = bulk(client, actions, raise_on_error=False, refresh="wait_for")
        return {"idno": idno, "indexed": success, "errors": list(errors) if isinstance(errors, list) else []}
    finally:
        client.close()


def backfill_variables_op(
    settings: Settings,
    *,
    batch_size: int = 200,
    max_records: int | None = None,
    recreate_index: bool = False,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Page through every variable in the catalog and (re)index it. Run once to populate the index, or again after
    a catalog-wide change: each document's ``_id`` is its ``uid``, so re-running is idempotent."""
    client = build_client(settings)
    try:
        if recreate_index and client.indices.exists(index=settings.variables_index):
            client.indices.delete(index=settings.variables_index)
        ensure_variables_index(client, settings)

        indexed = 0
        errors: list[Any] = []
        seen = 0
        pbar: Any = None
        if show_progress_bar:
            from tqdm.auto import tqdm

            pbar = tqdm(unit="variable", desc="Index variables")
        try:
            request_kwargs = _request_kwargs(settings)
            batch: list[dict[str, Any]] = []
            for variable in catalog_extract.iter_extract_variables(
                page_size=batch_size, max_items=max_records, **request_kwargs
            ):
                batch.append(variable)
                seen += 1
                if len(batch) >= batch_size:
                    actions = _bulk_actions(settings.variables_index, batch)
                    if actions:
                        success, batch_errors = bulk(client, actions, raise_on_error=False, refresh=False)
                        indexed += success
                        if isinstance(batch_errors, list):
                            errors.extend(batch_errors)
                    if pbar is not None:
                        pbar.update(len(batch))
                    batch = []
            if batch:
                actions = _bulk_actions(settings.variables_index, batch)
                if actions:
                    success, batch_errors = bulk(client, actions, raise_on_error=False, refresh=False)
                    indexed += success
                    if isinstance(batch_errors, list):
                        errors.extend(batch_errors)
                if pbar is not None:
                    pbar.update(len(batch))
        except Exception as e:
            raise VariablesExtractError(scrub_admin_credentials(str(e))) from e
        finally:
            if pbar is not None:
                pbar.close()

        client.indices.refresh(index=settings.variables_index)
        return {"seen": seen, "indexed": indexed, "errors": errors}
    finally:
        client.close()
