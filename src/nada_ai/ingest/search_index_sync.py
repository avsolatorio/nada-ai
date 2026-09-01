"""Reconcile the search index against NADA's ``search-index`` change queue.

NADA (as of the ``catalog-admin`` OpenAPI spec, tag ``Search index``) tracks a
change queue for keeping one external search index in sync with the catalog:
every study create/update/delete enqueues a row. This module polls that queue,
applies each change to our own index/collection, and acks it back — so a
missed webhook, a restart, or a cold start can all catch up by polling instead
of needing a full re-ingest.

Endpoints used (admin API key required, ``X-API-KEY``):

- ``GET  {base}/admin/search-index/status``          — queue/state counts
- ``GET  {base}/admin/search-index/queue``            — list pending/failed items
- ``POST {base}/admin/search-index/queue/{id}/ack``   — ack one item
- ``POST {base}/admin/search-index/requeue``          — retry one item, or reset all failed

Queue item shape (``SearchIndexQueueItem`` in the spec)::

    {
      "id": 123,                    # queue row id — used for ack
      "object_type": "survey",      # or "citation" (not handled here)
      "object_id": 456,             # NADA's internal numeric id
      "object_key": "WLD_2021_...", # the study idno (what we index by)
      "change_class": "upsert_full" | "upsert_partial" | "variables" | "delete",
      "status": "pending" | "failed",
      "attempts": 0,
      "last_error": null,
      "changed": 1732000000,        # unix time — MUST be echoed back on ack
      "fetch_document": true        # false for deletes (tombstone, no document)
    }

The queue item does **not** carry NADA's ``dataset_type`` (survey, geospatial,
timeseries, document, table, image, script, video, timeseriesdb) — only
``object_key``/idno. Since ``nada_ai.ingest`` needs an explicit
``metadata_type`` (indicator, document, geospatial, microdata — see
``ai4data.discovery.metadata.handler.MetadataLoader``) to know which loader to
use, this module looks up each new/changed idno's ``dataset_type`` via the
``search-metadata-extract`` API and maps it. Items whose ``dataset_type`` has
no known mapping (table, image, script, video) are acked as ``failed`` with a
clear reason rather than guessed at or silently dropped — that keeps them
visible in NADA's ``/admin/search-index/queue?status=failed`` for a human to
resolve, instead of retrying forever or corrupting the index with a wrong type.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import ai4data.discovery.catalog.extract as catalog_extract
import httpx
from ai4data.discovery.config import metadata_catalog
from pydantic import BaseModel

from nada_ai.ingest.service import delete_by_idno_op, index_ids_op
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0

#: NADA dataset_type -> nada_ai metadata_type. Anything not listed here has no
#: ingest path yet and is acked as failed rather than guessed at.
_DATASET_TYPE_TO_METADATA_TYPE: dict[str, str] = {
    "timeseries": "indicator",
    "timeseriesdb": "indicator",
    "document": "document",
    "geospatial": "geospatial",
    "survey": "microdata",
}


class SearchIndexSyncError(RuntimeError):
    """Raised for non-recoverable search-index API errors."""


class QueueItemChanged(SearchIndexSyncError):
    """Raised on 409 QUEUE_CHANGED: the row was coalesced between poll and ack.

    The row is still pending in NADA and will surface again on the next poll
    (with a fresh ``changed`` value) — this is not a failure, just a signal to
    move on rather than retry the ack with the stale timestamp.
    """


class SearchIndexQueueItem(BaseModel):
    id: int
    object_type: Literal["survey", "citation"]
    object_id: int
    object_key: str
    change_class: Literal["upsert_full", "upsert_partial", "variables", "delete"]
    status: Literal["pending", "failed"]
    attempts: int = 0
    last_error: str | None = None
    changed: int
    fetch_document: bool = True

    @property
    def is_delete(self) -> bool:
        return self.change_class == "delete" or not self.fetch_document


class SearchIndexStatus(BaseModel):
    status: str
    search_provider: str | None = None
    tracking_enabled: bool = False
    queue: dict[str, int] = {}
    state: dict[str, int] = {}


def _base_url(settings: Settings) -> str:
    if settings.search_index_base_url:
        return settings.search_index_base_url.rstrip("/")
    if not metadata_catalog.url:
        raise SearchIndexSyncError(
            "No search-index base URL configured. Set NADA_SEARCH_INDEX_BASE_URL "
            "(or AI4DATA_METADATA_CATALOG_URL) to your NADA instance."
        )
    return f"{metadata_catalog.url.rstrip('/')}/api"


def _headers(settings: Settings) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "nada-ai-search-index-sync/1.0"}
    api_key = settings.metadata_extract_api_key or metadata_catalog.x_api_key
    if api_key:
        headers["X-API-KEY"] = api_key
    bearer = settings.metadata_extract_auth_bearer or metadata_catalog.auth_bearer
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


def _cookies(settings: Settings) -> dict[str, str]:
    raw = settings.metadata_extract_auth_cookie or metadata_catalog.cookies
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        if name := name.strip():
            out[name] = value.strip()
    return out


def _client(settings: Settings) -> httpx.Client:
    return httpx.Client(
        base_url=_base_url(settings),
        headers=_headers(settings),
        cookies=_cookies(settings),
        timeout=_TIMEOUT,
    )


def get_status(settings: Settings) -> SearchIndexStatus:
    with _client(settings) as client:
        resp = client.get("/admin/search-index/status")
        resp.raise_for_status()
    return SearchIndexStatus.model_validate(resp.json())


def list_queue(
    settings: Settings,
    *,
    status: Literal["pending", "failed"] = "pending",
    object_type: Literal["survey", "citation"] | None = None,
    limit: int = 50,
) -> list[SearchIndexQueueItem]:
    params: dict[str, Any] = {"status": status, "limit": limit}
    if object_type:
        params["object_type"] = object_type
    with _client(settings) as client:
        resp = client.get("/admin/search-index/queue", params=params)
        resp.raise_for_status()
    data = resp.json()
    return [SearchIndexQueueItem.model_validate(item) for item in data.get("items", [])]


def ack_item(
    settings: Settings,
    item_id: int,
    *,
    result: Literal["indexed", "failed"],
    changed: int,
    error: str | None = None,
) -> dict[str, Any]:
    """Ack one queue item. Raises :class:`QueueItemChanged` on 409."""
    body: dict[str, Any] = {"result": result, "changed": changed}
    if error:
        body["error"] = error[:2000]
    with _client(settings) as client:
        resp = client.post(f"/admin/search-index/queue/{item_id}/ack", json=body)
    if resp.status_code == 409:
        raise QueueItemChanged(f"Queue item {item_id} changed between poll and ack")
    resp.raise_for_status()
    return resp.json()


def requeue_object(settings: Settings, object_type: Literal["survey", "citation"], object_id: int) -> dict[str, Any]:
    with _client(settings) as client:
        resp = client.post(
            "/admin/search-index/requeue",
            json={"object_type": object_type, "object_id": object_id},
        )
        resp.raise_for_status()
    return resp.json()


def requeue_failed(settings: Settings) -> dict[str, Any]:
    with _client(settings) as client:
        resp = client.post("/admin/search-index/requeue", json={"status": "failed"})
        resp.raise_for_status()
    return resp.json()


def _lookup_metadata_type(settings: Settings, idno: str) -> str | None:
    """Resolve NADA's dataset_type for idno and map it to a nada_ai metadata_type.

    Confirmed against a live instance: the study document has no top-level
    ``dataset_type`` — despite what the catalog-admin OpenAPI spec documents,
    the real field lives at ``study["filters"]["dataset_type"]`` (``filters``
    is NADA's own per-study computed facet dict — see ``ai4data``'s
    ``study_metadata_type()``, which reads it from the same place).
    """
    extract_base = settings.metadata_extract_base_url or catalog_extract.extract_base_url()
    kwargs: dict[str, Any] = {"headers": _headers(settings), "cookies": _cookies(settings)}
    if extract_base:
        kwargs["base_url"] = extract_base
    data = catalog_extract.fetch_extract_study(idno, **kwargs)
    study = data.get("study") if isinstance(data.get("study"), dict) else data
    filters = (study or {}).get("filters")
    dataset_type = filters.get("dataset_type") if isinstance(filters, dict) else None
    return _DATASET_TYPE_TO_METADATA_TYPE.get(dataset_type) if dataset_type else None


def reconcile_once(
    settings: Settings,
    *,
    limit: int = 50,
    embedding: Any | None = None,
) -> dict[str, Any]:
    """Poll one page of the pending queue and apply + ack each item.

    Only ``object_type=survey`` is handled — citations aren't part of the
    catalog metadata this index covers. Returns a summary dict; call again
    (e.g. on a schedule) to keep draining the queue, since one call only
    processes up to ``limit`` items.

    The upsert branch's ``index_ids_op`` call also syncs this idno's
    filters/facets — not via an extra call here, but because
    ``ingest.pipeline.iter_langdoc_records`` (which both backend writers
    route through) fetches and bakes in NADA's ``filters`` data as part of
    building the document/point payload itself (see
    ``settings.sync_filters_during_ingest``). So a queue-driven reindex keeps
    both content and facets in sync from a single fetch, with no separate
    filters-sync pass required.
    """
    items = list_queue(settings, status="pending", object_type="survey", limit=limit)
    summary = {"polled": len(items), "indexed": 0, "deleted": 0, "failed": 0, "ack_conflicts": 0}

    for item in items:
        idno = item.object_key
        result: Literal["indexed", "failed"]
        error: str | None
        try:
            if item.is_delete:
                delete_by_idno_op(settings, idno)
            else:
                metadata_type = _lookup_metadata_type(settings, idno)
                if metadata_type is None:
                    raise SearchIndexSyncError(
                        f"No metadata_type mapping for idno {idno!r} — unsupported or unknown dataset_type"
                    )
                index_ids_op(
                    settings,
                    idnos=[idno],
                    metadata_type=metadata_type,
                    force=True,
                    show_progress_bar=False,
                    embedding=embedding,
                )
        except Exception as e:  # noqa: BLE001 - reported back to NADA, not swallowed
            logger.warning("search-index reconcile failed for idno=%s: %s", idno, e)
            result, error = "failed", str(e)
            summary["failed"] += 1
        else:
            result, error = "indexed", None
            summary["deleted" if item.is_delete else "indexed"] += 1

        try:
            ack_item(settings, item.id, result=result, changed=item.changed, error=error)
        except QueueItemChanged:
            summary["ack_conflicts"] += 1

    return summary
