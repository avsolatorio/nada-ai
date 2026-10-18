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
``metadata_type`` (indicator, indicator-db, document, geospatial, microdata,
table, script, image, video — see
``ai4data.discovery.metadata.handler.MetadataLoader``) to know which loader to
use, this module looks up each new/changed idno's ``dataset_type`` via the
``search-metadata-extract`` API and maps it. Items whose ``dataset_type`` has
no known mapping are acked as ``failed`` with a clear reason rather than
guessed at or silently dropped — that keeps them visible in NADA's
``/admin/search-index/queue?status=failed`` for a human to resolve, instead of
retrying forever or corrupting the index with a wrong type.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

import ai4data.discovery.catalog.extract as catalog_extract
import httpx
from ai4data.discovery.config import metadata_catalog
from pydantic import BaseModel

from nada_ai.ingest.progress import CancelToken
from nada_ai.ingest.service import (
    _attribute_write_error,
    _error_by_idno,
    delete_by_idno_op,
    delete_by_idnos_op,
    index_ids_op,
)
from nada_ai.nada import gate as nada_gate
from nada_ai.nada.admin_auth import resolve_admin_cookies, resolve_admin_headers
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
_USER_AGENT = "nada-ai-search-index-sync/1.0"

#: NADA dataset_type -> nada_ai metadata_type. Anything not listed here has no
#: ingest path yet and is acked as failed rather than guessed at.
_DATASET_TYPE_TO_METADATA_TYPE: dict[str, str] = {
    "timeseries": "indicator",
    "timeseriesdb": "indicator-db",
    "timeseries-db": "indicator-db",
    "document": "document",
    "geospatial": "geospatial",
    "survey": "microdata",
    "table": "table",
    "script": "script",
    "image": "image",
    "video": "video",
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
    #: 0 for a catalog change, 1 for a row a sync queued (NADA lists 0 first)
    priority: int = 0
    #: the study's surveys.type (None for citations)
    data_type: str | None = None
    fetch_document: bool = True

    @property
    def is_delete(self) -> bool:
        return self.change_class == "delete" or not self.fetch_document


class SearchIndexStatus(BaseModel):
    status: str
    search_engine: str | None = None
    #: the engine whose index NADA keeps in sync (search_index_engine)
    index_engine: str | None = None
    tracking_enabled: bool = False
    #: what NADA's tracking rows describe: engine, index_generation, tracked_since, last_sync_at, last_poll_at
    tracked: dict[str, Any] | None = None
    queue: dict[str, Any] = {}
    state: dict[str, int] = {}


class QueuePage(BaseModel):
    items: list[SearchIndexQueueItem]
    total: int = 0
    tracking_enabled: bool = False
    #: why NADA refused the page when tracking_enabled is false: not_tracking_for_nada_ai | other_nada_ai |
    #: tracking_disabled
    reason: str | None = None


class DiffItem(BaseModel):
    idno: str
    type: str | None = None  # only present on diff_missing; NADA's surveys.type for this idno
    last_error: str | None = None  # the reported error, if this idno's last attempt failed


class ObjectTypeSummary(BaseModel):
    """How many catalog entries of object_type exist vs. their
    search_index_state breakdown — scoped to one object_type, unlike
    :class:`SearchIndexStatus` above (a global total across every type)."""

    object_type: str
    catalog_total: int
    state: dict[str, int] = {}


class DiffPage(BaseModel):
    items: list[DiffItem]
    total: int


class TypeBreakdownItem(BaseModel):
    """Catalog-vs-index coverage for one surveys.type value (microdata,
    geospatial, document, timeseries, ...) — what ObjectTypeSummary can't show
    since it lumps every type under NADA's single 'survey' object_type."""

    data_type: str
    catalog_total: int
    indexed: int
    missing: int
    stale: int
    errors: int = 0  # rows with a recorded last_error — genuinely attempted and failed


def _base_url(settings: Settings) -> str:
    if settings.search_index_base_url:
        return settings.search_index_base_url.rstrip("/")
    if not metadata_catalog.url:
        raise SearchIndexSyncError(
            "No search-index base URL configured. Set NADA_SEARCH_INDEX_BASE_URL "
            "(or AI4DATA_METADATA_CATALOG_URL) to your NADA instance."
        )
    return f"{metadata_catalog.url.rstrip('/')}/api"


#: The generation of this deployment's study index, sent to NADA with every search-index request (``X-Index-Generation``)
#: so NADA can refuse a nada-ai whose index is not the one it tracks. Read from the index's mapping, kept for a minute.
_generation_cache: dict[str, Any] = {"value": None, "at": float("-inf")}
_generation_lock = threading.Lock()
_GENERATION_TTL = 60.0


def index_generation(settings: Settings) -> str | None:
    """The study index's generation (``mappings._meta.generation``), or None when there is no study index."""
    if settings.search_backend != "opensearch":
        return None
    with _generation_lock:
        if time.monotonic() - _generation_cache["at"] < _GENERATION_TTL:
            return _generation_cache["value"]
        from nada_ai.search.backend.opensearch.client import build_client

        client = build_client(settings)
        try:
            mapping = client.indices.get_mapping(index=settings.studies_index, ignore_unavailable=True)
            meta = (next(iter(mapping.values()))["mappings"].get("_meta") or {}) if mapping else {}
            value = meta.get("generation")
        except Exception as e:  # noqa: BLE001 - the header is optional for NADA; OpenSearch being down must not stop NADA calls
            logger.warning("could not read the study index generation: %s", e)
            value = None
        finally:
            client.close()
        _generation_cache.update(value=value, at=time.monotonic())
        return value


def forget_index_generation() -> None:
    """Read the generation again on the next request (the index was just created or recreated)."""
    with _generation_lock:
        _generation_cache.update(value=None, at=float("-inf"))


def gated_transport(settings: Settings) -> nada_gate.GatedTransport:
    """The process's transport to NADA, through the request gate (installed on first use)."""
    return nada_gate.install(settings, _base_url(settings))


def _client(settings: Settings) -> httpx.Client:
    headers = resolve_admin_headers(user_agent=_USER_AGENT)
    generation = index_generation(settings)
    if generation:
        headers["X-Index-Generation"] = generation
    return httpx.Client(
        base_url=_base_url(settings),
        headers=headers,
        cookies=resolve_admin_cookies(),
        timeout=settings.nada_request_timeout_seconds,
        transport=gated_transport(settings),
    )


def get_status(settings: Settings) -> SearchIndexStatus:
    with _client(settings) as client:
        resp = client.get("/admin/search-index/status")
        resp.raise_for_status()
    return SearchIndexStatus.model_validate(resp.json())


def get_object_type_summary(settings: Settings, object_type: str) -> ObjectTypeSummary:
    """How many catalog entries of object_type exist vs. their
    search_index_state breakdown — this is the "records in DB vs in index"
    number a dashboard summary needs, scoped to one object_type (``get_status``
    above is a global total blended across every tracked type)."""
    with _client(settings) as client:
        resp = client.get("/admin/search-index/summary", params={"object_type": object_type})
        resp.raise_for_status()
    return ObjectTypeSummary.model_validate(resp.json())


def list_type_breakdown(settings: Settings, object_type: str = "survey") -> list[TypeBreakdownItem]:
    """Per-surveys.type catalog-vs-index coverage — see :class:`TypeBreakdownItem`."""
    with _client(settings) as client:
        resp = client.get("/admin/search-index/type-breakdown", params={"object_type": object_type})
        resp.raise_for_status()
    data = resp.json()
    return [TypeBreakdownItem.model_validate(i) for i in data.get("items", [])]


def list_queue(
    settings: Settings,
    *,
    status: Literal["pending", "failed"] = "pending",
    object_type: Literal["survey", "citation"] | None = None,
    limit: int = 50,
    exclude: list[int] | None = None,
) -> QueuePage:
    """One page of queue rows, catalog changes first. ``exclude``: rows held in this pass (not acked), skipped over."""
    params: dict[str, Any] = {"status": status, "limit": limit}
    if object_type:
        params["object_type"] = object_type
    if exclude:
        params["exclude"] = ",".join(str(i) for i in exclude)
    with _client(settings) as client:
        resp = client.get("/admin/search-index/queue", params=params)
        resp.raise_for_status()
    data = resp.json()
    return QueuePage(
        items=[SearchIndexQueueItem.model_validate(item) for item in data.get("items", [])],
        total=int(data.get("total") or 0),
        tracking_enabled=bool(data.get("tracking_enabled")),
        reason=data.get("reason"),
    )


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


class AckRefused(SearchIndexSyncError):
    """NADA answered an ack request with an error for all of it (not per item): ``code`` is NADA's message."""

    def __init__(self, code: str, status: int) -> None:
        super().__init__(f"NADA refused the acks: {code} (HTTP {status})")
        self.code = code
        self.status = status


def ack_batch(settings: Settings, acks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ack up to 100 rows in one request: ``[{id, result, changed, error?}]`` -> one result per row, in order
    (``{id, applied, result?|error?}``; ``QUEUE_CHANGED``/``QUEUE_NOT_FOUND`` are per-row results, not failures).

    Raises :class:`AckRefused` when NADA refuses the request as a whole (not tracking for this nada-ai, another
    index generation, access denied), and the gate's errors when NADA cannot be reached."""
    body = {"items": [{**a, "error": (a.get("error") or "")[:2000] or None} for a in acks]}
    with _client(settings) as client:
        resp = client.post("/admin/search-index/queue/ack", json=body)
    if resp.status_code >= 400:
        try:
            code = str(resp.json().get("message") or resp.status_code)
        except ValueError:
            code = str(resp.status_code)
        raise AckRefused(code, resp.status_code)
    return list(resp.json().get("results") or [])


def fetch_studies_by_sids(
    settings: Settings, sids: list[int], *, include_metadata: bool = False, include_resources: bool = False
) -> dict[str, Any]:
    """Extract records of these studies in one request: ``{studies, missing, errors}`` (see NADA's
    ``GET search-metadata-extract/studies?sids=``). ``include_metadata=False`` sends core fields and filters only;
    ``include_metadata`` with ``include_resources`` is the full record indexing reads."""
    if not sids:
        return {"studies": [], "missing": [], "errors": []}
    extract_base = (settings.metadata_extract_base_url or catalog_extract.extract_base_url() or "").rstrip("/")
    if not extract_base:
        raise SearchIndexSyncError("No metadata-extract base URL configured")
    with _client(settings) as client:
        resp = client.get(
            f"{extract_base}/studies",
            params={
                "sids": ",".join(str(i) for i in sids),
                "include_metadata": "1" if include_metadata else "0",
                "include_admin_metadata": "1" if include_metadata else "0",
                "include_resources": "1" if include_resources else "0",
            },
        )
        resp.raise_for_status()
    data = resp.json()
    return {
        "studies": list(data.get("studies") or []),
        "missing": [int(i) for i in data.get("missing") or []],
        "errors": list(data.get("errors") or []),
    }


def trigger_sync(settings: Settings, *, object_type: str = "all", refresh_indexed: bool = False) -> dict[str, Any]:
    """Ask NADA to queue what this index needs to match the catalog (``POST /admin/search-index/sync``)."""
    params = {"object_type": object_type, "refresh_indexed": "1" if refresh_indexed else "0"}
    with _client(settings) as client:
        resp = client.post("/admin/search-index/sync", params=params)
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


_STATE_BULK_CHUNK_SIZE = 500

#: NADA stores indexed study dataset types under the ``survey`` object type; citations use a separate object type.
STATE_OBJECT_TYPE_SURVEY = "survey"


def report_state_bulk(settings: Settings, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Best-effort: report indexing/deletion outcomes to NADA's search_index_state.

    ``items``: ``[{"object_type": "survey", "object_key": idno, "status": "indexed"|"failed"|"deleted", "error": str|None}, ...]``.
    ``error`` is optional and only meaningful (and only stored by NADA) when
    ``status == "failed"`` — the message from whatever raised, for a human
    checking the diff dashboard later without digging through logs.
    Chunked automatically (NADA's endpoint has no documented size limit, but a
    single admin-triggered full-catalog run can produce tens of thousands of
    these — sending them all in one request body isn't a good idea regardless).

    Callers should catch and log failures rather than let them fail the
    ingest job — by the time this is called, the actual index/delete already
    happened; this only affects NADA's bookkeeping of that fact, not the
    indexed content itself.
    """
    if not items:
        return {"applied": 0, "results": []}
    applied = 0
    results: list[dict[str, Any]] = []
    with _client(settings) as client:
        for start in range(0, len(items), _STATE_BULK_CHUNK_SIZE):
            chunk = items[start : start + _STATE_BULK_CHUNK_SIZE]
            resp = client.post("/admin/search-index/state/bulk", json={"items": chunk})
            resp.raise_for_status()
            data = resp.json()
            applied += int(data.get("applied") or 0)
            results.extend(data.get("results") or [])
    return {"applied": applied, "results": results}


def list_diff_missing(
    settings: Settings,
    *,
    object_type: str = STATE_OBJECT_TYPE_SURVEY,
    limit: int = 100,
    offset: int = 0,
    data_type: str | None = None,
    has_error: bool = False,
) -> DiffPage:
    """NADA catalog entries with no current 'indexed' search_index_state row
    (published or not — NADA's diff doesn't filter on that, see its docblock).

    ``total`` is a separate count, independent of ``limit``/``offset`` — an
    empty ``items`` page never means "nothing missing", check ``total``.
    ``data_type``, when given, narrows to one surveys.type value (microdata
    aka 'survey', geospatial, document, timeseries, ...) — for reconciling
    one data type at a time instead of the whole object_type at once.
    ``has_error``, when true, narrows to rows with a recorded last_error —
    genuinely attempted and failed, as opposed to simply never attempted yet.
    """
    params: dict[str, Any] = {"object_type": object_type, "limit": limit, "offset": offset}
    if data_type:
        params["data_type"] = data_type
    if has_error:
        params["has_error"] = "1"
    with _client(settings) as client:
        resp = client.get("/admin/search-index/diff/missing", params=params)
        resp.raise_for_status()
    return DiffPage.model_validate(resp.json())


def list_diff_stale(
    settings: Settings,
    *,
    object_type: str = STATE_OBJECT_TYPE_SURVEY,
    limit: int = 100,
    offset: int = 0,
    data_type: str | None = None,
) -> DiffPage:
    """search_index_state rows marked 'indexed' whose catalog entry is gone
    entirely (published or not — see list_diff_missing's docstring).

    ``data_type`` narrows to one surveys.type value, same as list_diff_missing.
    """
    params: dict[str, Any] = {"object_type": object_type, "limit": limit, "offset": offset}
    if data_type:
        params["data_type"] = data_type
    with _client(settings) as client:
        resp = client.get("/admin/search-index/diff/stale", params=params)
        resp.raise_for_status()
    return DiffPage.model_validate(resp.json())


def lookup_metadata_type(settings: Settings, idno: str) -> str | None:
    """Resolve NADA's dataset_type for idno and map it to a nada_ai metadata_type.

    The dataset type is read from ``study["filters"]["dataset_type"]`` in the metadata extract.
    """
    extract_base = settings.metadata_extract_base_url or catalog_extract.extract_base_url()
    kwargs: dict[str, Any] = {
        "headers": resolve_admin_headers(user_agent=_USER_AGENT),
        "cookies": resolve_admin_cookies(),
    }
    if extract_base:
        kwargs["base_url"] = extract_base
    data = catalog_extract.fetch_extract_study(idno, **kwargs)
    study = data.get("study") if isinstance(data.get("study"), dict) else data
    filters = (study or {}).get("filters")
    dataset_type = filters.get("dataset_type") if isinstance(filters, dict) else None
    return _DATASET_TYPE_TO_METADATA_TYPE.get(dataset_type) if dataset_type else None


def _apply_citation(settings: Settings, item: SearchIndexQueueItem) -> Literal["indexed", "deleted"]:
    """Apply one citation queue item. Citations are indexed on OpenSearch only (lexical, ``docs/citations-search-contract.md``);
    Qdrant has no citation index, so there is nothing to do — the item is still acked as handled rather than left
    pending forever."""
    from nada_ai.ingest.citations_index import delete_citation_op, sync_citation_op

    if settings.search_backend != "opensearch":
        return "indexed"
    if item.is_delete:
        delete_citation_op(settings, item.object_id)
        return "deleted"
    _raise_on_write_errors(f"citation {item.object_id}", sync_citation_op(settings, item.object_id)["errors"])
    return "indexed"


def _raise_on_write_errors(what: str, errors: list[Any]) -> None:
    """The sync ops return backend write errors instead of raising; raise so the queue item is acked failed, not
    indexed."""
    if errors:
        raise SearchIndexSyncError(f"{what}: {len(errors)} write error(s), first: {errors[0]}")


def _raise_if_not_indexed(idno: str, result: dict[str, Any]) -> None:
    """``index_ids_op`` never raises for a per-idno failure (metadata failed to load/parse, no documents produced,
    a backend write error) — it reports them in its return value. Raise so the queue item is acked failed: acking it
    indexed would also overwrite the failed ``search_index_state`` that ``index_ids_op`` just reported, and the study
    would drop out of NADA's "missing" diff, so neither the queue nor ``reconcile_diff_once`` would retry it."""
    reasons = list(_error_by_idno(result["load_errors"], result["empty_docs"]).values())
    reasons += [_attribute_write_error(e)[1] for e in result["errors"]]
    # A full index of a study includes its variables; their sync never raises, so read what it reported.
    reasons += [f"variables: {reason}" for reason in ((result.get("variables") or {}).get("failed") or {}).values()]
    if reasons:
        raise SearchIndexSyncError(f"index reported no success for idno {idno!r}: {'; '.join(reasons)}")


def metadata_type_for(settings: Settings, item: SearchIndexQueueItem) -> str | None:
    """The nada_ai metadata_type of a study item: from the ``data_type`` NADA sends with it, looked up when absent."""
    if item.data_type:
        return _DATASET_TYPE_TO_METADATA_TYPE.get(item.data_type)
    return lookup_metadata_type(settings, item.object_key)


def full_record(settings: Settings, sid: int) -> dict[str, Any]:
    """One study's full extract record (metadata and resources), fetched by its id."""
    batch = fetch_studies_by_sids(settings, [sid], include_metadata=True, include_resources=True)
    if batch["errors"]:
        raise SearchIndexSyncError(f"NADA could not build the extract of study {sid}: {batch['errors'][0].get('message')}")
    if not batch["studies"]:
        raise SearchIndexSyncError(f"Study {sid} is not in NADA's catalog")
    return batch["studies"][0]


def _index_in_full(
    settings: Settings, item: SearchIndexQueueItem, embedding: Any | None, extract: dict[str, Any] | None = None
) -> str:
    """Index one study in full from its record fetched **by id**: the record is put where ai4data's loader reads it
    (its metadata cache), so the loader never fetches by idno — NADA cannot resolve every idno in a URL (``:``,
    ``/``, spaces). ``extract``: the record when the caller already fetched it with its metadata. Indexing does not
    force a re-embed: chunks whose text did not change keep their vectors."""
    metadata_type = metadata_type_for(settings, item)
    if metadata_type is None:
        raise SearchIndexSyncError(
            f"No metadata_type mapping for idno {item.object_key!r} (dataset_type {item.data_type!r})"
        )
    record = extract if extract is not None and isinstance(extract.get("metadata"), dict) else full_record(
        settings, item.object_id
    )
    catalog_extract.write_metadata_cache(
        catalog_extract.study_to_catalog_metadata(record), item.object_key, metadata_type, include_resources=True
    )
    report = index_ids_op(
        settings,
        idnos=[item.object_key],
        metadata_type=metadata_type,
        force=False,
        show_progress_bar=False,
        embedding=embedding,
    )
    _raise_if_not_indexed(item.object_key, report)
    return "indexed"


def process_item(
    settings: Settings,
    item: SearchIndexQueueItem,
    *,
    extract: dict[str, Any] | None = None,
    embedding: Any | None = None,
) -> str:
    """Apply one queue item to the index: ``indexed`` | ``deleted``. Does not ack it (the caller acks in batches).

    ``extract``: the study's extract record when the caller already fetched it (a page of partial updates is fetched
    in one request); fetched here otherwise. Raises for anything that went wrong — :class:`nada_gate.NadaTrouble`
    when NADA could not serve a request, anything else when the item itself could not be applied.
    """
    if item.object_type == "citation":
        return _apply_citation(settings, item)
    if item.is_delete:
        delete_by_idno_op(settings, item.object_key)
        return "deleted"
    if item.change_class == "variables":
        # Only this study's variables changed (a DDI/data-dictionary re-import): sync them, nothing else. Qdrant has
        # no variable index, so there is nothing to do there.
        if settings.search_backend == "opensearch":
            from nada_ai.ingest.variables_index import sync_survey_variables_op

            _raise_on_write_errors(
                f"variables of {item.object_key!r}",
                sync_survey_variables_op(settings, item.object_key, sid=item.object_id)["errors"],
            )
        return "indexed"
    if item.change_class == "upsert_partial" and settings.search_backend == "opensearch":
        # A change of the study's options (publish state, license, data class, links, DOI, featured, collections),
        # not of its metadata: applied in place, with no metadata read and no embedding. A study not in the index
        # yet is indexed in full.
        from nada_ai.ingest.service import apply_study_options_from_extract, apply_study_options_op

        if extract is not None:
            core = extract.get("core_fields") or {}
            result = apply_study_options_from_extract(
                settings, item.object_key, int(core.get("catalog_id") or item.object_id), core, extract.get("filters") or {}
            )
        else:
            result = apply_study_options_op(settings, item.object_key)
        if result.get("updated"):
            return "indexed"
    return _index_in_full(settings, item, embedding, extract)


def reconcile_once(
    settings: Settings,
    *,
    limit: int = 50,
    embedding: Any | None = None,
) -> dict[str, Any]:
    """Process one page of the pending queue and ack it (the ``reconcile_search_index`` CLI command).

    One pass, no pausing or retrying across items: the in-process scheduler (``app.reconcile_scheduler``) is what
    works through the queue for a running deployment. Stops at the first item NADA could not serve.
    """
    page = list_queue(settings, status="pending", limit=limit)
    summary: dict[str, Any] = {"polled": len(page.items), "indexed": 0, "deleted": 0, "failed": 0, "not_processed": 0}
    if not page.tracking_enabled:
        summary["reason"] = page.reason
        return summary

    acks: list[dict[str, Any]] = []
    for n, item in enumerate(page.items):
        try:
            action = process_item(settings, item, embedding=embedding)
        except nada_gate.NadaTrouble as e:
            logger.warning("search-index reconcile stopped: NADA could not serve a request: %s", e)
            summary["not_processed"] = len(page.items) - n
            break
        except Exception as e:  # noqa: BLE001 - reported back to NADA, not swallowed
            logger.warning("search-index reconcile failed for idno=%s: %s", item.object_key, e)
            acks.append({"id": item.id, "result": "failed", "changed": item.changed, "error": str(e)})
            summary["failed"] += 1
            continue
        acks.append({"id": item.id, "result": "indexed", "changed": item.changed})
        summary[action] += 1

    for start in range(0, len(acks), 100):
        ack_batch(settings, acks[start : start + 100])
    return summary


#: Absolute circuit breaker on reconcile_diff_once's re-fetch loop — real
#: termination is guaranteed by the offset only ever moving forward past items
#: that stay in the diff; this only guards against an unforeseen bug turning
#: that into an infinite loop.
_DIFF_RECONCILE_MAX_ITERATIONS = 10_000

DIFF_RECONCILE_NEEDS_STATE_REPORTING = (
    "reconcile-diff needs NADA_REPORT_SEARCH_INDEX_STATE_ENABLED=true: NADA's diff only changes when nada-ai "
    "reports what it indexed, so without it the run cannot make progress"
)


def reconcile_diff_once(
    settings: Settings,
    *,
    object_type: str = STATE_OBJECT_TYPE_SURVEY,
    page_size: int = 200,
    embedding: Any | None = None,
    data_type: str | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
    remove_only: bool = False,
) -> dict[str, Any]:
    """Pull NADA's missing/stale diff for object_type and resolve all of it:
    index every missing idno, delete every stale one.

    ``data_type``, when given, narrows the whole run to one surveys.type
    value (microdata aka 'survey', geospatial, document, timeseries, ...) —
    for reconciling one data type at a time from the per-type breakdown
    instead of the whole object_type in one go.

    ``remove_only`` (the dashboard's "Clean up") deletes the stale entries and
    leaves the missing ones alone: the missing diff is not even fetched, so
    ``missing_total`` is 0 and nothing is indexed or embedded.

    This is deliberately the *same* call whether it's the first run ever (on
    a fresh deployment "missing" is simply the whole catalog — this doubles as
    backfill) or a routine later reconciliation (only genuine drift surfaces).

    Requires ``report_search_index_state_enabled``, and raises
    :class:`SearchIndexSyncError` without it: NADA's diff is computed from its
    ``search_index_state``, which only changes when nada-ai reports what it
    indexed/deleted. Without reporting, nothing ever leaves the diff, and the run
    could not tell the work it did from the work still to do.

    Re-fetches each diff live on every iteration rather than paging through a
    fixed snapshot: a successfully indexed/deleted item drops out of the
    "missing"/"stale" result on the *next* fetch (NADA's own diff query, not a
    local cache), so naive ``offset += page_size`` would skip entries as the set
    shrinks mid-run. An item that stays — one that failed (correctly: it's still
    not indexed), was skipped, or was already handled earlier in this run —
    moves the next fetch's offset past it, so a page full of them never hides
    the rest of the diff. The in-memory ``seen`` sets mean each item is
    attempted once per run, not retried every time it comes back.

    ``progress_cb``, if given, is called with a Jobs-shaped snapshot
    (``processed``/``total``/``percent``/``failed``/``current_idno``) after
    totals are known and again after every idno — same shape as
    ``IngestProgressTracker`` so ``GET /jobs/{id}`` can render a bar.
    ``cancel_token`` is checked once per idno so Stop actually stops the
    worker thread (``asyncio.Task.cancel`` alone cannot).
    """
    if not settings.report_search_index_state_enabled:
        raise SearchIndexSyncError(DIFF_RECONCILE_NEEDS_STATE_REPORTING)

    summary: dict[str, Any] = {
        "missing_total": 0,
        "stale_total": 0,
        "indexed": 0,
        "deleted": 0,
        "failed": 0,
        "skipped": 0,
    }

    # Peek both sides first so Jobs has a stable denominator from tick 0
    # instead of a missing-only total that jumps when stale work starts.
    # These first pages also seed the loops below — no extra HTTP vs. the
    # previous "fetch inside the loop" shape.
    missing_page = (
        DiffPage(items=[], total=0)
        if remove_only
        else list_diff_missing(settings, object_type=object_type, limit=page_size, offset=0, data_type=data_type)
    )
    stale_page = list_diff_stale(settings, object_type=object_type, limit=page_size, offset=0, data_type=data_type)
    summary["missing_total"] = missing_page.total
    summary["stale_total"] = stale_page.total
    total = missing_page.total + stale_page.total
    processed = 0

    def _emit(current_idno: str | None = None, *, phase: str) -> None:
        if progress_cb is None:
            return
        progress_cb(
            {
                "processed": processed,
                "total": total,
                "failed": summary["failed"],
                "current_idno": current_idno,
                "percent": round(100 * processed / total, 1) if total else 100.0,
                "phase": phase,
            }
        )

    def _cancelled() -> bool:
        return cancel_token is not None and cancel_token.is_set()

    _emit(phase="missing")

    seen_missing: set[str] = set()
    page = missing_page
    offset = 0
    for i in range(_DIFF_RECONCILE_MAX_ITERATIONS):
        if _cancelled():
            summary["cancelled"] = True
            _emit(phase="missing")
            return summary
        if i > 0:
            page = list_diff_missing(
                settings, object_type=object_type, limit=page_size, offset=offset, data_type=data_type
            )
        if not page.items:
            break
        new_items = [it for it in page.items if it.idno not in seen_missing]
        succeeded = 0
        for item in new_items:
            if _cancelled():
                summary["cancelled"] = True
                _emit(current_idno=item.idno, phase="missing")
                return summary
            seen_missing.add(item.idno)
            resolved_type = _DATASET_TYPE_TO_METADATA_TYPE.get(item.type) if item.type else None
            if resolved_type is None:
                summary["skipped"] += 1
                logger.warning(
                    "diff reconcile: no metadata_type mapping for idno=%s dataset_type=%s", item.idno, item.type
                )
                processed += 1
                _emit(item.idno, phase="missing")
                continue
            try:
                result = index_ids_op(
                    settings,
                    idnos=[item.idno],
                    metadata_type=resolved_type,
                    force=True,
                    show_progress_bar=False,
                    embedding=embedding,
                )
            except Exception as e:  # noqa: BLE001 - one idno's failure must not stop the reconcile run
                logger.warning("diff reconcile: index failed for idno=%s: %s", item.idno, e)
                summary["failed"] += 1
                processed += 1
                _emit(item.idno, phase="missing")
                continue

            # index_ids_op never raises for a per-idno failure (metadata failed to
            # load/parse, or loaded but produced no documents, or a backend write
            # error) — those are reported in its return value instead, so a call
            # that didn't raise does not by itself mean this idno got indexed.
            # Since this call is scoped to exactly one idno, any of these being
            # non-empty can only be about *this* idno.
            variables_failed = (result.get("variables") or {}).get("failed")
            if result.get("load_errors") or result.get("empty_docs") or result.get("errors") or variables_failed:
                summary["failed"] += 1
                logger.warning("diff reconcile: index reported no success for idno=%s: %s", item.idno, result)
            else:
                summary["indexed"] += 1
                succeeded += 1
            processed += 1
            _emit(item.idno, phase="missing")
        # Everything else on this page is still in the diff: move the next fetch past it.
        offset += len(page.items) - succeeded

    seen_stale: set[str] = set()
    page = stale_page
    offset = 0
    for i in range(_DIFF_RECONCILE_MAX_ITERATIONS):
        if _cancelled():
            summary["cancelled"] = True
            _emit(phase="stale")
            return summary
        if i > 0:
            page = list_diff_stale(
                settings, object_type=object_type, limit=page_size, offset=offset, data_type=data_type
            )
        if not page.items:
            break
        new_idnos = [it.idno for it in page.items if it.idno not in seen_stale]
        seen_stale.update(new_idnos)
        succeeded = 0
        if new_idnos:
            try:
                delete_by_idnos_op(settings, new_idnos)
                summary["deleted"] += len(new_idnos)
                succeeded = len(new_idnos)
            except Exception as e:  # noqa: BLE001 - report and move on, same as the indexing loop above
                logger.warning("diff reconcile: delete failed for idnos=%s: %s", new_idnos, e)
                summary["failed"] += len(new_idnos)
        offset += len(page.items) - succeeded
        for idno in new_idnos:
            processed += 1
            _emit(idno, phase="stale")
            if _cancelled():
                summary["cancelled"] = True
                return summary

    return summary


def _scan(client: Any, index: str, fields: list[str], page_size: int = 1000):
    """Every document's ``fields`` in ``index``, a page at a time (search_after on _id). Nothing when it does not exist."""
    if not client.indices.exists(index=index):
        return
    after: list[Any] | None = None
    while True:
        body: dict[str, Any] = {"size": page_size, "_source": fields, "sort": [{"_id": "asc"}], "query": {"match_all": {}}}
        if after is not None:
            body["search_after"] = after
        hits = client.search(index=index, body=body)["hits"]["hits"]
        if not hits:
            return
        for hit in hits:
            yield hit.get("_source") or {}
        after = hits[-1]["sort"]


def _sids_with_chunks(client: Any, index: str) -> set[int]:
    """The sids that have at least one chunk (embedded text) in the chunk index."""
    from nada_ai.search.backend.opensearch.mapping import metadata_field

    if not client.indices.exists(index=index):
        return set()
    sids: set[int] = set()
    after: dict[str, Any] | None = None
    while True:
        composite: dict[str, Any] = {"size": 1000, "sources": [{"sid": {"terms": {"field": metadata_field("sid")}}}]}
        if after is not None:
            composite["after"] = after
        resp = client.search(index=index, body={"size": 0, "aggs": {"sids": {"composite": composite}}})
        agg = resp["aggregations"]["sids"]
        sids.update(int(b["key"]["sid"]) for b in agg["buckets"] if b["key"].get("sid") is not None)
        after = agg.get("after_key")
        if not agg["buckets"] or after is None:
            return sids


def import_index_state(
    settings: Settings,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Tell NADA what this deployment's index already holds, then have it queue what is left to do.

    For an index that existed before NADA tracked it (NADA cleared its tracking when its index engine changed): every
    study in the study index that has chunks, and every citation in the citation index, is reported ``indexed`` by
    sid / citation id — also the ones gone from NADA's catalog, so NADA's stale diff finds them. Studies without
    chunks are not reported, so they count as missing and are indexed in full. NADA then queues (``refresh_indexed``)
    a delete for what is gone, a partial update (published status, filters; no embedding) for what is indexed, and a
    full index for what is missing.

    Reads only document ids and keys from OpenSearch; embeds nothing.
    """
    from nada_ai.search.backend.opensearch.client import build_client

    if settings.search_backend != "opensearch":
        raise SearchIndexSyncError("Importing the index state needs the OpenSearch backend")

    client = build_client(settings)
    items: list[dict[str, Any]] = []
    try:
        with_chunks = _sids_with_chunks(client, settings.index_name)
        studies = without_chunks = 0
        for doc in _scan(client, settings.studies_index, ["sid", "idno"]):
            sid, idno = doc.get("sid"), doc.get("idno")
            if sid is None or not idno:
                continue
            if int(sid) not in with_chunks:
                without_chunks += 1
                continue
            studies += 1
            items.append({"object_type": "survey", "object_key": str(idno), "object_id": int(sid), "status": "indexed"})
        citations = 0
        for doc in _scan(client, settings.citations_index, ["citation_id", "uuid"]):
            cid, uuid = doc.get("citation_id"), doc.get("uuid")
            if cid is None or not uuid:
                continue
            citations += 1
            items.append({"object_type": "citation", "object_key": str(uuid), "object_id": int(cid), "status": "indexed"})
    finally:
        client.close()

    summary: dict[str, Any] = {
        "studies": studies,
        "studies_without_chunks": without_chunks,
        "citations": citations,
        "reported": 0,
        "skipped": 0,
    }
    total = len(items)
    for start in range(0, total, _STATE_BULK_CHUNK_SIZE):
        if cancel_token is not None and cancel_token.is_set():
            summary["cancelled"] = True
            return summary
        result = report_state_bulk(settings, items[start : start + _STATE_BULK_CHUNK_SIZE])
        summary["reported"] += int(result.get("applied") or 0)
        summary["skipped"] += sum(1 for r in result.get("results") or [] if not r.get("applied"))
        if progress_cb is not None:
            done = min(start + _STATE_BULK_CHUNK_SIZE, total)
            progress_cb({"processed": done, "total": total, "percent": round(100 * done / total, 1) if total else 100.0})

    summary["queued"] = trigger_sync(settings, object_type="all", refresh_indexed=True).get("queued")
    return summary
