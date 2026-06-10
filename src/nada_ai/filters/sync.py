"""Sync external filter dicts to ``metadata.filter_fields`` on all points for an idno."""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from nada_ai.filters.indexes import (
    ensure_opensearch_filter_fields_mapping,
    ensure_qdrant_filter_field_indexes,
    qdrant_dynamic_facet_indexes_ready,
)
from nada_ai.ingest.qdrant_writer import _client as qdrant_client
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.mapping import METADATA_OBJECT_KEY, metadata_field
from nada_ai.search.dynamic_filters import (
    FILTER_FACETS_KEY,
    FILTER_FIELDS_KEY,
    facets_map_from_filter_fields_rows,
    normalize_external_filters,
    normalized_to_facets_map,
    unwrap_external_filters,
)
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


class FilterBackfillResult(TypedDict):
    updated_points: int
    skipped_points: int
    groups_written: int


class FilterSyncResult(TypedDict):
    idno: str
    updated_points: int
    found: bool


def _needs_filter_facets_backfill(meta: dict[str, Any]) -> dict[str, list[str]] | None:
    facets = meta.get(FILTER_FACETS_KEY)
    if isinstance(facets, dict) and facets:
        return None
    rows = meta.get(FILTER_FIELDS_KEY)
    if not isinstance(rows, list) or not rows:
        return None
    derived = facets_map_from_filter_fields_rows(rows)
    return derived or None


def backfill_filter_facets(settings: Settings, *, show_progress_bar: bool = False) -> FilterBackfillResult:
    """Populate ``metadata.filter_facets`` from existing ``filter_fields`` on all Qdrant points."""
    if settings.search_backend != "qdrant":
        raise ValueError("backfill_filter_facets is only supported for the qdrant backend")

    coll = settings.qdrant_collection
    client = qdrant_client(settings)
    updated = 0
    skipped = 0
    groups_written = 0
    try:
        from collections import defaultdict

        pending: dict[str, list[Any]] = defaultdict(list)
        offset: Any = None
        iterator: Any = None
        while True:
            batch, offset = client.scroll(
                collection_name=coll,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not batch:
                break
            if iterator is None and show_progress_bar:
                from tqdm.auto import tqdm

                total = client.count(collection_name=coll).count
                iterator = tqdm(total=total, unit="point", desc="Backfill filter_facets")
            for point in batch:
                if iterator is not None:
                    iterator.update(1)
                meta = (point.payload or {}).get("metadata") or {}
                if not isinstance(meta, dict):
                    skipped += 1
                    continue
                derived = _needs_filter_facets_backfill(meta)
                if derived is None:
                    skipped += 1
                    continue
                sig = json.dumps(derived, sort_keys=True)
                pending[sig].append(point.id)
            if offset is None:
                break

        if iterator is not None:
            iterator.close()

        write_iter: Any = pending.items()
        if show_progress_bar and pending:
            from tqdm.auto import tqdm

            write_iter = tqdm(pending.items(), unit="group", desc="Write filter_facets")

        for sig, point_ids in write_iter:
            facets = json.loads(sig)
            client.set_payload(
                collection_name=coll,
                payload={FILTER_FACETS_KEY: facets},
                key=METADATA_OBJECT_KEY,
                points=point_ids,
                wait=True,
            )
            updated += len(point_ids)
            groups_written += 1
    finally:
        client.close()

    return FilterBackfillResult(
        updated_points=updated,
        skipped_points=skipped,
        groups_written=groups_written,
    )


def _idno_filter(idno: str) -> qm.Filter:
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key=metadata_field("idno"),
                match=qm.MatchValue(value=idno),
            )
        ]
    )


def _sync_qdrant(settings: Settings, idno: str, normalized: list[dict[str, list[str]]]) -> FilterSyncResult:
    coll = settings.qdrant_collection
    client = qdrant_client(settings)
    try:
        idno_fl = _idno_filter(idno)
        count_resp = client.count(collection_name=coll, count_filter=idno_fl)
        point_count = int(count_resp.count)
        if point_count == 0:
            return FilterSyncResult(idno=idno, updated_points=0, found=False)

        # Merge into nested metadata; a top-level {"metadata": {...}} would replace the whole object.
        client.set_payload(
            collection_name=coll,
            payload={
                FILTER_FIELDS_KEY: normalized,
                FILTER_FACETS_KEY: normalized_to_facets_map(normalized),
            },
            key=METADATA_OBJECT_KEY,
            points=qm.FilterSelector(filter=idno_fl),
            wait=True,
        )
        return FilterSyncResult(idno=idno, updated_points=point_count, found=True)
    finally:
        client.close()


def _sync_opensearch(settings: Settings, idno: str, normalized: list[dict[str, list[str]]]) -> FilterSyncResult:
    index_name = settings.index_name
    client = build_client(settings)
    try:
        count_body = {"query": {"term": {metadata_field("idno"): idno}}}
        count_resp = client.count(index=index_name, body=count_body)
        point_count = int(count_resp.get("count") or 0)
        if point_count == 0:
            return FilterSyncResult(idno=idno, updated_points=0, found=False)

        update_body = {
            "query": {"term": {metadata_field("idno"): idno}},
            "script": {
                "source": (
                    "if (ctx._source.metadata == null) { ctx._source.metadata = new HashMap(); } "
                    "ctx._source.metadata.filter_fields = params.ff;"
                ),
                "params": {"ff": normalized},
            },
        }
        resp = client.update_by_query(index=index_name, body=update_body, refresh=True)
        updated = int(resp.get("updated") or 0)
        return FilterSyncResult(idno=idno, updated_points=updated, found=True)
    finally:
        try:
            client.transport.close()
        except Exception:
            pass


def sync_filters_for_idno(settings: Settings, idno: str, filters_dict: dict[str, Any]) -> FilterSyncResult:
    """Normalize and write ``filter_fields`` to every indexed point for ``idno``."""
    idno = idno.strip()
    if not idno:
        raise ValueError("idno must be non-empty")
    normalized = normalize_external_filters(filters_dict)
    if settings.search_backend == "qdrant":
        return _sync_qdrant(settings, idno, normalized)
    return _sync_opensearch(settings, idno, normalized)


def sync_filters_batch(
    settings: Settings,
    records: list[dict[str, Any]],
    *,
    show_progress_bar: bool = False,
) -> list[FilterSyncResult]:
    results: list[FilterSyncResult] = []
    iterator: Any = records
    if show_progress_bar and records:
        from tqdm.auto import tqdm

        iterator = tqdm(records, unit="idno", desc="Sync filter_fields")

    for record in iterator:
        idno = str(record.get("idno") or "").strip()
        if not idno:
            results.append(FilterSyncResult(idno="", updated_points=0, found=False))
            continue
        filters_raw = record.get("filters")
        if filters_raw is None:
            filters_raw = {k: v for k, v in record.items() if k != "idno"}
        results.append(sync_filters_for_idno(settings, idno, filters_raw))
    return results


def get_filter_fields_for_idno(settings: Settings, idno: str) -> dict[str, Any]:
    """Read ``filter_fields`` and point count for an idno (sample from first matching point)."""
    idno = idno.strip()
    if settings.search_backend == "qdrant":
        coll = settings.qdrant_collection
        client = qdrant_client(settings)
        try:
            idno_fl = _idno_filter(idno)
            count_resp = client.count(collection_name=coll, count_filter=idno_fl)
            point_count = int(count_resp.count)
            if point_count == 0:
                return {"idno": idno, "found": False, "point_count": 0, "filter_fields": None}
            batch, _ = client.scroll(
                collection_name=coll,
                scroll_filter=idno_fl,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            ff = None
            facets = None
            if batch:
                meta = (batch[0].payload or {}).get("metadata") or {}
                ff = meta.get(FILTER_FIELDS_KEY)
                facets = meta.get(FILTER_FACETS_KEY)
            return {
                "idno": idno,
                "found": True,
                "point_count": point_count,
                "filter_fields": ff,
                "filter_facets": facets,
            }
        finally:
            client.close()

    index_name = settings.index_name
    client = build_client(settings)
    try:
        body = {
            "size": 1,
            "query": {"term": {metadata_field("idno"): idno}},
            "_source": {"includes": [metadata_field(FILTER_FIELDS_KEY)]},
        }
        resp = client.search(index=index_name, body=body)
        hits = resp.get("hits", {}).get("hits") or []
        total = resp.get("hits", {}).get("total") or {}
        if isinstance(total, dict):
            point_count = int(total.get("value") or 0)
        else:
            point_count = int(total or 0)
        if not hits:
            return {"idno": idno, "found": False, "point_count": 0, "filter_fields": None}
        src = hits[0].get("_source") or {}
        meta = src.get("metadata") or {}
        return {
            "idno": idno,
            "found": True,
            "point_count": point_count,
            "filter_fields": meta.get(FILTER_FIELDS_KEY),
            "filter_facets": meta.get(FILTER_FACETS_KEY),
        }
    finally:
        try:
            client.transport.close()
        except Exception:
            pass


def ensure_filter_indexes_op(settings: Settings) -> dict[str, Any]:
    """Ensure dynamic filter indexes/mappings exist on the active backend."""
    if settings.search_backend == "qdrant":
        client = qdrant_client(settings)
        try:
            indexes = ensure_qdrant_filter_field_indexes(client, settings.qdrant_collection, strict=True, settings=settings)
            ready = qdrant_dynamic_facet_indexes_ready(client, settings.qdrant_collection, settings)
            return {
                "backend": "qdrant",
                "collection": settings.qdrant_collection,
                "indexes": indexes,
                "ready": ready,
            }
        finally:
            client.close()

    client = build_client(settings)
    try:
        mapping = ensure_opensearch_filter_fields_mapping(client, settings.index_name)
        return {"backend": "opensearch", **mapping}
    finally:
        try:
            client.transport.close()
        except Exception:
            pass


def parse_filters_input(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse CLI/API filter payload (supports wrapped ``filters`` key)."""
    return unwrap_external_filters(raw)
