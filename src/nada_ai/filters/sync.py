"""Sync external filter dicts to the filters of every indexed document of a study (by its NADA id, ``sid``)."""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from qdrant_client.http import models as qm

from nada_ai.filters.indexes import (
    ensure_opensearch_filter_facets_mapping,
    ensure_qdrant_filter_field_indexes,
    ensure_qdrant_filter_field_indexes_for_keys,
    qdrant_dynamic_facet_indexes_ready,
)
from nada_ai.ingest.qdrant_writer import _client as qdrant_client
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.mapping import METADATA_OBJECT_KEY, metadata_field
from nada_ai.search.dynamic_filters import (
    FILTER_FACETS_KEY,
    FILTER_FIELDS_KEY,
    FIXED_FILTER_KEYS,
    facets_map_from_filter_fields_rows,
    facets_to_filter_field_rows,
    load_dynamic_facet_keys,
    load_excluded_facet_keys,
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
    sid: int
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


def _sid_filter(sid: int) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key=metadata_field("sid"), match=qm.MatchValue(value=sid))])


def _sync_qdrant(settings: Settings, sid: int, normalized: list[dict[str, list[str]]]) -> FilterSyncResult:
    coll = settings.qdrant_collection
    client = qdrant_client(settings)
    try:
        sid_fl = _sid_filter(sid)
        count_resp = client.count(collection_name=coll, count_filter=sid_fl)
        point_count = int(count_resp.count)
        if point_count == 0:
            return FilterSyncResult(sid=sid, updated_points=0, found=False)

        # Merge into nested metadata; a top-level {"metadata": {...}} would replace the whole object.
        client.set_payload(
            collection_name=coll,
            payload={
                FILTER_FIELDS_KEY: normalized,
                FILTER_FACETS_KEY: normalized_to_facets_map(normalized),
            },
            key=METADATA_OBJECT_KEY,
            points=qm.FilterSelector(filter=sid_fl),
            wait=True,
        )
        return FilterSyncResult(sid=sid, updated_points=point_count, found=True)
    finally:
        client.close()


def _sync_opensearch(settings: Settings, sid: int, normalized: list[dict[str, list[str]]]) -> FilterSyncResult:
    """Rewrite one study's flat ``filter_facets`` on its chunk documents and on its study document."""
    client = build_client(settings)
    try:
        indexed = client.exists(index=settings.studies_index, id=str(sid))
        chunk_query = {"term": {metadata_field("sid"): sid}}
        count_resp = client.count(index=settings.index_name, body={"query": chunk_query}, ignore_unavailable=True)
        if int(count_resp.get("count") or 0) == 0 and not indexed:
            return FilterSyncResult(sid=sid, updated_points=0, found=False)

        facets = normalized_to_facets_map(normalized)
        chunk_script = {
            "source": (
                "if (ctx._source.metadata == null) { ctx._source.metadata = new HashMap(); } "
                "ctx._source.metadata.filter_facets = params.facets;"
            ),
            "params": {"facets": facets},
        }
        resp = client.update_by_query(
            index=settings.index_name, body={"query": chunk_query, "script": chunk_script}, refresh=True
        )
        if indexed:
            # The study index keeps only a few fields in its _source, so it is never updated in place (that would drop
            # its searchable text): the study document is rewritten from NADA's extract record instead.
            from nada_ai.ingest.service import apply_study_options_op

            apply_study_options_op(settings, sid)
        return FilterSyncResult(sid=sid, updated_points=int(resp.get("updated") or 0), found=True)
    finally:
        try:
            client.transport.close()
        except Exception:
            pass


def auto_register_new_facet_keys(settings: Settings, keys: list[str]) -> list[str]:
    """Promote any not-yet-known key in ``keys`` to facetable, indexing it (Qdrant).

    Called on every filter sync so a newly-observed key from NADA's data
    (rather than a human editing the registry) becomes facetable automatically
    — see the discussion in docs/GUIDE.md's dynamic-filters section. Skips:

    - keys already facetable (nothing to do)
    - keys on the excluded (deny) list — an admin explicitly removed one of
      these via ``remove_facet_keys``; auto-registration must not resurrect
      it just because NADA sent it again
    - keys in ``FIXED_FILTER_KEYS`` — handled via the static metadata path,
      not the dynamic facet path, promoting them would be a no-op at best

    Returns the list of keys actually newly registered.
    """
    if not keys:
        return []
    current = load_dynamic_facet_keys(settings)
    excluded = load_excluded_facet_keys(settings)
    candidates = frozenset(str(k).strip() for k in keys if str(k).strip())
    new_keys = sorted(candidates - current - excluded - FIXED_FILTER_KEYS)
    if not new_keys:
        return []

    # Import here (not at module top) to avoid a facets_service <-> sync circular import.
    from nada_ai.filters.facets_service import add_facet_keys

    add_facet_keys(settings, new_keys)
    logger.info("Auto-registered new facet key(s) from NADA data: %s", new_keys)

    if settings.search_backend == "qdrant":
        client = qdrant_client(settings)
        try:
            ensure_qdrant_filter_field_indexes_for_keys(client, settings.qdrant_collection, new_keys)
        finally:
            client.close()

    return new_keys


def sync_filters_for_sid(settings: Settings, sid: int, filters_dict: dict[str, Any]) -> FilterSyncResult:
    """Normalize and write the filters of every indexed document of study ``sid``.

    Also auto-registers (and, on Qdrant, indexes) any key present in
    ``filters_dict`` that isn't already a known facet — see
    ``auto_register_new_facet_keys``. This runs before the write so the
    facet index exists before the data that would populate it lands.
    """
    sid = int(sid)
    if sid <= 0:
        raise ValueError("sid must be a positive integer")
    normalized = normalize_external_filters(filters_dict)
    auto_register_new_facet_keys(settings, [entry["key"] for entry in normalized])
    if settings.search_backend == "qdrant":
        return _sync_qdrant(settings, sid, normalized)
    return _sync_opensearch(settings, sid, normalized)


def sync_filters_batch(
    settings: Settings,
    records: list[dict[str, Any]],
    *,
    show_progress_bar: bool = False,
) -> list[FilterSyncResult]:
    """Sync ``[{sid, filters}, ...]`` records (a record without ``filters`` holds them at its top level)."""
    results: list[FilterSyncResult] = []
    iterator: Any = records
    if show_progress_bar and records:
        from tqdm.auto import tqdm

        iterator = tqdm(records, unit="study", desc="Sync filter_fields")

    for record in iterator:
        try:
            sid = int(record.get("sid") or 0)
        except (TypeError, ValueError):
            sid = 0
        if sid <= 0:
            results.append(FilterSyncResult(sid=0, updated_points=0, found=False))
            continue
        filters_raw = record.get("filters")
        if filters_raw is None:
            filters_raw = {k: v for k, v in record.items() if k != "sid"}
        results.append(sync_filters_for_sid(settings, sid, filters_raw))
    return results


def get_filter_fields_for_sid(settings: Settings, sid: int) -> dict[str, Any]:
    """Read the stored filters and document count of study ``sid`` (sampled from its first document)."""
    sid = int(sid)
    if settings.search_backend == "qdrant":
        coll = settings.qdrant_collection
        client = qdrant_client(settings)
        try:
            sid_fl = _sid_filter(sid)
            count_resp = client.count(collection_name=coll, count_filter=sid_fl)
            point_count = int(count_resp.count)
            if point_count == 0:
                return {"sid": sid, "found": False, "point_count": 0, "filter_fields": None}
            batch, _ = client.scroll(
                collection_name=coll,
                scroll_filter=sid_fl,
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
                "sid": sid,
                "found": True,
                "point_count": point_count,
                "filter_fields": ff,
                "filter_facets": facets,
            }
        finally:
            client.close()

    client = build_client(settings)
    try:
        body = {
            "size": 1,
            "track_total_hits": True,
            "query": {"term": {metadata_field("sid"): sid}},
            "_source": {"includes": [metadata_field(FILTER_FACETS_KEY)]},
        }
        resp = client.search(index=settings.index_name, body=body, ignore_unavailable=True)
        hits = resp.get("hits", {}).get("hits") or []
        total = resp.get("hits", {}).get("total") or {}
        point_count = int(total.get("value") or 0) if isinstance(total, dict) else int(total or 0)
        if not hits:
            return {"sid": sid, "found": False, "point_count": 0, "filter_fields": None}
        meta = (hits[0].get("_source") or {}).get("metadata") or {}
        facets = meta.get(FILTER_FACETS_KEY)
        return {
            "sid": sid,
            "found": True,
            "point_count": point_count,
            "filter_fields": facets_to_filter_field_rows(facets),
            "filter_facets": facets,
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
            indexes = ensure_qdrant_filter_field_indexes(
                client, settings.qdrant_collection, strict=True, settings=settings
            )
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
        mapping = ensure_opensearch_filter_facets_mapping(client, settings)
        return {"backend": "opensearch", **mapping}
    finally:
        try:
            client.transport.close()
        except Exception:
            pass


def parse_filters_input(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse CLI/API filter payload (supports wrapped ``filters`` key)."""
    return unwrap_external_filters(raw)
