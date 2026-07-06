"""Dynamic filter_fields normalization, query building, and facet helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from qdrant_client.http import models as qm

from nada_ai.search.backend.opensearch.mapping import metadata_field
from nada_ai.settings import Settings

FILTER_FIELDS_KEY = "filter_fields"
FILTER_FACETS_KEY = "filter_facets"
FILTER_FIELDS_PATH = metadata_field(FILTER_FIELDS_KEY)

FIXED_FILTER_KEYS = frozenset(
    {
        "type",
        "idno",
        "idnos",
        "geographies",
        "source",
        "periodicity",
        "document_type",
        "authors",
        "year_start",
        "year_end",
    }
)

_DEFAULT_FACETS_PATH = Path(__file__).resolve().parents[3] / "config" / "dynamic_filter_facets.json"
_DEFAULT_FACETABLE: frozenset[str] = frozenset(
    {
        "doctype",
        "published",
        "dataset_type",
        "repositoryid",
        "repositories",
        "countries",
        "regions",
        "tags",
        "years",
    }
)


def _resolve_facets_path(settings: Settings | None) -> tuple[Path, bool]:
    """Return (path, from_explicit_config).

    ``from_explicit_config`` is True when the path comes from settings or the
    default config file, False when we are falling back to built-in defaults
    (no file at all).  The returned path is always where we *would* write even
    if it does not yet exist.
    """
    if settings is not None and settings.dynamic_filter_facets_path:
        return Path(settings.dynamic_filter_facets_path), True
    return _DEFAULT_FACETS_PATH, _DEFAULT_FACETS_PATH.is_file()


def load_dynamic_facet_keys(settings: Settings | None = None) -> frozenset[str]:
    """Load facetable dynamic filter keys from config (env override optional)."""
    path, from_file = _resolve_facets_path(settings)
    if not from_file or not path.is_file():
        return _DEFAULT_FACETABLE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = data.get("facetable") or []
        return frozenset(str(k) for k in keys)
    except (OSError, json.JSONDecodeError, TypeError):
        return _DEFAULT_FACETABLE


def save_dynamic_facet_keys(
    keys: frozenset[str] | set[str] | list[str],
    settings: Settings | None = None,
) -> Path:
    """Atomically persist ``keys`` to the facets config file.

    Creates parent directories if needed.  Uses ``tempfile.mkstemp`` for a
    unique per-call temp file (so concurrent callers never clobber each other's
    in-flight write) then renames atomically into place.  Returns the path
    written.
    """
    path, _ = _resolve_facets_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps({"facetable": sorted(str(k) for k in keys if str(k).strip())}, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    fd, tmp_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    tmp = Path(tmp_str)
    try:
        try:
            os.write(fd, content)
        finally:
            os.close(fd)  # always release the OS file descriptor
        tmp.replace(path)  # atomic on POSIX; near-atomic on Windows
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def unwrap_external_filters(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept ``{"filters": {...}}`` or a bare filter dict."""
    if "filters" in raw and isinstance(raw["filters"], dict):
        return dict(raw["filters"])
    return dict(raw)


def _coerce_value_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def normalized_to_facets_map(normalized: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Flat map for Qdrant payload indexes / native facet API (``metadata.filter_facets.<key>``)."""
    return {str(entry["key"]): list(entry["value"]) for entry in normalized}


def dynamic_facet_qdrant_key(field: str) -> str:
    """Indexed payload path for faceting one dynamic filter key in Qdrant."""
    return metadata_field(f"{FILTER_FACETS_KEY}.{field}")


def normalize_external_filters(raw: dict[str, Any]) -> list[dict[str, list[str]]]:
    """Convert external filter dict to stored ``[{key, value: [str, ...]}, ...]``."""
    filters = unwrap_external_filters(raw)
    entries: list[dict[str, list[str]]] = []
    for key, value in filters.items():
        if value is None:
            continue
        values = _coerce_value_strings(value)
        if not values:
            continue
        entries.append({"key": str(key), "value": values})
    entries.sort(key=lambda e: e["key"])
    return entries


def split_filters(filters: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition request filters into fixed (legacy metadata) and dynamic (filter_fields) keys."""
    if not filters:
        return {}, {}
    fixed: dict[str, Any] = {}
    dynamic: dict[str, Any] = {}
    for key, value in filters.items():
        if value is None:
            continue
        if key in FIXED_FILTER_KEYS:
            fixed[key] = value
        else:
            dynamic[key] = value
    return fixed, dynamic


def _query_values(value: Any) -> list[str]:
    return _coerce_value_strings(value)


def dynamic_filters_to_qdrant_conditions(dynamic: dict[str, Any]) -> list[qm.Condition]:
    """Build Qdrant conditions on ``metadata.filter_facets.<key>`` (flat, indexed paths)."""
    if not dynamic:
        return []
    clauses: list[qm.Condition] = []
    for key, value in dynamic.items():
        values = _query_values(value)
        if not values:
            continue
        value_cond = (
            qm.MatchAny(any=values) if len(values) > 1 else qm.MatchValue(value=values[0])
        )
        clauses.append(
            qm.FieldCondition(
                key=dynamic_facet_qdrant_key(str(key)),
                match=value_cond,
            )
        )
    return clauses


def dynamic_filters_to_opensearch_clauses(dynamic: dict[str, Any]) -> list[dict[str, Any]]:
    if not dynamic:
        return []
    clauses: list[dict[str, Any]] = []
    for key, value in dynamic.items():
        values = _query_values(value)
        if not values:
            continue
        must: list[dict[str, Any]] = [
            {"term": {f"{FILTER_FIELDS_PATH}.key": str(key)}},
        ]
        if len(values) == 1:
            must.append({"term": {f"{FILTER_FIELDS_PATH}.value": values[0]}})
        else:
            must.append({"terms": {f"{FILTER_FIELDS_PATH}.value": values}})
        clauses.append(
            {
                "nested": {
                    "path": FILTER_FIELDS_PATH,
                    "query": {"bool": {"must": must}},
                }
            }
        )
    return clauses


def _filter_fields_array_map(meta: dict[str, Any]) -> dict[str, set[str]]:
    """Build key -> values from legacy ``metadata.filter_fields`` array."""
    rows = meta.get(FILTER_FIELDS_KEY) or []
    out: dict[str, set[str]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        k = row.get("key")
        if k is None:
            continue
        stored = _coerce_value_strings(row.get("value"))
        if stored:
            out[str(k)] = set(stored)
    return out


def stored_dynamic_filters_map(sample: dict[str, Any]) -> dict[str, set[str]]:
    """Build key -> stored values from ``metadata.filter_facets`` (preferred) or ``filter_fields``."""
    meta = sample.get("metadata") if "metadata" in sample else sample
    if not isinstance(meta, dict):
        return {}
    facets = meta.get(FILTER_FACETS_KEY)
    if isinstance(facets, dict) and facets:
        out: dict[str, set[str]] = {}
        for key, value in facets.items():
            stored = _coerce_value_strings(value)
            if stored:
                out[str(key)] = set(stored)
        if out:
            return out
    return _filter_fields_array_map(meta)


def facets_map_from_filter_fields_rows(rows: list[dict[str, Any]] | None) -> dict[str, list[str]]:
    """Derive ``filter_facets`` map from stored ``filter_fields`` rows."""
    if not rows:
        return {}
    return normalized_to_facets_map(rows)


def match_dynamic_filters(sample: dict[str, Any], dynamic: dict[str, Any]) -> dict[str, Any]:
    """Evaluate dynamic filters against ``sample`` (metadata dict or full _source)."""
    if not dynamic:
        return {"all_matched": True, "per_field": {}}
    stored = stored_dynamic_filters_map(sample)
    per: dict[str, Any] = {}
    ok = True
    for key, value in dynamic.items():
        expected = _query_values(value)
        actual = sorted(stored.get(str(key), set()))
        if not expected:
            m = True
        elif not actual:
            m = False
        else:
            m = bool(set(expected) & set(actual))
        per[str(key)] = {"expected": expected, "actual": actual, "matched": m}
        ok &= m
    return {"all_matched": ok, "per_field": per}


def resolve_facet_fields(
    requested: list[str] | None,
    settings: Settings | None = None,
) -> tuple[list[str], list[str]]:
    """Return (static_facet_fields, dynamic_facet_fields)."""
    from nada_ai.search.backend.opensearch.queries import FACET_FIELD_WHITELIST

    dynamic_keys = load_dynamic_facet_keys(settings)
    if requested:
        static = [f for f in requested if f in FACET_FIELD_WHITELIST]
        dynamic = [f for f in requested if f in dynamic_keys]
        return static, dynamic
    return sorted(FACET_FIELD_WHITELIST), sorted(dynamic_keys)


def dynamic_facet_aggs(dynamic_keys: list[str]) -> dict[str, Any]:
    """OpenSearch nested aggregations for dynamic filter_fields facets."""
    if not dynamic_keys:
        return {}
    aggs: dict[str, Any] = {}
    for name in dynamic_keys:
        aggs[name] = {
            "nested": {"path": FILTER_FIELDS_PATH},
            "aggs": {
                "filtered": {
                    "filter": {"term": {f"{FILTER_FIELDS_PATH}.key": name}},
                    "aggs": {
                        "values": {
                            "terms": {"field": f"{FILTER_FIELDS_PATH}.value", "size": 200},
                        }
                    },
                }
            },
        }
    return aggs


def unwrap_dynamic_facet_buckets(field: str, agg: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize nested agg buckets for a dynamic facet field."""
    filtered = agg.get("filtered") or {}
    values = filtered.get("values") or {}
    buckets = values.get("buckets") or []
    return [{"value": b.get("key"), "count": int(b.get("doc_count", 0))} for b in buckets]


def aggregate_dynamic_facet_rows(
    payloads: list[dict[str, Any]],
    field: str,
    *,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Count facet values for one dynamic key from point payloads (legacy scroll fallback)."""
    return aggregate_dynamic_facet_rows_multi(payloads, [field], limit=limit).get(field, [])


def aggregate_dynamic_facet_rows_multi(
    payloads: list[dict[str, Any]],
    fields: list[str],
    *,
    limit: int = 200,
) -> dict[str, list[dict[str, Any]]]:
    """Count facet values for multiple dynamic keys in one pass over payloads."""
    from collections import Counter

    if not fields:
        return {}
    field_set = frozenset(fields)
    counts: dict[str, Counter[str]] = {field: Counter() for field in fields}
    for payload in payloads:
        stored = stored_dynamic_filters_map(payload)
        for key in field_set:
            for value in sorted(stored.get(key, set())):
                counts[key][value] += 1
    out: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        rows = [{"value": value, "count": int(count)} for value, count in counts[field].most_common(limit)]
        rows.sort(key=lambda r: (-r["count"], str(r["value"])))
        out[field] = rows
    return out
