"""Dynamic filter_fields normalization, query building, and facet helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qdrant_client.http import models as qm

from nada_ai.search.backend.opensearch.mapping import metadata_field
from nada_ai.settings import Settings

FILTER_FIELDS_KEY = "filter_fields"
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


def load_dynamic_facet_keys(settings: Settings | None = None) -> frozenset[str]:
    """Load facetable dynamic filter keys from config (env override optional)."""
    path: Path | None = None
    if settings is not None and settings.dynamic_filter_facets_path:
        path = Path(settings.dynamic_filter_facets_path)
    elif _DEFAULT_FACETS_PATH.is_file():
        path = _DEFAULT_FACETS_PATH
    if path is None or not path.is_file():
        return _DEFAULT_FACETABLE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = data.get("facetable") or []
        return frozenset(str(k) for k in keys)
    except (OSError, json.JSONDecodeError, TypeError):
        return _DEFAULT_FACETABLE


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
            qm.NestedCondition(
                nested=qm.Nested(
                    key=FILTER_FIELDS_PATH,
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(key="key", match=qm.MatchValue(value=str(key))),
                            qm.FieldCondition(key="value", match=value_cond),
                        ]
                    ),
                )
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


def _filter_fields_map(sample: dict[str, Any]) -> dict[str, set[str]]:
    """Build key -> stored values from ``metadata.filter_fields`` on a hit sample."""
    meta = sample.get("metadata") if "metadata" in sample else sample
    if not isinstance(meta, dict):
        return {}
    rows = meta.get(FILTER_FIELDS_KEY) or []
    out: dict[str, set[str]] = {}
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


def match_dynamic_filters(sample: dict[str, Any], dynamic: dict[str, Any]) -> dict[str, Any]:
    """Evaluate dynamic filters against ``sample`` (metadata dict or full _source)."""
    if not dynamic:
        return {"all_matched": True, "per_field": {}}
    stored = _filter_fields_map(sample)
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
