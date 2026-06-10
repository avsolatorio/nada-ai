"""OpenSearch implementation of :class:`nada_ai.search.ports.SearchBackendPort`."""

from __future__ import annotations

from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError

from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD, metadata_field
from nada_ai.search.backend.opensearch.queries import (
    build_filters,
    build_idno_fast_query,
    build_search_query,
    merge_facets_into_body,
)
from nada_ai.search.dynamic_filters import resolve_facet_fields, unwrap_dynamic_facet_buckets
from nada_ai.search.explain_filters import compute_filter_match
from nada_ai.search.ports import RecommendParams, SearchOutcome, SearchParams
from nada_ai.search.vector_fusion import fuse_chunk_embeddings
from nada_ai.settings import Settings


def _resolve_facet_fields(settings: Settings, requested: list[str] | None) -> tuple[list[str], list[str]]:
    if not requested:
        return resolve_facet_fields(None, settings)
    return resolve_facet_fields(requested, settings)


def _normalize_facets(
    aggregations: dict[str, Any] | None,
    static_fields: list[str],
    dynamic_fields: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not aggregations:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for field in static_fields:
        agg = aggregations.get(field) or {}
        buckets = agg.get("buckets") or []
        rows = [{"value": b.get("key"), "count": int(b.get("doc_count", 0))} for b in buckets]
        if field == "geographies":
            rows.sort(key=lambda r: str(r["value"]).lower())
        else:
            rows.sort(key=lambda r: (-r["count"], str(r["value"])))
        out[field] = rows
    for field in dynamic_fields:
        agg = aggregations.get(field) or {}
        rows = unwrap_dynamic_facet_buckets(field, agg)
        rows.sort(key=lambda r: (-r["count"], str(r["value"])))
        out[field] = rows
    return out


def _hits_from_response(resp: dict[str, Any]) -> tuple[int | None, list[dict[str, Any]]]:
    total = resp.get("hits", {}).get("total", {})
    if isinstance(total, dict):
        total_val = total.get("value")
    else:
        total_val = total
    hits_out: list[dict[str, Any]] = []
    for h in resp.get("hits", {}).get("hits", []):
        item: dict[str, Any] = {
            "_id": h.get("_id"),
            "_score": h.get("_score"),
            "_source": h.get("_source", {}),
        }
        if h.get("inner_hits"):
            item["inner_hits"] = h["inner_hits"]
        hits_out.append(item)
    return total_val, hits_out


class OpenSearchSearchBackend:
    def __init__(self, client: AsyncOpenSearch, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def health(self) -> dict[str, Any]:
        ok = await self._client.cluster.health()
        return {"status": "ok", "backend": "opensearch", "cluster": ok.get("status"), "index": self._settings.index_name}

    async def search(self, params: SearchParams) -> SearchOutcome:
        static_facets, dynamic_facets = (
            _resolve_facet_fields(self._settings, params.facet_fields) if params.include_facets else ([], [])
        )
        has_facets = bool(static_facets or dynamic_facets)

        if params.use_idno_fast_path:
            q = build_idno_fast_query(
                params.query.strip(),
                params.filters,
                params.size,
                params.from_,
                collapse_field=params.collapse_field,
                collapse_inner_hits=params.collapse_inner_hits,
                include_embedding=params.include_embedding,
                static_facet_fields=static_facets or None,
                dynamic_facet_fields=dynamic_facets or None,
            )
        else:
            q = build_search_query(
                self._settings,
                query_text=params.query,
                mode=params.mode,
                query_vector=params.query_vector,
                filters=params.filters,
                size=params.size,
                from_=params.from_,
                knn_k=params.knn_k,
                collapse_field=params.collapse_field,
                collapse_inner_hits=params.collapse_inner_hits,
                include_embedding=params.include_embedding,
            )
            if has_facets:
                merge_facets_into_body(q, static_facets or None, dynamic_facets or None)

        resp = await self._client.search(index=self._settings.index_name, body=q)
        total_val, hits_out = _hits_from_response(resp)
        facets = (
            _normalize_facets(resp.get("aggregations"), static_facets, dynamic_facets) if has_facets else None
        )
        return SearchOutcome(total=total_val, hits=hits_out, facets=facets or None, debug_request=q)

    async def recommend_by_idno(self, params: RecommendParams) -> SearchOutcome:
        if self._settings.embedding_backend == "opensearch_ml":
            raise ValueError(
                "recommend_by_idno requires stored vectors in _source. With opensearch_ml, embeddings may be "
                "inaccessible from the client; use embedding_backend=local for this API or extend ingest to store "
                "retrievable vectors."
            )

        seed = params.idno.strip()
        fetch_body: dict[str, Any] = {
            "query": {"bool": {"filter": [{"term": {metadata_field("idno"): seed}}]}},
            "size": 500,
            "_source": [EMBEDDING_FIELD, "page_content", "metadata"],
        }
        try:
            seed_resp = await self._client.search(index=self._settings.index_name, body=fetch_body)
        except NotFoundError as e:
            raise ValueError(f"Index `{self._settings.index_name}` not found") from e

        vecs: list[list[float]] = []
        for h in seed_resp.get("hits", {}).get("hits", []):
            src = h.get("_source") or {}
            emb = src.get(EMBEDDING_FIELD)
            if isinstance(emb, list) and emb:
                vecs.append([float(x) for x in emb])
        if not vecs:
            raise ValueError(f"No stored embeddings found for idno `{seed}`; cannot build recommendation vector.")

        query_vector = fuse_chunk_embeddings(vecs, params.vector_strategy)

        filter_clauses = build_filters(params.filters)
        if params.exclude_idno:
            filter_clauses.append({"bool": {"must_not": [{"term": {metadata_field("idno"): seed}}]}})

        knn_inner = {
            "knn": {
                EMBEDDING_FIELD: {
                    "vector": query_vector,
                    "k": params.knn_k,
                    "filter": {"bool": {"filter": filter_clauses}} if filter_clauses else None,
                }
            }
        }
        # Remove None filter key if empty
        knn_body = knn_inner["knn"][EMBEDDING_FIELD]
        if knn_body.get("filter") is None:
            del knn_body["filter"]

        q: dict[str, Any] = {
            "query": knn_inner,
            "size": params.size,
            "from": 0,
            "track_total_hits": True,
            "_source": {"excludes": [EMBEDDING_FIELD]},
        }
        static_facets, dynamic_facets = (
            _resolve_facet_fields(self._settings, params.facet_fields) if params.include_facets else ([], [])
        )
        has_facets = bool(static_facets or dynamic_facets)
        if has_facets:
            merge_facets_into_body(q, static_facets or None, dynamic_facets or None)

        resp = await self._client.search(index=self._settings.index_name, body=q)
        total_val, hits_out = _hits_from_response(resp)
        facets = (
            _normalize_facets(resp.get("aggregations"), static_facets, dynamic_facets) if has_facets else None
        )
        return SearchOutcome(total=total_val, hits=hits_out, facets=facets or None, debug_request=q)

    async def explain_by_idno(self, idno: str, filters: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(filters or {})
        merged["idno"] = idno.strip()
        q = build_idno_fast_query(
            merged["idno"],
            {k: v for k, v in merged.items() if k != "idno"} or None,
            size=1,
            from_=0,
            include_embedding=False,
        )
        resp = await self._client.search(index=self._settings.index_name, body=q)
        _, hits = _hits_from_response(resp)
        if not hits:
            return {
                "idno": idno.strip(),
                "found": False,
                "filters_applied": merged,
                "sample_source": None,
                "filter_match": None,
            }
        src = hits[0].get("_source") or {}
        meta = src.get("metadata") or {}
        keys = (
            "type",
            "qfield",
            "idno",
            "source",
            "geographies",
            "periodicity",
            "document_type",
            "authors",
            "year_start",
            "year_end",
        )
        sample = {k: meta[k] for k in keys if k in meta}
        sample_with_meta = {**sample, "metadata": meta}
        filter_only = {k: v for k, v in merged.items() if k != "idno"} or None
        fm = (
            compute_filter_match(sample_with_meta, filter_only)
            if filter_only
            else {"all_matched": True, "per_field": {}}
        )
        return {
            "idno": idno.strip(),
            "found": True,
            "filters_applied": merged,
            "sample_source": sample,
            "filter_match": fm,
        }

