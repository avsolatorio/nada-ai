"""Qdrant implementation of :class:`nada_ai.search.ports.SearchBackendPort`."""

from __future__ import annotations

import asyncio
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD, TEXT_FIELD
from nada_ai.search.backend.opensearch.queries import FACET_FIELD_WHITELIST
from nada_ai.search.backend.qdrant.filters import _filter_must_conditions, filters_to_qdrant_filter
from nada_ai.search.canonical import stored_filter_field_name
from nada_ai.search.explain_filters import compute_filter_match
from nada_ai.search.ports import RecommendParams, SearchOutcome, SearchParams
from nada_ai.search.vector_fusion import fuse_chunk_embeddings
from nada_ai.settings import Settings


def _sanitize_facet_fields(requested: list[str] | None) -> list[str]:
    if not requested:
        return sorted(FACET_FIELD_WHITELIST)
    return [f for f in requested if f in FACET_FIELD_WHITELIST]


def _merge_must_filters(base: qm.Filter | None, extra_must: list[qm.Condition]) -> qm.Filter | None:
    must: list[qm.Condition] = []
    if base and base.must:
        must.extend(base.must)
    must.extend(extra_must)
    if not must:
        return None
    return qm.Filter(must=must)


def _record_to_hit(
    rec: qm.Record,
    *,
    score: float | None = None,
    include_embedding: bool,
) -> dict[str, Any]:
    payload = dict(rec.payload or {})
    if not include_embedding:
        payload.pop(EMBEDDING_FIELD, None)
    pid = rec.id
    sid = str(pid) if not isinstance(pid, str) else pid
    return {"_id": sid, "_score": score, "_source": payload}


def _scored_point_to_hit(sp: qm.ScoredPoint, *, include_embedding: bool) -> dict[str, Any]:
    payload = dict(sp.payload or {})
    if not include_embedding:
        payload.pop(EMBEDDING_FIELD, None)
    pid = sp.id
    sid = str(pid) if not isinstance(pid, str) else pid
    return {"_id": sid, "_score": sp.score, "_source": payload}


def _rrf_merge(id_lists: list[list[Any]], *, k: int = 60, limit: int) -> list[Any]:
    scores: dict[Any, float] = {}
    for ids in id_lists:
        for rank, pid in enumerate(ids, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.keys(), key=lambda p: -scores[p])
    return ordered[:limit]


class QdrantSearchBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )

    async def aclose(self) -> None:
        await self._client.close()

    def _collection(self) -> str:
        return self._settings.qdrant_collection

    async def health(self) -> dict[str, Any]:
        collections = await self._client.get_collections()
        names = {c.name for c in collections.collections}
        ok = self._collection() in names
        return {
            "status": "ok" if ok else "degraded",
            "backend": "qdrant",
            "collection": self._collection(),
            "collection_exists": ok,
        }

    async def _facet_counts(
        self,
        *,
        facet_fields: list[str],
        facet_filter: qm.Filter | None,
    ) -> dict[str, list[dict[str, Any]]]:
        async def one(field: str) -> tuple[str, list[dict[str, Any]]]:
            resp = await self._client.facet(
                collection_name=self._collection(),
                key=stored_filter_field_name(field),
                facet_filter=facet_filter,
                limit=200,
            )
            rows = [{"value": h.value, "count": int(h.count)} for h in (resp.hits or [])]
            if field == "geographies":
                rows.sort(key=lambda r: str(r["value"]).lower())
            else:
                rows.sort(key=lambda r: (-r["count"], str(r["value"])))
            return field, rows

        pairs = await asyncio.gather(*[one(f) for f in facet_fields])
        return dict(pairs)

    def _match_text_filter(self, query_text: str, base_filter: qm.Filter | None) -> qm.Filter:
        text_cond = qm.FieldCondition(key=TEXT_FIELD, match=qm.MatchText(text=query_text))
        merged = _merge_must_filters(base_filter, [text_cond])
        assert merged is not None
        return merged

    async def _count_points(self, coll: str, count_filter: qm.Filter | None) -> int:
        resp = await self._client.count(collection_name=coll, count_filter=count_filter)
        return int(resp.count)

    async def _count_vector_neighbors_above_threshold(
        self,
        coll: str,
        query_vector: list[float],
        base_fl: qm.Filter | None,
        score_threshold: float,
    ) -> tuple[int, bool]:
        """Count points with dense similarity >= ``score_threshold`` (paginated; may hit cap)."""
        cap = self._settings.qdrant_vector_count_scan_cap
        q = qm.NearestQuery(nearest=query_vector)
        total = 0
        page_size = 256
        while True:
            if total >= cap:
                return cap, True
            chunk_limit = min(page_size, cap - total)
            resp = await self._client.query_points(
                collection_name=coll,
                query=q,
                query_filter=base_fl,
                score_threshold=score_threshold,
                limit=chunk_limit,
                offset=total,
                with_payload=False,
                with_vectors=False,
            )
            pts = list(resp.points or [])
            if not pts:
                return total, False
            total += len(pts)
            if len(pts) < chunk_limit:
                return total, False

    async def _vector_total_for_response(
        self,
        coll: str,
        query_vector: list[float],
        base_fl: qm.Filter | None,
        score_threshold: float | None,
    ) -> tuple[int, str, bool]:
        """Return (total, total_basis, capped_similarity_count)."""
        if score_threshold is None:
            n = await self._count_points(coll, base_fl)
            return n, "metadata_filters_only", False
        n, capped = await self._count_vector_neighbors_above_threshold(coll, query_vector, base_fl, score_threshold)
        basis = "vector_similarity_above_threshold_capped" if capped else "vector_similarity_above_threshold"
        return n, basis, capped

    async def _keyword_scroll_hits(
        self,
        *,
        query_text: str,
        base_filter: qm.Filter | None,
        need: int,
        include_embedding: bool,
        scroll_filter: qm.Filter | None = None,
    ) -> list[qm.Record]:
        """Payload ``MatchText`` on ``page_content`` (requires a text payload index on ingest)."""
        flt = scroll_filter or self._match_text_filter(query_text, base_filter)
        out: list[qm.Record] = []
        offset: Any = None
        while len(out) < need:
            batch, offset = await self._client.scroll(
                collection_name=self._collection(),
                scroll_filter=flt,
                limit=min(256, max(need - len(out), 1)),
                offset=offset,
                with_payload=True,
                with_vectors=include_embedding,
            )
            if not batch:
                break
            out.extend(batch)
        return out[:need]

    async def _keyword_scroll_ids(
        self,
        *,
        query_text: str,
        base_filter: qm.Filter | None,
        limit: int,
        scroll_filter: qm.Filter | None = None,
    ) -> list[Any]:
        flt = scroll_filter or self._match_text_filter(query_text, base_filter)
        ids: list[Any] = []
        offset: Any = None
        while len(ids) < limit:
            batch, offset = await self._client.scroll(
                collection_name=self._collection(),
                scroll_filter=flt,
                limit=min(128, max(limit - len(ids), 1)),
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not batch:
                break
            ids.extend(rec.id for rec in batch)
        return ids[:limit]

    async def search(self, params: SearchParams) -> SearchOutcome:
        facet_fields = _sanitize_facet_fields(params.facet_fields) if params.include_facets else None
        coll = self._collection()
        base_fl = filters_to_qdrant_filter(params.filters)
        debug: dict[str, Any] = {"backend": "qdrant", "collection": coll, "mode": params.mode}

        if params.use_idno_fast_path:
            merged = dict(params.filters or {})
            merged["idno"] = params.query.strip()
            fl = filters_to_qdrant_filter(merged)
            count_resp = await self._client.count(collection_name=coll, count_filter=fl)
            total = int(count_resp.count)
            need = min(params.from_ + params.size, 10_000)
            batch, _ = await self._client.scroll(
                collection_name=coll,
                scroll_filter=fl,
                limit=need,
                offset=None,
                with_payload=True,
                with_vectors=params.include_embedding,
            )
            page = batch[params.from_ : params.from_ + params.size]
            hits = [_record_to_hit(r, score=1.0, include_embedding=params.include_embedding) for r in page]
            facets = await self._facet_for_search(facet_fields, params.mode, params.query, params.query_vector, base_fl, fl)
            debug.update({"kind": "idno_fast"})
            return SearchOutcome(total=total, hits=hits, facets=facets, debug_request=debug)

        if params.mode == "keyword":
            need = params.from_ + params.size
            kw_flt = self._match_text_filter(params.query, base_fl)
            count_resp, recs = await asyncio.gather(
                self._client.count(collection_name=coll, count_filter=kw_flt),
                self._keyword_scroll_hits(
                    query_text=params.query,
                    base_filter=base_fl,
                    need=need,
                    include_embedding=params.include_embedding,
                    scroll_filter=kw_flt,
                ),
            )
            total = int(count_resp.count)
            page = recs[params.from_ : params.from_ + params.size]
            hits = [_record_to_hit(r, score=1.0, include_embedding=params.include_embedding) for r in page]
            facet_fl = self._facet_filter_for_mode("keyword", params.query, base_fl)
            facets = await self._facet_for_search(facet_fields, "keyword", params.query, None, base_fl, facet_fl)
            debug.update({"kind": "keyword_match_text", "need": need, "total_basis": "match_text_and_filters"})
            return SearchOutcome(total=total, hits=hits, facets=facets, debug_request=debug)

        if not params.query_vector:
            raise ValueError("query_vector required for vector/hybrid modes with Qdrant (local embedding backend).")

        if params.mode == "vector":
            return await self._vector_search(
                params,
                facet_fields,
                base_fl,
                debug,
                collapse=bool(params.collapse_field),
            )

        return await self._hybrid_search(params, facet_fields, base_fl, debug)

    def _facet_filter_for_mode(self, mode: str, query_text: str, base_fl: qm.Filter | None) -> qm.Filter | None:
        if mode == "keyword":
            return self._match_text_filter(query_text, base_fl)
        return base_fl

    async def _facet_for_search(
        self,
        facet_fields: list[str] | None,
        mode: str,
        query_text: str,
        query_vector: list[float] | None,
        base_fl: qm.Filter | None,
        facet_context_fl: qm.Filter | None,
    ) -> dict[str, list[dict[str, Any]]] | None:
        if not facet_fields:
            return None
        fl = facet_context_fl if facet_context_fl is not None else self._facet_filter_for_mode(mode, query_text, base_fl)
        return await self._facet_counts(facet_fields=facet_fields, facet_filter=fl)

    async def _vector_search(
        self,
        params: SearchParams,
        facet_fields: list[str] | None,
        base_fl: qm.Filter | None,
        debug: dict[str, Any],
        *,
        collapse: bool,
    ) -> SearchOutcome:
        coll = self._collection()
        q = qm.NearestQuery(nearest=params.query_vector)
        thr = params.vector_score_threshold
        if params.collapse_field and collapse:
            inner = params.collapse_inner_hits or {"name": "variants", "size": 10}
            gsize = 1 + int(inner.get("size", 10))
            group_limit = params.from_ + params.size
            (total, total_basis, capped_sim), groups_resp = await asyncio.gather(
                self._vector_total_for_response(coll, params.query_vector, base_fl, thr),
                self._client.query_points_groups(
                    collection_name=coll,
                    group_by=stored_filter_field_name(params.collapse_field),
                    query=q,
                    query_filter=base_fl,
                    limit=group_limit,
                    group_size=gsize,
                    with_payload=True,
                    with_vectors=params.include_embedding,
                    score_threshold=thr,
                ),
            )
            groups = groups_resp
            hits_out: list[dict[str, Any]] = []
            for g in groups.groups or []:
                pts = list(g.hits or [])
                if not pts:
                    continue
                top = pts[0]
                hit = _scored_point_to_hit(top, include_embedding=params.include_embedding)
                rest = pts[1:]
                if rest:
                    name = inner.get("name", "variants")
                    hit["inner_hits"] = {
                        str(name): {
                            "hits": {
                                "hits": [
                                    _scored_point_to_hit(p, include_embedding=params.include_embedding) for p in rest
                                ]
                            }
                        }
                    }
                hits_out.append(hit)
            hits_out = hits_out[params.from_ : params.from_ + params.size]
            facets = await self._facet_for_search(facet_fields, "vector", params.query, params.query_vector, base_fl, base_fl)
            debug.update(
                {
                    "kind": "vector_groups",
                    "group_by": params.collapse_field,
                    "total_basis": total_basis,
                    "vector_score_threshold": thr,
                    "similarity_count_cap_hit": capped_sim,
                }
            )
            return SearchOutcome(total=total, hits=hits_out, facets=facets, debug_request=debug)

        off = params.from_
        (total, total_basis, capped_sim), resp = await asyncio.gather(
            self._vector_total_for_response(coll, params.query_vector, base_fl, thr),
            self._client.query_points(
                collection_name=coll,
                query=q,
                query_filter=base_fl,
                limit=params.size,
                offset=off,
                with_payload=True,
                with_vectors=params.include_embedding,
                score_threshold=thr,
            ),
        )
        hits_out = [_scored_point_to_hit(p, include_embedding=params.include_embedding) for p in (resp.points or [])]
        facets = await self._facet_for_search(facet_fields, "vector", params.query, params.query_vector, base_fl, base_fl)
        debug.update(
            {
                "kind": "vector",
                "offset": off,
                "total_basis": total_basis,
                "vector_score_threshold": thr,
                "similarity_count_cap_hit": capped_sim,
            }
        )
        return SearchOutcome(total=total, hits=hits_out, facets=facets, debug_request=debug)

    async def _hybrid_search(
        self,
        params: SearchParams,
        facet_fields: list[str] | None,
        base_fl: qm.Filter | None,
        debug: dict[str, Any],
    ) -> SearchOutcome:
        coll = self._collection()
        prefetch_k = max(params.knn_k, params.size + params.from_, 50)
        kw_flt = self._match_text_filter(params.query, base_fl)
        thr = params.vector_score_threshold
        (total, total_basis, capped_sim), vec_resp, text_ids = await asyncio.gather(
            self._vector_total_for_response(coll, params.query_vector, base_fl, thr),
            self._client.query_points(
                collection_name=coll,
                query=qm.NearestQuery(nearest=params.query_vector),
                query_filter=base_fl,
                limit=prefetch_k,
                offset=0,
                with_payload=False,
                with_vectors=False,
                score_threshold=thr,
            ),
            self._keyword_scroll_ids(
                query_text=params.query,
                base_filter=base_fl,
                limit=prefetch_k,
                scroll_filter=kw_flt,
            ),
        )
        vec_ids = [p.id for p in (vec_resp.points or [])]
        merged_ids = _rrf_merge([vec_ids, text_ids], limit=prefetch_k)
        window = merged_ids[params.from_ : params.from_ + params.size]
        if not window:
            facets = await self._facet_for_search(facet_fields, "hybrid", params.query, params.query_vector, base_fl, base_fl)
            debug.update(
                {
                    "kind": "hybrid_rrf",
                    "prefetch_k": prefetch_k,
                    "merged": 0,
                    "total_basis": total_basis,
                    "vector_score_threshold": thr,
                    "similarity_count_cap_hit": capped_sim,
                }
            )
            return SearchOutcome(total=total, hits=[], facets=facets, debug_request=debug)

        recs = await self._client.retrieve(
            collection_name=coll,
            ids=window,
            with_payload=True,
            with_vectors=params.include_embedding,
        )
        by_id = {r.id: r for r in recs}
        hits_out: list[dict[str, Any]] = []
        for pid in window:
            rec = by_id.get(pid)
            if rec is None:
                continue
            hits_out.append(_record_to_hit(rec, score=1.0, include_embedding=params.include_embedding))
        facets = await self._facet_for_search(facet_fields, "hybrid", params.query, params.query_vector, base_fl, base_fl)
        debug.update(
            {
                "kind": "hybrid_rrf",
                "prefetch_k": prefetch_k,
                "total_basis": total_basis,
                "vector_score_threshold": thr,
                "similarity_count_cap_hit": capped_sim,
            }
        )
        return SearchOutcome(total=total, hits=hits_out, facets=facets, debug_request=debug)

    async def recommend_by_idno(self, params: RecommendParams) -> SearchOutcome:
        seed = params.idno.strip()
        fl = filters_to_qdrant_filter({"idno": seed})
        recs, _ = await self._client.scroll(
            collection_name=self._collection(),
            scroll_filter=fl,
            limit=500,
            with_payload=True,
            with_vectors=True,
        )
        vecs: list[list[float]] = []
        for r in recs:
            vec = r.vector
            if isinstance(vec, dict):
                vec = vec.get("") or next(iter(vec.values()), None)
            if isinstance(vec, list) and vec:
                vecs.append([float(x) for x in vec])
        if not vecs:
            raise ValueError(f"No stored embeddings found for idno `{seed}`; cannot build recommendation vector.")
        qv = fuse_chunk_embeddings(vecs, params.vector_strategy)

        must = list(_filter_must_conditions(params.filters))
        must_not: list[qm.FieldCondition] | None = None
        if params.exclude_idno:
            must_not = [
                qm.FieldCondition(key=stored_filter_field_name("idno"), match=qm.MatchValue(value=seed))
            ]
        rec_fl = qm.Filter(must=must or None, must_not=must_not)

        q = qm.NearestQuery(nearest=qv)
        facet_fields = _sanitize_facet_fields(params.facet_fields) if params.include_facets else None
        thr = params.vector_score_threshold
        coll = self._collection()
        if thr is None:
            resp = await self._client.query_points(
                collection_name=coll,
                query=q,
                query_filter=rec_fl,
                limit=params.size,
                offset=0,
                with_payload=True,
                with_vectors=False,
            )
            total = None
        else:
            (total, tb, capped_sim), resp = await asyncio.gather(
                self._vector_total_for_response(coll, qv, rec_fl, thr),
                self._client.query_points(
                    collection_name=coll,
                    query=q,
                    query_filter=rec_fl,
                    limit=params.size,
                    offset=0,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=thr,
                ),
            )
        hits_out = [_scored_point_to_hit(p, include_embedding=False) for p in (resp.points or [])]
        facets = await self._facet_counts(facet_fields=facet_fields, facet_filter=rec_fl) if facet_fields else None
        debug: dict[str, Any] = {
            "backend": "qdrant",
            "kind": "recommend_by_idno",
            "seed": seed,
            "vector_score_threshold": thr,
        }
        if thr is not None:
            debug["total_basis"] = tb
            debug["similarity_count_cap_hit"] = capped_sim
        return SearchOutcome(total=total, hits=hits_out, facets=facets, debug_request=debug)

    async def explain_by_idno(self, idno: str, filters: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(filters or {})
        merged["idno"] = idno.strip()
        fl = filters_to_qdrant_filter(merged)
        batch, _ = await self._client.scroll(
            collection_name=self._collection(),
            scroll_filter=fl,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            return {
                "idno": idno.strip(),
                "found": False,
                "filters_applied": merged,
                "sample_source": None,
                "filter_match": None,
            }
        payload = dict(batch[0].payload or {})
        meta = payload.get("metadata") or {}
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
        filter_only = {k: v for k, v in merged.items() if k != "idno"} or None
        fm = compute_filter_match(dict(sample), filter_only) if filter_only else {"all_matched": True, "per_field": {}}
        return {
            "idno": idno.strip(),
            "found": True,
            "filters_applied": merged,
            "sample_source": sample,
            "filter_match": fm,
        }
