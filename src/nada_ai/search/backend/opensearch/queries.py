from __future__ import annotations

from typing import Any, Literal

from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD, TEXT_FIELD
from nada_ai.settings import Settings

SearchMode = Literal["keyword", "vector", "hybrid"]


def _neural_query_body(
    settings: Settings,
    query_text: str,
    knn_k: int,
    filter_clauses: list[dict[str, Any]],
) -> dict[str, Any]:
    """``neural`` query: embedding is computed in-cluster from ``query_text`` via ML Commons."""
    mid = settings.opensearch_ml_model_id
    if not mid:
        raise ValueError("opensearch_ml_model_id required for OpenSearch ML neural search")
    body: dict[str, Any] = {
        "field": EMBEDDING_FIELD,
        "query_text": query_text,
        "model_id": mid,
        "k": knn_k,
    }
    if filter_clauses:
        body["filter"] = {"bool": {"filter": filter_clauses}}
    return body


def _knn_field_body(
    query_vector: list[float],
    knn_k: int,
    filter_clauses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Body for ``knn.<field>``: vector + k, optionally OpenSearch efficient ``filter``."""
    body: dict[str, Any] = {
        "vector": query_vector,
        "k": knn_k,
    }
    if filter_clauses:
        body["filter"] = {"bool": {"filter": filter_clauses}}
    return body


def build_filters(filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not filters:
        return []
    clauses: list[dict[str, Any]] = []
    if t := filters.get("type"):
        clauses.append({"term": {"type": t}})
    if idno := filters.get("idno"):
        clauses.append({"term": {"idno": idno}})
    if idnos := filters.get("idnos"):
        clauses.append({"terms": {"idno": idnos}})
    if g := filters.get("geographies"):
        clauses.append({"terms": {"geographies": g}})
    if s := filters.get("source"):
        if isinstance(s, list):
            clauses.append({"terms": {"source": s}})
        else:
            clauses.append({"term": {"source": s}})
    if p := filters.get("periodicity"):
        clauses.append({"term": {"periodicity": p}})
    if dt := filters.get("document_type"):
        clauses.append({"term": {"document_type": dt}})
    if authors := filters.get("authors"):
        clauses.append({"terms": {"authors": authors}})
    if filters.get("year_start") is not None or filters.get("year_end") is not None:
        yr: dict[str, int] = {}
        if filters.get("year_start") is not None:
            yr["gte"] = int(filters["year_start"])
        if filters.get("year_end") is not None:
            yr["lte"] = int(filters["year_end"])
        clauses.append({"range": {"year_start": yr}})
    return clauses


def build_search_query(
    settings: Settings,
    query_text: str,
    mode: SearchMode,
    query_vector: list[float] | None,
    filters: dict[str, Any] | None,
    size: int = 10,
    from_: int = 0,
    knn_k: int = 50,
    collapse_field: str | None = None,
    collapse_inner_hits: dict[str, Any] | None = None,
    include_embedding: bool = True,
) -> dict[str, Any]:
    filter_clauses = build_filters(filters)
    filter_block = {"bool": {"filter": filter_clauses}} if filter_clauses else None

    kw_boost = float(settings.hybrid_keyword_boost)
    use_ml = settings.embedding_backend == "opensearch_ml"

    if mode == "keyword":
        inner: dict[str, Any] = {
            "multi_match": {
                "query": query_text,
                "fields": [TEXT_FIELD],
                "type": "best_fields",
            }
        }
        q = _wrap_filter(inner, filter_block)
        return _search_body(
            q,
            size,
            from_,
            collapse_field=collapse_field,
            collapse_inner_hits=collapse_inner_hits,
            include_embedding=include_embedding,
        )

    if mode == "vector":
        if use_ml:
            inner = {"neural": _neural_query_body(settings, query_text, knn_k, filter_clauses)}
            return _search_body(
                inner,
                size,
                from_,
                collapse_field=collapse_field,
                collapse_inner_hits=collapse_inner_hits,
                include_embedding=include_embedding,
            )
        if not query_vector:
            raise ValueError("query_vector required for vector mode (local embedding backend)")
        # Efficient k-NN: put filters inside the knn clause so ``k`` neighbors are taken
        # from the filtered set (bool.filter + knn would post-filter global top-k → empty hits).
        inner = {
            "knn": {
                EMBEDDING_FIELD: _knn_field_body(query_vector, knn_k, filter_clauses),
            }
        }
        return _search_body(
            inner,
            size,
            from_,
            collapse_field=collapse_field,
            collapse_inner_hits=collapse_inner_hits,
            include_embedding=include_embedding,
        )

    # hybrid
    if use_ml:
        mm_ml: dict[str, Any] = {
            "multi_match": {
                "query": query_text,
                "fields": [TEXT_FIELD],
                "boost": kw_boost,
                "type": "best_fields",
            }
        }
        if filter_clauses:
            keyword_query_ml: dict[str, Any] = {"bool": {"must": [mm_ml], "filter": filter_clauses}}
        else:
            keyword_query_ml = mm_ml
        neural_q: dict[str, Any] = {"neural": _neural_query_body(settings, query_text, knn_k, filter_clauses)}
        inner_ml = {"bool": {"should": [keyword_query_ml, neural_q], "minimum_should_match": 1}}
        return _search_body(
            inner_ml,
            size,
            from_,
            collapse_field=collapse_field,
            collapse_inner_hits=collapse_inner_hits,
            include_embedding=include_embedding,
        )

    if not query_vector:
        raise ValueError("query_vector required for hybrid mode (local embedding backend)")
    # Weights: lexical boost on multi_match; knn scores are on native scale — tune `hybrid_*_boost` via query string if needed.
    mm = {
        "multi_match": {
            "query": query_text,
            "fields": [TEXT_FIELD],
            "boost": kw_boost,
            "type": "best_fields",
        }
    }
    if filter_clauses:
        keyword_query: dict[str, Any] = {"bool": {"must": [mm], "filter": filter_clauses}}
    else:
        keyword_query = mm
    knn_query: dict[str, Any] = {
        "knn": {EMBEDDING_FIELD: _knn_field_body(query_vector, knn_k, filter_clauses)},
    }
    should = [keyword_query, knn_query]
    inner = {"bool": {"should": should, "minimum_should_match": 1}}
    q = inner
    return _search_body(
        q,
        size,
        from_,
        collapse_field=collapse_field,
        collapse_inner_hits=collapse_inner_hits,
        include_embedding=include_embedding,
    )


def _wrap_filter(inner: dict[str, Any], filter_block: dict[str, Any] | None) -> dict[str, Any]:
    if not filter_block:
        return inner
    return {"bool": {"must": [inner], "filter": filter_block["bool"]["filter"]}}


def _search_body(
    query: dict[str, Any],
    size: int,
    from_: int,
    collapse_field: str | None = None,
    collapse_inner_hits: dict[str, Any] | None = None,
    include_embedding: bool = True,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": query,
        "size": size,
        "from": from_,
        "track_total_hits": True,
    }
    if not include_embedding:
        body["_source"] = {"excludes": [EMBEDDING_FIELD]}
    if collapse_field:
        collapse: dict[str, Any] = {"field": collapse_field}
        if collapse_inner_hits:
            inner: dict[str, Any] = {
                "name": collapse_inner_hits["name"],
                "size": collapse_inner_hits["size"],
            }
            if not include_embedding:
                inner["_source"] = {"excludes": [EMBEDDING_FIELD]}
            collapse["inner_hits"] = inner
        body["collapse"] = collapse
    return body
