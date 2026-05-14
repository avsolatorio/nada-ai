"""FastEmbed BM25 sparse vectors for Qdrant lexical search (optional ``qdrant_sparse_lexical``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from qdrant_client.http import models as qm


def _to_sparse_vector(emb: Any) -> qm.SparseVector:
    idx = [int(i) for i in emb.indices]
    val = [float(v) for v in emb.values]
    return qm.SparseVector(indices=idx, values=val)


@lru_cache(maxsize=4)
def _model(model_id: str) -> object:
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=model_id)


def embed_documents_sparse(texts: list[str], *, model_id: str, batch_size: int = 256) -> list[qm.SparseVector]:
    """Encode passages for upsert (BM25 document side)."""
    if not texts:
        return []
    model = _model(model_id)
    out: list[qm.SparseVector] = []
    for emb in model.embed(texts, batch_size=min(batch_size, max(len(texts), 1))):
        out.append(_to_sparse_vector(emb))
    if len(out) != len(texts):
        raise RuntimeError(f"sparse embed count mismatch: expected {len(texts)}, got {len(out)}")
    return out


def embed_query_sparse(text: str, *, model_id: str) -> qm.SparseVector:
    """Encode a single search query (BM25 query side)."""
    if not (text or "").strip():
        return qm.SparseVector(indices=[], values=[])
    model = _model(model_id)
    emb = next(model.query_embed([text]))
    return _to_sparse_vector(emb)
