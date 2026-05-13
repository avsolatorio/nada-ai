"""Fuse multiple chunk embeddings into one query vector (recommendations)."""

from __future__ import annotations

import numpy as np


def fuse_chunk_embeddings(rows: list[list[float]], strategy: str) -> list[float]:
    """Element-wise mean or max-pool over ``rows``, then L2-normalize (empty input → [])."""
    if not rows:
        return []
    mat = np.array(rows, dtype=np.float64)
    if strategy == "max_pool":
        v = np.max(mat, axis=0)
    else:
        v = np.mean(mat, axis=0)
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return v.astype(float).tolist()
