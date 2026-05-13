"""Recommendation vector fusion (shared with backends)."""

import math

from nada_ai.search.vector_fusion import fuse_chunk_embeddings


def test_fuse_mean_normalizes():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    v = fuse_chunk_embeddings([a, b], "mean")
    assert len(v) == 3
    n = math.sqrt(sum(x * x for x in v))
    assert abs(n - 1.0) < 1e-5


def test_fuse_max_pool():
    rows = [[1.0, 2.0], [3.0, 0.0]]
    v = fuse_chunk_embeddings(rows, "max_pool")
    assert abs(v[0] - 3.0 / math.sqrt(13)) < 1e-5
