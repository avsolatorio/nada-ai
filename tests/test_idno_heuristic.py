"""Idno-shaped query heuristic (fast path trigger)."""

from nada_ai.search.query_heuristics import looks_like_catalog_idno


def test_idno_like_compact_token():
    assert looks_like_catalog_idno("WB.1234") is True
    assert looks_like_catalog_idno("a b") is False
    assert looks_like_catalog_idno("ab") is False
