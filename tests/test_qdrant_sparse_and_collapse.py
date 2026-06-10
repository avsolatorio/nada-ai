"""Unit tests for Qdrant RRF helpers and collapse payload keys (no live Qdrant)."""

from __future__ import annotations

from nada_ai.search.backend.qdrant.search_backend import (
    _collapse_key_from_payload,
    _rrf_merge,
    _rrf_merge_with_scores,
)


def test_rrf_merge_with_scores_orders_by_sum():
    ordered, scores = _rrf_merge_with_scores([[10, 20], [20, 30]], k=60, limit=10)
    assert scores[20] == scores[20]
    assert ordered[0] == 20
    assert _rrf_merge([[10, 20], [20, 30]], limit=10) == ordered


def test_collapse_key_metadata_nested():
    pl = {"metadata": {"idno": "ABC", "type": "indicator"}}
    assert _collapse_key_from_payload(pl, "idno") == "ABC"


def test_collapse_key_root_field():
    assert _collapse_key_from_payload({"page_content": "hello"}, "page_content") == "hello"


def test_collapse_key_missing_returns_none():
    assert _collapse_key_from_payload({"metadata": {}}, "idno") is None
