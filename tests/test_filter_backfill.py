"""Tests for filter_facets backfill helper."""

from nada_ai.filters.sync import _needs_filter_facets_backfill


def test_needs_backfill_when_facets_missing():
    meta = {
        "filter_fields": [{"key": "countries", "value": ["181"]}],
    }
    assert _needs_filter_facets_backfill(meta) == {"countries": ["181"]}


def test_skips_when_facets_present():
    meta = {
        "filter_fields": [{"key": "countries", "value": ["181"]}],
        "filter_facets": {"countries": ["181"]},
    }
    assert _needs_filter_facets_backfill(meta) is None


def test_skips_when_no_filter_fields():
    assert _needs_filter_facets_backfill({}) is None
