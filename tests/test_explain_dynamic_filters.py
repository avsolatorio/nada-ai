"""Tests for dynamic filter explain matching."""

from nada_ai.search.explain_filters import compute_filter_match


def test_explain_dynamic_filter_match():
    sample = {
        "metadata": {
            "filter_fields": [
                {"key": "countries", "value": ["181"]},
                {"key": "doctype", "value": ["1"]},
            ]
        }
    }
    out = compute_filter_match(sample, {"countries": [181], "doctype": 1})
    assert out["all_matched"] is True
    assert out["per_field"]["dynamic.countries"]["matched"] is True


def test_explain_dynamic_filter_miss():
    sample = {"metadata": {"filter_fields": [{"key": "countries", "value": ["181"]}]}}
    out = compute_filter_match(sample, {"countries": [999]})
    assert out["all_matched"] is False
