"""Tests for dynamic facet aggregation helpers."""

from nada_ai.search.dynamic_filters import dynamic_facet_aggs, resolve_facet_fields, unwrap_dynamic_facet_buckets


def test_resolve_facet_fields_defaults():
    static, dynamic = resolve_facet_fields(None)
    assert "type" in static
    assert "countries" in dynamic


def test_unwrap_dynamic_facet_buckets():
    agg = {
        "filtered": {
            "values": {
                "buckets": [
                    {"key": "181", "doc_count": 5},
                    {"key": "7", "doc_count": 2},
                ]
            }
        }
    }
    rows = unwrap_dynamic_facet_buckets("countries", agg)
    assert rows == [{"value": "181", "count": 5}, {"value": "7", "count": 2}]


def test_dynamic_facet_aggs_has_nested_path():
    aggs = dynamic_facet_aggs(["doctype"])
    assert aggs["doctype"]["nested"]["path"] == "metadata.filter_fields"
