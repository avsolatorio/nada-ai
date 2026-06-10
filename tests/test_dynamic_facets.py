"""Tests for dynamic facet aggregation helpers."""

from nada_ai.search.dynamic_filters import (
    aggregate_dynamic_facet_rows,
    aggregate_dynamic_facet_rows_multi,
    dynamic_facet_aggs,
    normalized_to_facets_map,
    resolve_facet_fields,
    unwrap_dynamic_facet_buckets,
)


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


def test_aggregate_dynamic_facet_rows_scopes_by_key():
    payloads = [
        {
            "metadata": {
                "filter_facets": {
                    "doctype": ["1"],
                    "years": ["2015"],
                    "dataset_type": ["document"],
                }
            }
        },
        {
            "metadata": {
                "filter_facets": {
                    "doctype": ["2"],
                    "years": ["2014"],
                }
            }
        },
    ]
    rows = aggregate_dynamic_facet_rows(payloads, "doctype")
    assert rows == [{"value": "1", "count": 1}, {"value": "2", "count": 1}]


def test_aggregate_dynamic_facet_rows_multi_one_pass():
    payloads = [
        {
            "metadata": {
                "filter_facets": {
                    "doctype": ["1"],
                    "years": ["2015"],
                }
            }
        }
    ]
    out = aggregate_dynamic_facet_rows_multi(payloads, ["doctype", "years"])
    assert out["doctype"] == [{"value": "1", "count": 1}]
    assert out["years"] == [{"value": "2015", "count": 1}]


def test_normalized_to_facets_map():
    normalized = [{"key": "countries", "value": ["181"]}, {"key": "years", "value": ["2015", "2016"]}]
    assert normalized_to_facets_map(normalized) == {"countries": ["181"], "years": ["2015", "2016"]}


def test_dynamic_facet_qdrant_key():
    from nada_ai.search.dynamic_filters import dynamic_facet_qdrant_key

    assert dynamic_facet_qdrant_key("doctype") == "metadata.filter_facets.doctype"
