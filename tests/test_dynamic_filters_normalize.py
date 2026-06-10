"""Tests for external filter normalization."""

from nada_ai.search.dynamic_filters import normalize_external_filters, split_filters, unwrap_external_filters

SAMPLE = {
    "filters": {
        "doctype": 1,
        "published": 1,
        "dataset_type": "document",
        "formid": None,
        "form_model": None,
        "year_start": 2025,
        "year_end": 2025,
        "years": [2025],
        "repositoryid": "central",
        "repositories": ["central"],
        "countries": [181],
        "regions": [7, 9],
        "data_class_id": None,
        "tags": [],
    }
}


def test_normalize_sample_external_filters():
    out = normalize_external_filters(SAMPLE)
    by_key = {row["key"]: row["value"] for row in out}
    assert by_key["doctype"] == ["1"]
    assert by_key["countries"] == ["181"]
    assert by_key["regions"] == ["7", "9"]
    assert by_key["years"] == ["2025"]
    assert "formid" not in by_key
    assert "tags" not in by_key


def test_unwrap_external_filters():
    assert unwrap_external_filters(SAMPLE) == SAMPLE["filters"]
    assert unwrap_external_filters({"a": 1}) == {"a": 1}


def test_split_filters_fixed_vs_dynamic():
    fixed, dynamic = split_filters(
        {
            "type": "document",
            "countries": [181],
            "doctype": 1,
        }
    )
    assert fixed == {"type": "document"}
    assert dynamic == {"countries": [181], "doctype": 1}
