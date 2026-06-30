"""Unit tests for mcp_server/analytics.py — pure aggregation functions."""

from __future__ import annotations

import pytest

from nada_ai.mcp_server.analytics import (
    _apply_dimension_filter,
    _label_column_for,
    _parse_obs,
    compare,
    get_extremes,
    growth,
    rank,
    summarize,
)
from nada_ai.nada.models import IndicatorSchema, TimeseriesDataRow

# ---------------------------------------------------------------------------
# Helpers to build test data
# ---------------------------------------------------------------------------

DSD_RESULT = {
    "sid": 1,
    "dsd_id": 1,
    "components": [
        {"name": "GEO", "label": "Geography", "data_type": "string",
         "column_type": "geography", "codelist_id": None, "time_period_format": None},
        {"name": "GEO_NAME", "label": "Geography name", "data_type": "string",
         "column_type": "attribute", "codelist_id": None, "time_period_format": None},
        {"name": "PERIOD", "label": "Period", "data_type": "string",
         "column_type": "time_period", "codelist_id": None, "time_period_format": "YYYY"},
        {"name": "VALUE", "label": "Value", "data_type": "double",
         "column_type": "observation_value", "codelist_id": None, "time_period_format": None},
    ],
    "reporting_year_bounds": {"min": 2000, "max": 2022},
}

DSD_WITH_SEX = {
    **DSD_RESULT,
    "components": DSD_RESULT["components"] + [
        {"name": "SEX", "label": "Sex", "data_type": "string",
         "column_type": "dimension", "codelist_id": "99", "time_period_format": None},
    ],
}


def make_schema(dsd=None) -> IndicatorSchema:
    return IndicatorSchema.from_api_result("TEST", dsd or DSD_RESULT)


def make_rows(data: list[dict]) -> list[TimeseriesDataRow]:
    rows = []
    for d in data:
        known = {k: v for k, v in d.items()
                 if k in TimeseriesDataRow.model_fields}
        extra = {k: v for k, v in d.items()
                 if k not in TimeseriesDataRow.model_fields}
        row = TimeseriesDataRow.model_validate({**known, **extra})
        rows.append(row)
    return rows


SAMPLE_ROWS = make_rows([
    {"GEO": "A", "GEO_NAME": "Alpha", "PERIOD": "2020", "VALUE": "10.0"},
    {"GEO": "B", "GEO_NAME": "Beta",  "PERIOD": "2020", "VALUE": "30.0"},
    {"GEO": "C", "GEO_NAME": "Gamma", "PERIOD": "2020", "VALUE": "20.0"},
    {"GEO": "A", "GEO_NAME": "Alpha", "PERIOD": "2021", "VALUE": "12.0"},
    {"GEO": "B", "GEO_NAME": "Beta",  "PERIOD": "2021", "VALUE": "25.0"},
    {"GEO": "C", "GEO_NAME": "Gamma", "PERIOD": "2021", "VALUE": "22.0"},
])

ROWS_WITH_SEX = make_rows([
    {"GEO": "A", "PERIOD": "2020", "VALUE": "10.0", "SEX": "M"},
    {"GEO": "A", "PERIOD": "2020", "VALUE": "12.0", "SEX": "F"},
    {"GEO": "B", "PERIOD": "2020", "VALUE": "30.0", "SEX": "M"},
    {"GEO": "B", "PERIOD": "2020", "VALUE": "35.0", "SEX": "F"},
])


# ---------------------------------------------------------------------------
# _parse_obs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp,expected", [
    ("4.5", 4.5),
    (4.5, 4.5),
    ("0", 0.0),
    (None, None),
    ("", None),
    ("nan", None),
    ("inf", None),
    ("not_a_number", None),
])
def test_parse_obs(inp, expected):
    assert _parse_obs(inp) == expected


# ---------------------------------------------------------------------------
# _apply_dimension_filter
# ---------------------------------------------------------------------------

def test_apply_dimension_filter_empty_passes_all():
    schema = make_schema(DSD_WITH_SEX)
    result = _apply_dimension_filter(ROWS_WITH_SEX, schema, {})
    assert len(result) == 4


def test_apply_dimension_filter_sex_female():
    schema = make_schema(DSD_WITH_SEX)
    result = _apply_dimension_filter(ROWS_WITH_SEX, schema, {"SEX": "F"})
    assert len(result) == 2
    for row in result:
        assert (row.model_extra or {}).get("SEX") == "F"


def test_apply_dimension_filter_no_match():
    schema = make_schema(DSD_WITH_SEX)
    result = _apply_dimension_filter(ROWS_WITH_SEX, schema, {"SEX": "T"})
    assert result == []


# ---------------------------------------------------------------------------
# _label_column_for
# ---------------------------------------------------------------------------

def test_label_column_for_finds_name_column():
    schema = make_schema()
    assert _label_column_for(schema, "GEO") == "GEO_NAME"


def test_label_column_for_no_match():
    schema = make_schema(DSD_WITH_SEX)
    # SEX has no companion label column in our fixture
    assert _label_column_for(schema, "SEX") is None


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------

def test_rank_top_n_default():
    schema = make_schema()
    result = rank(SAMPLE_ROWS, schema, period="2020", n=3)
    assert result.error is None
    assert len(result.rows) == 3
    assert result.rows[0].ref_area == "B"   # highest value 30
    assert result.rows[0].value == 30.0
    assert result.rows[1].ref_area == "C"
    assert result.rows[2].ref_area == "A"
    assert result.rows[0].rank == 1


def test_rank_bottom_n_ascending():
    schema = make_schema()
    result = rank(SAMPLE_ROWS, schema, period="2020", n=2, ascending=True)
    assert result.rows[0].ref_area == "A"   # lowest value 10
    assert result.rows[1].ref_area == "C"


def test_rank_includes_labels():
    schema = make_schema()
    result = rank(SAMPLE_ROWS, schema, period="2020", n=3)
    assert result.rows[0].ref_area_label == "Beta"
    assert result.rows[2].ref_area_label == "Alpha"


def test_rank_reports_total_ref_areas():
    schema = make_schema()
    result = rank(SAMPLE_ROWS, schema, period="2020", n=2)
    assert result.total_ref_areas == 3


def test_rank_unknown_period_returns_empty():
    schema = make_schema()
    result = rank(SAMPLE_ROWS, schema, period="1999")
    assert result.rows == []
    assert result.total_ref_areas == 0


def test_rank_with_dimension_filter():
    schema = make_schema(DSD_WITH_SEX)
    result = rank(ROWS_WITH_SEX, schema, period="2020", dimensions={"SEX": "F"})
    assert len(result.rows) == 2
    assert result.rows[0].ref_area == "B"  # 35 > 12
    assert result.dimensions_applied == {"SEX": "F"}


def test_rank_missing_schema_columns_raises():
    schema = IndicatorSchema(idno="X")  # no columns resolved
    with pytest.raises(ValueError, match="missing required columns"):
        rank(SAMPLE_ROWS, schema, period="2020")


# ---------------------------------------------------------------------------
# get_extremes
# ---------------------------------------------------------------------------

def test_get_extremes_finds_global_max_min():
    schema = make_schema()
    result = get_extremes(SAMPLE_ROWS, schema)
    assert result.error is None
    assert result.maximum is not None
    assert result.minimum is not None
    assert result.maximum.value == 30.0
    assert result.maximum.ref_area == "B"
    assert result.maximum.period == "2020"
    assert result.minimum.value == 10.0
    assert result.minimum.ref_area == "A"
    assert result.minimum.period == "2020"


def test_get_extremes_total_observations():
    schema = make_schema()
    result = get_extremes(SAMPLE_ROWS, schema)
    assert result.total_observations == 6


def test_get_extremes_with_dimension_filter():
    schema = make_schema(DSD_WITH_SEX)
    result = get_extremes(ROWS_WITH_SEX, schema, dimensions={"SEX": "M"})
    assert result.maximum.value == 30.0
    assert result.minimum.value == 10.0


def test_get_extremes_empty_rows():
    schema = make_schema()
    result = get_extremes([], schema)
    assert result.maximum is None
    assert result.minimum is None
    assert result.total_observations == 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare_pivots_by_period():
    schema = make_schema()
    result = compare(SAMPLE_ROWS, schema, ref_areas=["A", "B"])
    assert result.error is None
    periods = [r.period for r in result.rows]
    assert "2020" in periods
    assert "2021" in periods

    row_2020 = next(r for r in result.rows if r.period == "2020")
    assert row_2020.values["A"] == 10.0
    assert row_2020.values["B"] == 30.0


def test_compare_missing_ref_area_is_none():
    schema = make_schema()
    result = compare(SAMPLE_ROWS, schema, ref_areas=["A", "MISSING"])
    row_2020 = next(r for r in result.rows if r.period == "2020")
    assert row_2020.values["A"] == 10.0
    assert row_2020.values["MISSING"] is None


def test_compare_rows_sorted_by_period():
    schema = make_schema()
    result = compare(SAMPLE_ROWS, schema, ref_areas=["A"])
    periods = [r.period for r in result.rows]
    assert periods == sorted(periods)


def test_compare_includes_labels():
    schema = make_schema()
    result = compare(SAMPLE_ROWS, schema, ref_areas=["A", "B"])
    assert result.ref_area_labels["A"] == "Alpha"
    assert result.ref_area_labels["B"] == "Beta"


def test_compare_with_dimension_filter():
    schema = make_schema(DSD_WITH_SEX)
    result = compare(ROWS_WITH_SEX, schema, ref_areas=["A", "B"], dimensions={"SEX": "F"})
    row = result.rows[0]
    assert row.values["A"] == 12.0
    assert row.values["B"] == 35.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_correct_stats():
    schema = make_schema()
    result = summarize(SAMPLE_ROWS, schema, period="2020")
    assert result.error is None
    assert result.stats.count == 3
    assert result.stats.min_value == 10.0
    assert result.stats.max_value == 30.0
    assert result.stats.mean == pytest.approx(20.0)
    assert result.stats.median == 20.0
    assert result.stats.min_ref_area == "A"
    assert result.stats.max_ref_area == "B"


def test_summarize_std_dev():
    schema = make_schema()
    result = summarize(SAMPLE_ROWS, schema, period="2020")
    import statistics
    expected_std = statistics.stdev([10.0, 30.0, 20.0])
    assert result.stats.std == pytest.approx(expected_std)


def test_summarize_single_value_std_is_zero():
    schema = make_schema()
    rows = make_rows([{"GEO": "A", "PERIOD": "2020", "VALUE": "5.0"}])
    result = summarize(rows, schema, period="2020")
    assert result.stats.std == 0.0
    assert result.stats.count == 1


def test_summarize_unknown_period_returns_error():
    schema = make_schema()
    result = summarize(SAMPLE_ROWS, schema, period="1999")
    assert result.error is not None
    assert result.stats.count == 0


def test_summarize_with_dimension_filter():
    schema = make_schema(DSD_WITH_SEX)
    result = summarize(ROWS_WITH_SEX, schema, period="2020", dimensions={"SEX": "M"})
    assert result.stats.count == 2
    assert result.stats.max_value == 30.0
    assert result.stats.min_value == 10.0


# ---------------------------------------------------------------------------
# growth
# ---------------------------------------------------------------------------

def test_growth_computes_changes():
    schema = make_schema()
    result = growth(SAMPLE_ROWS, schema, base_period="2020", end_period="2021")
    assert result.error is None
    assert len(result.rows) == 3

    row_b = next(r for r in result.rows if r.ref_area == "B")
    assert row_b.base_value == 30.0
    assert row_b.end_value == 25.0
    assert row_b.absolute_change == pytest.approx(-5.0)
    assert row_b.pct_change == pytest.approx(-5.0 / 30.0 * 100)


def test_growth_none_when_missing_period():
    schema = make_schema()
    rows = make_rows([
        {"GEO": "A", "PERIOD": "2020", "VALUE": "10.0"},
        # No 2021 data for A
    ])
    result = growth(rows, schema, base_period="2020", end_period="2021")
    row_a = next(r for r in result.rows if r.ref_area == "A")
    assert row_a.base_value == 10.0
    assert row_a.end_value is None
    assert row_a.absolute_change is None
    assert row_a.pct_change is None


def test_growth_filters_to_requested_ref_areas():
    schema = make_schema()
    result = growth(SAMPLE_ROWS, schema, ref_areas=["A", "B"],
                    base_period="2020", end_period="2021")
    assert len(result.rows) == 2
    codes = {r.ref_area for r in result.rows}
    assert codes == {"A", "B"}


def test_growth_zero_base_value_no_pct():
    schema = make_schema()
    rows = make_rows([
        {"GEO": "A", "PERIOD": "2020", "VALUE": "0.0"},
        {"GEO": "A", "PERIOD": "2021", "VALUE": "5.0"},
    ])
    result = growth(rows, schema, base_period="2020", end_period="2021")
    row_a = result.rows[0]
    assert row_a.absolute_change == 5.0
    assert row_a.pct_change is None  # division by zero guard


def test_growth_with_dimension_filter():
    schema = make_schema(DSD_WITH_SEX)
    result = growth(ROWS_WITH_SEX, schema, base_period="2020", end_period="2020",
                    dimensions={"SEX": "F"})
    # base and end are the same period — abs change = 0
    for row in result.rows:
        assert row.absolute_change == pytest.approx(0.0)
