"""Tests for timeseries schema, codelist, and analytical response models."""

from __future__ import annotations

import pytest

from nada_ai.nada.models import (
    DSComponent,
    ExtremesResponse,
    GrowthResponse,
    IndicatorSchema,
    IndicatorSchemaResponse,
    RankResponse,
    RankRow,
    SummarizeResponse,
    TimeseriesDataResponse,
    TimeseriesDataRow,
)

# ---------------------------------------------------------------------------
# Sample DSD result fixture
# ---------------------------------------------------------------------------

SAMPLE_DSD_RESULT = {
    "sid": 2388,
    "dsd_id": 25,
    "components": [
        {"id": "1", "name": "DATASET", "label": "Dataset ID", "description": "DB id",
         "data_type": "string", "column_type": "attribute", "codelist_id": None,
         "time_period_format": None},
        {"id": "2", "name": "INDICATOR", "label": "Indicator code", "description": None,
         "data_type": "string", "column_type": "indicator_id", "codelist_id": None,
         "time_period_format": None},
        {"id": "3", "name": "INDICATOR_NAME", "label": "Indicator name", "description": None,
         "data_type": "string", "column_type": "attribute", "codelist_id": None,
         "time_period_format": None},
        {"id": "4", "name": "COUNTRY_CODE", "label": "Country/area code", "description": None,
         "data_type": "string", "column_type": "geography", "codelist_id": "24",
         "time_period_format": None},
        {"id": "5", "name": "COUNTRY_NAME", "label": "Country/area name", "description": None,
         "data_type": "string", "column_type": "attribute", "codelist_id": None,
         "time_period_format": None},
        {"id": "6", "name": "FREQ", "label": "Frequency", "description": None,
         "data_type": "string", "column_type": "periodicity", "codelist_id": None,
         "time_period_format": None},
        {"id": "7", "name": "TIME_PERIOD", "label": "Time period", "description": None,
         "data_type": "string", "column_type": "time_period", "codelist_id": None,
         "time_period_format": "YYYY"},
        {"id": "8", "name": "OBS_VALUE", "label": "Observation value", "description": None,
         "data_type": "double", "column_type": "observation_value", "codelist_id": None,
         "time_period_format": None},
    ],
    "time_period_component": "TIME_PERIOD",
    "observation_value_component": "OBS_VALUE",
    "reporting_year_bounds": {"min": 1990, "max": 2023},
}

SAMPLE_DSD_WITH_DIMENSION = {
    **SAMPLE_DSD_RESULT,
    "components": SAMPLE_DSD_RESULT["components"] + [
        {"id": "9", "name": "SEX", "label": "Sex", "description": "Sex disaggregation",
         "data_type": "string", "column_type": "dimension", "codelist_id": "42",
         "time_period_format": None},
    ],
}


# ---------------------------------------------------------------------------
# DSComponent
# ---------------------------------------------------------------------------

def test_ds_component_parses_extra_fields():
    comp = DSComponent.model_validate({
        "id": "1", "sort_order": "0", "data_structure_id": "25",
        "name": "TIME_PERIOD", "label": "Time period",
        "data_type": "string", "column_type": "time_period",
        "codelist_id": None, "time_period_format": "YYYY",
        "metadata": None, "created": "123", "updated": "123",
        "created_by": "1", "updated_by": "1",
    })
    assert comp.name == "TIME_PERIOD"
    assert comp.column_type == "time_period"
    assert comp.time_period_format == "YYYY"


# ---------------------------------------------------------------------------
# IndicatorSchema.from_api_result
# ---------------------------------------------------------------------------

def test_indicator_schema_resolves_role_columns():
    schema = IndicatorSchema.from_api_result("VC.IHR.PSRC.P5", SAMPLE_DSD_RESULT)
    assert schema.idno == "VC.IHR.PSRC.P5"
    assert schema.geo_column == "COUNTRY_CODE"
    assert schema.time_column == "TIME_PERIOD"
    assert schema.obs_column == "OBS_VALUE"
    assert schema.freq_column == "FREQ"
    assert schema.time_period_format == "YYYY"
    assert schema.reporting_year_bounds == {"min": 1990, "max": 2023}


def test_indicator_schema_no_free_dimensions_for_wdi():
    schema = IndicatorSchema.from_api_result("TEST", SAMPLE_DSD_RESULT)
    assert schema.dimension_columns == []


def test_indicator_schema_detects_dimension_column():
    schema = IndicatorSchema.from_api_result("TEST", SAMPLE_DSD_WITH_DIMENSION)
    assert "SEX" in schema.dimension_columns
    assert len(schema.dimension_columns) == 1


def test_indicator_schema_missing_components_returns_none_columns():
    schema = IndicatorSchema.from_api_result("EMPTY", {"components": []})
    assert schema.geo_column is None
    assert schema.time_column is None
    assert schema.obs_column is None


def test_indicator_schema_response_error_path():
    resp = IndicatorSchemaResponse(error="Something went wrong")
    assert resp.error == "Something went wrong"
    assert resp.schema_ is None


# ---------------------------------------------------------------------------
# TimeseriesDataRow — extra columns via model_extra
# ---------------------------------------------------------------------------

def test_timeseries_data_row_known_columns():
    row = TimeseriesDataRow.model_validate({
        "COUNTRY_CODE": "KEN",
        "TIME_PERIOD": "2022",
        "OBS_VALUE": "4.5",
        "INDICATOR_NAME": "Homicides",
    })
    assert row.COUNTRY_CODE == "KEN"
    assert row.TIME_PERIOD == "2022"
    assert row.OBS_VALUE == "4.5"


def test_timeseries_data_row_extra_dimension_column():
    row = TimeseriesDataRow.model_validate({
        "COUNTRY_CODE": "KEN",
        "TIME_PERIOD": "2022",
        "OBS_VALUE": "4.5",
        "SEX": "F",
        "AGE_GROUP": "15-24",
    })
    assert row.model_extra["SEX"] == "F"
    assert row.model_extra["AGE_GROUP"] == "15-24"


# ---------------------------------------------------------------------------
# TimeseriesDataResponse
# ---------------------------------------------------------------------------

def test_timeseries_data_response_has_more_logic():
    resp = TimeseriesDataResponse(idno="X", total=100, found=10, limit=10, offset=0)
    assert resp.has_more is False  # default

    resp2 = TimeseriesDataResponse(idno="X", total=100, found=10, limit=10, offset=0, has_more=True)
    assert resp2.has_more is True


# ---------------------------------------------------------------------------
# Analytical response models — basic validation
# ---------------------------------------------------------------------------

def test_rank_response_serialises():
    resp = RankResponse(
        idno="X", period="2022", n=3, ascending=False,
        geo_column="COUNTRY_CODE", time_column="TIME_PERIOD", obs_column="OBS_VALUE",
        rows=[RankRow(rank=1, ref_area="KEN", period="2022", value=4.5)],
        total_ref_areas=50,
    )
    assert resp.rows[0].ref_area == "KEN"
    assert resp.rows[0].rank == 1
    assert resp.total_ref_areas == 50


def test_extremes_response_optional_points():
    resp = ExtremesResponse(
        idno="X", geo_column="G", time_column="T", obs_column="O",
        maximum=None, minimum=None,
    )
    assert resp.maximum is None
    assert resp.minimum is None


def test_growth_response_pct_change_none_when_base_missing():
    resp = GrowthResponse(
        idno="X", base_period="2010", end_period="2022",
        geo_column="G", obs_column="O",
    )
    assert resp.rows == []


def test_summarize_response_zero_count():
    resp = SummarizeResponse(
        idno="X", period="2022", geo_column="G", obs_column="O",
        error="No data found for period '2022' after applying filters.",
    )
    assert resp.stats.count == 0
    assert resp.error is not None
