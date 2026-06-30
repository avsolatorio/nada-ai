"""Tests for analytical MCP tools registered in mcp_server/tools.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from nada_ai.nada.models import (
    CodelistEntry,
    CodelistResponse,
    CompareResponse,
    CompareRow,
    ExtremesResponse,
    GrowthResponse,
    IndicatorSchema,
    IndicatorSchemaResponse,
    RankResponse,
    RankRow,
    SummarizeResponse,
    SummaryStats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DSD_RESULT = {
    "sid": 1, "dsd_id": 1,
    "components": [
        {"name": "COUNTRY_CODE", "label": "Country", "data_type": "string",
         "column_type": "geography", "codelist_id": "24", "time_period_format": None},
        {"name": "COUNTRY_NAME", "label": "Country name", "data_type": "string",
         "column_type": "attribute", "codelist_id": None, "time_period_format": None},
        {"name": "TIME_PERIOD", "label": "Period", "data_type": "string",
         "column_type": "time_period", "codelist_id": None, "time_period_format": "YYYY"},
        {"name": "OBS_VALUE", "label": "Value", "data_type": "double",
         "column_type": "observation_value", "codelist_id": None, "time_period_format": None},
    ],
    "reporting_year_bounds": {"min": 2000, "max": 2022},
}

SAMPLE_DATA_PAYLOAD = {
    "status": "success",
    "result": {
        "data": [
            {"COUNTRY_CODE": "KEN", "COUNTRY_NAME": "Kenya",
             "TIME_PERIOD": "2020", "OBS_VALUE": "4.5",
             "sid": 1, "dsd_id": 1, "idno": "TEST"},
            {"COUNTRY_CODE": "UGA", "COUNTRY_NAME": "Uganda",
             "TIME_PERIOD": "2020", "OBS_VALUE": "7.1",
             "sid": 1, "dsd_id": 1, "idno": "TEST"},
        ],
        "limit": 100, "offset": 0, "total": 2, "found": 2,
    },
}


def _schema_resp() -> IndicatorSchemaResponse:
    return IndicatorSchemaResponse(schema_=IndicatorSchema.from_api_result("TEST", DSD_RESULT))


def _error_schema_resp() -> IndicatorSchemaResponse:
    return IndicatorSchemaResponse(error="Schema not available")


# ---------------------------------------------------------------------------
# Tool registration — verify names and existence
# ---------------------------------------------------------------------------

def test_analytical_tools_are_registered():
    """All analytical tools must appear in the MCP server tool list."""
    import nada_ai.mcp_server  # noqa: F401 — triggers registration
    from nada_ai.mcp_server._server_definition import mcp

    tool_names = asyncio.run(mcp.list_tools())
    names = {t.name for t in tool_names}

    assert any("get_schema" in n for n in names), f"get_schema not found in {names}"
    assert any("get_codelist" in n for n in names), f"get_codelist not found in {names}"
    assert any("rank" in n for n in names), f"rank not found in {names}"
    assert any("extremes" in n for n in names), f"extremes not found in {names}"
    assert any("compare" in n for n in names), f"compare not found in {names}"
    assert any("summarize" in n for n in names), f"summarize not found in {names}"
    assert any("growth" in n for n in names), f"growth not found in {names}"


# ---------------------------------------------------------------------------
# _nada_get_schema
# ---------------------------------------------------------------------------

def test_nada_get_schema_success():
    from nada_ai.mcp_server.tools import _nada_get_schema

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_schema_resp()):
        result = asyncio.run(_nada_get_schema("TEST"))

    assert result.error is None
    assert result.schema_.idno == "TEST"
    assert result.schema_.geo_column == "COUNTRY_CODE"


def test_nada_get_schema_propagates_error():
    from nada_ai.mcp_server.tools import _nada_get_schema

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_error_schema_resp()):
        result = asyncio.run(_nada_get_schema("BAD"))

    assert result.error is not None


# ---------------------------------------------------------------------------
# _nada_get_codelist
# ---------------------------------------------------------------------------

def test_nada_get_codelist_success():
    from nada_ai.mcp_server.tools import _nada_get_codelist

    codelist = CodelistResponse(
        idno="TEST", component="COUNTRY_CODE", label_column="COUNTRY_NAME",
        entries=[CodelistEntry(code="KEN", label="Kenya"),
                 CodelistEntry(code="UGA", label="Uganda")],
        is_complete=True,
    )
    with patch("nada_ai.mcp_server.tools.nada_api.get_codelist",
               new_callable=AsyncMock, return_value=codelist):
        result = asyncio.run(_nada_get_codelist("TEST", "COUNTRY_CODE"))

    assert result.error is None
    assert len(result.entries) == 2
    assert result.entries[0].code == "KEN"


# ---------------------------------------------------------------------------
# _nada_rank
# ---------------------------------------------------------------------------

def test_nada_rank_returns_rank_response():
    from nada_ai.mcp_server.tools import _nada_rank

    rank_resp = RankResponse(
        idno="TEST", period="2020", n=2, ascending=False,
        geo_column="COUNTRY_CODE", time_column="TIME_PERIOD", obs_column="OBS_VALUE",
        rows=[
            RankRow(rank=1, ref_area="UGA", period="2020", value=7.1),
            RankRow(rank=2, ref_area="KEN", period="2020", value=4.5),
        ],
        total_ref_areas=2,
    )

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_schema_resp()), \
         patch("nada_ai.mcp_server.tools.nada_api.get_all_timeseries_data",
               new_callable=AsyncMock) as mock_data, \
         patch("nada_ai.mcp_server.tools.analytics.rank", return_value=rank_resp):
        from nada_ai.nada.models import TimeseriesDataResponse
        mock_data.return_value = TimeseriesDataResponse(idno="TEST", total=2, found=2)
        result = asyncio.run(_nada_rank("TEST", "2020", n=2))

    assert result.rows[0].ref_area == "UGA"
    assert result.rows[0].value == 7.1


def test_nada_rank_schema_error_returns_error_response():
    from nada_ai.mcp_server.tools import _nada_rank

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_error_schema_resp()):
        result = asyncio.run(_nada_rank("BAD", "2020"))

    assert result.error is not None


# ---------------------------------------------------------------------------
# _nada_extremes
# ---------------------------------------------------------------------------

def test_nada_extremes_schema_error():
    from nada_ai.mcp_server.tools import _nada_extremes

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_error_schema_resp()):
        result = asyncio.run(_nada_extremes("BAD"))

    assert result.error is not None


def test_nada_extremes_success_delegation():
    from nada_ai.mcp_server.tools import _nada_extremes
    from nada_ai.nada.models import ExtremePoint

    extremes_resp = ExtremesResponse(
        idno="TEST", geo_column="COUNTRY_CODE",
        time_column="TIME_PERIOD", obs_column="OBS_VALUE",
        maximum=ExtremePoint(ref_area="UGA", period="2020", value=7.1),
        minimum=ExtremePoint(ref_area="KEN", period="2020", value=4.5),
        total_observations=2,
    )

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_schema_resp()), \
         patch("nada_ai.mcp_server.tools.nada_api.get_all_timeseries_data",
               new_callable=AsyncMock) as mock_data, \
         patch("nada_ai.mcp_server.tools.analytics.get_extremes", return_value=extremes_resp):
        from nada_ai.nada.models import TimeseriesDataResponse
        mock_data.return_value = TimeseriesDataResponse(idno="TEST")
        result = asyncio.run(_nada_extremes("TEST"))

    assert result.maximum.value == 7.1
    assert result.minimum.value == 4.5


# ---------------------------------------------------------------------------
# _nada_compare
# ---------------------------------------------------------------------------

def test_nada_compare_schema_error():
    from nada_ai.mcp_server.tools import _nada_compare

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_error_schema_resp()):
        result = asyncio.run(_nada_compare("BAD", ["KEN", "UGA"]))

    assert result.error is not None


def test_nada_compare_success():
    from nada_ai.mcp_server.tools import _nada_compare

    compare_resp = CompareResponse(
        idno="TEST", ref_areas=["KEN", "UGA"],
        geo_column="COUNTRY_CODE", time_column="TIME_PERIOD", obs_column="OBS_VALUE",
        rows=[CompareRow(period="2020", values={"KEN": 4.5, "UGA": 7.1})],
    )

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_schema_resp()), \
         patch("nada_ai.mcp_server.tools.nada_api.get_all_timeseries_data",
               new_callable=AsyncMock) as mock_data, \
         patch("nada_ai.mcp_server.tools.analytics.compare", return_value=compare_resp):
        from nada_ai.nada.models import TimeseriesDataResponse
        mock_data.return_value = TimeseriesDataResponse(idno="TEST")
        result = asyncio.run(_nada_compare("TEST", ["KEN", "UGA"]))

    assert result.rows[0].values["KEN"] == 4.5


# ---------------------------------------------------------------------------
# _nada_summarize
# ---------------------------------------------------------------------------

def test_nada_summarize_schema_error():
    from nada_ai.mcp_server.tools import _nada_summarize

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_error_schema_resp()):
        result = asyncio.run(_nada_summarize("BAD", "2020"))

    assert result.error is not None


def test_nada_summarize_success():
    from nada_ai.mcp_server.tools import _nada_summarize

    summarize_resp = SummarizeResponse(
        idno="TEST", period="2020",
        geo_column="COUNTRY_CODE", obs_column="OBS_VALUE",
        stats=SummaryStats(count=2, min_value=4.5, max_value=7.1,
                           mean=5.8, median=5.8, std=1.86,
                           min_ref_area="KEN", max_ref_area="UGA"),
    )

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_schema_resp()), \
         patch("nada_ai.mcp_server.tools.nada_api.get_all_timeseries_data",
               new_callable=AsyncMock) as mock_data, \
         patch("nada_ai.mcp_server.tools.analytics.summarize", return_value=summarize_resp):
        from nada_ai.nada.models import TimeseriesDataResponse
        mock_data.return_value = TimeseriesDataResponse(idno="TEST")
        result = asyncio.run(_nada_summarize("TEST", "2020"))

    assert result.stats.count == 2
    assert result.stats.max_ref_area == "UGA"


# ---------------------------------------------------------------------------
# _nada_growth
# ---------------------------------------------------------------------------

def test_nada_growth_schema_error():
    from nada_ai.mcp_server.tools import _nada_growth

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_error_schema_resp()):
        result = asyncio.run(_nada_growth("BAD", "2010", "2020"))

    assert result.error is not None


def test_nada_growth_success():
    from nada_ai.mcp_server.tools import _nada_growth
    from nada_ai.nada.models import GrowthRow

    growth_resp = GrowthResponse(
        idno="TEST", base_period="2010", end_period="2020",
        geo_column="COUNTRY_CODE", obs_column="OBS_VALUE",
        rows=[GrowthRow(ref_area="KEN", base_value=3.0, end_value=4.5,
                        absolute_change=1.5, pct_change=50.0)],
    )

    with patch("nada_ai.mcp_server.tools.nada_api.get_indicator_schema",
               new_callable=AsyncMock, return_value=_schema_resp()), \
         patch("nada_ai.mcp_server.tools.nada_api.get_all_timeseries_data",
               new_callable=AsyncMock) as mock_data, \
         patch("nada_ai.mcp_server.tools.analytics.growth", return_value=growth_resp):
        from nada_ai.nada.models import TimeseriesDataResponse
        mock_data.return_value = TimeseriesDataResponse(idno="TEST")
        result = asyncio.run(_nada_growth("TEST", "2010", "2020"))

    assert result.rows[0].pct_change == 50.0
