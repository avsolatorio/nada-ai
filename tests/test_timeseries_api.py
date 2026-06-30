"""Tests for get_indicator_schema, get_codelist, and get_all_timeseries_data API functions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nada_ai.nada.api import get_all_timeseries_data, get_codelist, get_indicator_schema
from nada_ai.nada.models import IndicatorSchemaResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SCHEMA_PAYLOAD = {
    "status": "success",
    "result": {
        "sid": 2388,
        "dsd_id": 25,
        "components": [
            {"name": "DATASET", "label": "Dataset ID", "data_type": "string",
             "column_type": "attribute", "codelist_id": None, "time_period_format": None},
            {"name": "COUNTRY_CODE", "label": "Country code", "data_type": "string",
             "column_type": "geography", "codelist_id": "24", "time_period_format": None},
            {"name": "COUNTRY_NAME", "label": "Country name", "data_type": "string",
             "column_type": "attribute", "codelist_id": None, "time_period_format": None},
            {"name": "TIME_PERIOD", "label": "Time period", "data_type": "string",
             "column_type": "time_period", "codelist_id": None, "time_period_format": "YYYY"},
            {"name": "OBS_VALUE", "label": "Value", "data_type": "double",
             "column_type": "observation_value", "codelist_id": None, "time_period_format": None},
        ],
        "reporting_year_bounds": {"min": 1990, "max": 2023},
    },
}

SAMPLE_DATA_PAYLOAD = {
    "status": "success",
    "result": {
        "data": [
            {"COUNTRY_CODE": "KEN", "COUNTRY_NAME": "Kenya",
             "TIME_PERIOD": "2020", "OBS_VALUE": "4.5",
             "INDICATOR_NAME": "Test Indicator", "sid": 1, "dsd_id": 25, "idno": "TEST"},
            {"COUNTRY_CODE": "UGA", "COUNTRY_NAME": "Uganda",
             "TIME_PERIOD": "2020", "OBS_VALUE": "7.1",
             "INDICATOR_NAME": "Test Indicator", "sid": 1, "dsd_id": 25, "idno": "TEST"},
            {"COUNTRY_CODE": "TZA", "COUNTRY_NAME": "Tanzania",
             "TIME_PERIOD": "2021", "OBS_VALUE": "3.2",
             "INDICATOR_NAME": "Test Indicator", "sid": 1, "dsd_id": 25, "idno": "TEST"},
        ],
        "limit": 100,
        "offset": 0,
        "total": 3,
        "found": 3,
    },
}


def _mock_client(payload: dict) -> AsyncMock:
    mock_resp = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_resp
    return mock_client


# ---------------------------------------------------------------------------
# get_indicator_schema
# ---------------------------------------------------------------------------

def test_get_indicator_schema_success():
    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(SAMPLE_SCHEMA_PAYLOAD)
        resp = asyncio.run(get_indicator_schema("TEST"))

    assert resp.error is None
    assert resp.schema_ is not None
    assert resp.schema_.idno == "TEST"
    assert resp.schema_.geo_column == "COUNTRY_CODE"
    assert resp.schema_.time_column == "TIME_PERIOD"
    assert resp.schema_.obs_column == "OBS_VALUE"
    assert resp.schema_.time_period_format == "YYYY"
    assert resp.schema_.reporting_year_bounds == {"min": 1990, "max": 2023}


def test_get_indicator_schema_http_error():
    req = httpx.Request("GET", "http://test")
    mock_resp = httpx.Response(404, text="Not found", request=req)
    exc = httpx.HTTPStatusError("Not found", request=req, response=mock_resp)

    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        client.get.side_effect = exc
        mock_cls.return_value = client

        resp = asyncio.run(get_indicator_schema("BAD"))

    assert resp.error is not None
    assert "404" in resp.error
    assert resp.schema_ is None


def test_get_indicator_schema_api_error_status():
    payload = {"status": "error", "message": "Indicator not found"}
    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(payload)
        resp = asyncio.run(get_indicator_schema("MISSING"))

    assert resp.error is not None
    assert "Indicator not found" in resp.error


# ---------------------------------------------------------------------------
# get_codelist
# ---------------------------------------------------------------------------

def test_get_codelist_success():
    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(SAMPLE_SCHEMA_PAYLOAD)
        schema_resp = asyncio.run(get_indicator_schema("TEST"))

    # Now patch both schema and data calls
    with patch("nada_ai.nada.api.get_indicator_schema", return_value=schema_resp), \
         patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(SAMPLE_DATA_PAYLOAD)
        codelist = asyncio.run(get_codelist("TEST", "COUNTRY_CODE"))

    assert codelist.error is None
    codes = {e.code for e in codelist.entries}
    assert "KEN" in codes
    assert "UGA" in codes
    assert "TZA" in codes


def test_get_codelist_unknown_component():
    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(SAMPLE_SCHEMA_PAYLOAD)
        schema_resp = asyncio.run(get_indicator_schema("TEST"))

    with patch("nada_ai.nada.api.get_indicator_schema", return_value=schema_resp):
        codelist = asyncio.run(get_codelist("TEST", "NONEXISTENT_COL"))

    assert codelist.error is not None
    assert "NONEXISTENT_COL" in codelist.error


def test_get_codelist_schema_error_propagates():
    error_resp = IndicatorSchemaResponse(error="Schema not available")
    with patch("nada_ai.nada.api.get_indicator_schema", return_value=error_resp):
        codelist = asyncio.run(get_codelist("BAD", "COUNTRY_CODE"))

    assert codelist.error is not None
    assert codelist.entries == []


def test_get_codelist_finds_label_column():
    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(SAMPLE_SCHEMA_PAYLOAD)
        schema_resp = asyncio.run(get_indicator_schema("TEST"))

    with patch("nada_ai.nada.api.get_indicator_schema", return_value=schema_resp), \
         patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(SAMPLE_DATA_PAYLOAD)
        codelist = asyncio.run(get_codelist("TEST", "COUNTRY_CODE"))

    # Should detect COUNTRY_NAME as label column
    assert codelist.label_column == "COUNTRY_NAME"
    kenya = next((e for e in codelist.entries if e.code == "KEN"), None)
    assert kenya is not None
    assert kenya.label == "Kenya"


# ---------------------------------------------------------------------------
# get_all_timeseries_data
# ---------------------------------------------------------------------------

def test_get_all_timeseries_data_single_page():
    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(SAMPLE_DATA_PAYLOAD)
        resp = asyncio.run(get_all_timeseries_data("TEST"))

    assert resp.error is None
    assert len(resp.data) == 3
    assert resp.total == 3
    assert resp.has_more is False


def test_get_all_timeseries_data_paginates():
    page1 = {
        "status": "success",
        "result": {
            "data": [{"COUNTRY_CODE": f"C{i}", "TIME_PERIOD": "2020", "OBS_VALUE": str(i),
                       "sid": 1, "dsd_id": 1, "idno": "T"}
                     for i in range(5)],
            "limit": 5, "offset": 0, "total": 8, "found": 5,
        },
    }
    page2 = {
        "status": "success",
        "result": {
            "data": [{"COUNTRY_CODE": f"C{i}", "TIME_PERIOD": "2020", "OBS_VALUE": str(i),
                       "sid": 1, "dsd_id": 1, "idno": "T"}
                     for i in range(5, 8)],
            "limit": 5, "offset": 5, "total": 8, "found": 3,
        },
    }

    def make_mock(payloads):
        responses = [
            httpx.Response(200, json=p, request=httpx.Request("GET", "http://test"))
            for p in payloads
        ]
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        client.get.side_effect = responses
        return client

    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = make_mock([page1, page2])
        resp = asyncio.run(get_all_timeseries_data("TEST", page_size=5))

    assert len(resp.data) == 8
    assert resp.total == 8
    assert resp.has_more is False


def test_get_all_timeseries_data_respects_max_rows():
    # Server honours the limit=min(page_size, max_rows)=50 and returns 50 rows.
    # The function should not make a second request and should set has_more=True.
    capped_page = {
        "status": "success",
        "result": {
            "data": [{"COUNTRY_CODE": f"C{i}", "TIME_PERIOD": "2020", "OBS_VALUE": str(i),
                       "sid": 1, "dsd_id": 1, "idno": "T"}
                     for i in range(50)],
            "limit": 50, "offset": 0, "total": 10000, "found": 50,
        },
    }
    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(capped_page)
        resp = asyncio.run(get_all_timeseries_data("TEST", max_rows=50, page_size=100))

    assert len(resp.data) == 50
    assert resp.has_more is True


def test_get_all_timeseries_data_propagates_error():
    payload = {"status": "error", "message": "Bad request"}
    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_client(payload)
        resp = asyncio.run(get_all_timeseries_data("BAD"))

    assert resp.error is not None
    assert resp.data == []
