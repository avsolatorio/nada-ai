"""Tests for NADA catalog search API client and models."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from nada_ai.nada.api import search_catalog
from nada_ai.nada.models import CatalogSearchRequest
from nada_ai.mcp_server.security_validator import validate_catalog_search_arguments

SAMPLE_API_RESPONSE = {
    "status": "success",
    "result": {
        "found": "196",
        "total": "1380",
        "limit": "2",
        "offset": 0,
        "page": 1,
        "rows": [
            {
                "id": "437",
                "type": "timeseries",
                "idno": "UIS_ADMI.GRADE2OR3PRIM.MAT",
                "title": "Administration of a nationally representative learning assessment in Grade 2 or 3 in mathematics (number)",
                "url": "http://nada.example.org/catalog/437",
                "ts_dimensions": "MEASURE,REF_AREA",
                "ts_data_count": "1334",
            },
            {
                "id": "512",
                "type": "timeseries",
                "idno": "WDI_SP.POP.TOTL",
                "title": "Population, total",
                "nation": "World",
                "authoring_entity": "World Bank",
                "url": "http://nada.example.org/catalog/512",
                "ts_dimensions": "COUNTRY,FREQ,INDICATOR",
                "ts_frequency": "Annual",
                "ts_data_count": "13456",
                "ts_db_title": "World Development Indicators",
            },
        ],
        "search_counts_by_type": {
            "survey": "185",
            "document": "913",
            "timeseries": "196",
        },
    },
    "params": {
        "type": "timeseries",
        "ps": "2",
        "sort_by": "title",
        "sort_order": "asc",
    },
}


def test_catalog_search_request_maps_params():
    request = CatalogSearchRequest(
        keywords="population",
        type="timeseries,survey",
        from_year=2010,
        to_year=2023,
        country_iso3="rwa|ken",
        include_iso3=True,
        include_facets=True,
        include_resources=True,
        data_access_type="public",
        study_id=42,
        repository="central",
        page_size=25,
        page=2,
        sort_by="popularity",
        sort_order="desc",
    )
    params = request.to_api_params()

    assert params["sk"] == "population"
    assert params["type"] == "timeseries,survey"
    assert params["from"] == 2010
    assert params["to"] == 2023
    assert params["country_iso3"] == "rwa|ken"
    assert params["inc_iso"] == 1
    assert params["include_facets"] == 1
    assert params["include_resources"] == "true"
    assert params["dtype"] == "public"
    assert params["sid"] == 42
    assert params["repo"] == "central"
    assert params["ps"] == 25
    assert params["page"] == 2
    assert params["sort_by"] == "popularity"
    assert params["sort_order"] == "desc"


def test_catalog_search_request_omits_unset_optional_params():
    request = CatalogSearchRequest()
    params = request.to_api_params()

    assert "sk" not in params
    assert params["type"] == "timeseries"
    assert "inc_iso" not in params
    assert "include_facets" not in params
    assert params["ps"] == 15
    assert params["page"] == 1


def test_build_paged_response_from_fixture():
    from nada_ai.nada.api import _build_paged_response

    request = CatalogSearchRequest(page_size=2, page=1)
    response = _build_paged_response(request, SAMPLE_API_RESPONSE["result"], params=SAMPLE_API_RESPONSE["params"])

    assert response.count == 2
    assert response.total_count == 196
    assert response.page == 1
    assert response.page_size == 2
    assert response.has_more is True
    assert response.next_page == 2
    assert response.items[0].idno == "UIS_ADMI.GRADE2OR3PRIM.MAT"
    assert response.items[1].ts_db_title == "World Development Indicators"
    assert response.search_counts_by_type["timeseries"] == "196"
    assert response.params["type"] == "timeseries"


def test_build_paged_response_last_page():
    from nada_ai.nada.api import _build_paged_response

    request = CatalogSearchRequest(page_size=50, page=4)
    result = {"found": "196", "rows": []}
    response = _build_paged_response(request, result)

    assert response.has_more is False
    assert response.next_page is None


def test_search_catalog_success():
    mock_response = httpx.Response(200, json=SAMPLE_API_RESPONSE, request=httpx.Request("GET", "http://test"))

    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        response = asyncio.run(
            search_catalog(CatalogSearchRequest(keywords="population", type="timeseries", page_size=2))
        )

    assert response.error is None
    assert response.count == 2
    assert response.items[0].id == "437"
    mock_client.get.assert_awaited_once()
    call_kwargs = mock_client.get.await_args.kwargs
    assert call_kwargs["params"]["sk"] == "population"
    assert call_kwargs["params"]["type"] == "timeseries"


def test_search_catalog_http_error():
    request = httpx.Request("GET", "http://test")
    mock_response = httpx.Response(403, text="Forbidden", request=request)
    http_error = httpx.HTTPStatusError("Forbidden", request=request, response=mock_response)

    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.side_effect = http_error
        mock_client_cls.return_value = mock_client

        response = asyncio.run(search_catalog(CatalogSearchRequest(keywords="test")))

    assert response.error is not None
    assert "403" in response.error
    assert response.items == []


def test_search_catalog_api_error_status():
    payload = {"status": "error", "message": "Invalid filter"}
    mock_response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))

    with patch("nada_ai.nada.api.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        response = asyncio.run(search_catalog(CatalogSearchRequest()))

    assert response.error == "Catalog search API error: Invalid filter"


def test_validate_catalog_search_browse_mode_allowed():
    is_valid, error = validate_catalog_search_arguments({})
    assert is_valid is True
    assert error is None

    is_valid, error = validate_catalog_search_arguments({"keywords": "  "})
    assert is_valid is True
    assert error is None


def test_validate_catalog_search_keywords_enforced():
    is_valid, error = validate_catalog_search_arguments({"keywords": "ab"})
    assert is_valid is False
    assert error is not None

    is_valid, error = validate_catalog_search_arguments({"keywords": "population census"})
    assert is_valid is True
    assert error is None
