"""Tests for UI FastMCPApps — verify apps render without error and produce valid PrefabApp."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest
from prefab_ui.app import PrefabApp


def _get_json(app):
    """Extract the state dict from either a PrefabApp or a ToolResult."""
    if isinstance(app, PrefabApp):
        return app.to_json()
    # ToolResult — structured_content carries the prefab JSON
    if hasattr(app, "structured_content") and isinstance(app.structured_content, dict):
        return app.structured_content
    raise TypeError(f"Unexpected app type: {type(app)}")


# ---------------------------------------------------------------------------
# search_app
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_app_renders():
    from nada_ai.mcp_server.search_app import search_catalog_ui

    app = await search_catalog_ui()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d
    assert "$prefab" in d or "view" in d


# ---------------------------------------------------------------------------
# schema_app
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_app_renders():
    from nada_ai.mcp_server.apps.schema_app import explore_indicator_schema

    app = await explore_indicator_schema()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


@pytest.mark.asyncio
async def test_schema_app_with_idno():
    from nada_ai.mcp_server.apps.schema_app import explore_indicator_schema

    app = await explore_indicator_schema(idno="SP.POP.TOTL")
    d = _get_json(app)
    assert d["state"]["idno"] == "SP.POP.TOTL"


# ---------------------------------------------------------------------------
# rank_app
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rank_app_renders():
    from nada_ai.mcp_server.apps.rank_app import rank_ref_areas

    app = await rank_ref_areas()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


@pytest.mark.asyncio
async def test_rank_app_with_args():
    from nada_ai.mcp_server.apps.rank_app import rank_ref_areas

    app = await rank_ref_areas(idno="SP.POP.TOTL", period="2022", n=5)
    d = _get_json(app)
    assert d["state"]["idno"] == "SP.POP.TOTL"
    assert d["state"]["period"] == "2022"
    assert d["state"]["n"] == 5


@pytest.mark.asyncio
async def test_rank_app_ascending_flag():
    from nada_ai.mcp_server.apps.rank_app import rank_ref_areas

    app = await rank_ref_areas(idno="TEST", period="2020", ascending=True)
    d = _get_json(app)
    assert d["state"]["ascending"] is True


# ---------------------------------------------------------------------------
# compare_app
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_app_renders():
    from nada_ai.mcp_server.apps.compare_app import compare_ref_areas

    app = await compare_ref_areas()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


@pytest.mark.asyncio
async def test_compare_app_keeps_ref_areas_live_bound():
    """State exposes the raw ref_areas string (matching the live text Input's
    binding) rather than a pre-parsed snapshot list that would go stale as
    soon as the user edits the field — see do_compare's docstring."""
    from nada_ai.mcp_server.apps.compare_app import compare_ref_areas

    app = await compare_ref_areas(idno="TEST", ref_areas="KEN,UGA,TZA")
    d = _get_json(app)
    assert d["state"]["ref_areas"] == "KEN,UGA,TZA"
    assert "ref_areas_list" not in d["state"]


@pytest.mark.asyncio
async def test_compare_app_empty_ref_areas():
    from nada_ai.mcp_server.apps.compare_app import compare_ref_areas

    app = await compare_ref_areas()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert d["state"]["ref_areas"] == ""
    assert "ref_areas_list" not in d["state"]


# ---------------------------------------------------------------------------
# summarize_app
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_app_renders():
    from nada_ai.mcp_server.apps.summarize_app import summarize_indicator

    app = await summarize_indicator()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


@pytest.mark.asyncio
async def test_summarize_app_with_args():
    from nada_ai.mcp_server.apps.summarize_app import summarize_indicator

    app = await summarize_indicator(idno="NY.GDP.MKTP.CD", period="2021")
    d = _get_json(app)
    assert d["state"]["idno"] == "NY.GDP.MKTP.CD"
    assert d["state"]["period"] == "2021"


# ---------------------------------------------------------------------------
# growth_app
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_growth_app_renders():
    from nada_ai.mcp_server.apps.growth_app import show_growth

    app = await show_growth()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


@pytest.mark.asyncio
async def test_growth_app_with_args():
    from nada_ai.mcp_server.apps.growth_app import show_growth

    app = await show_growth(idno="NY.GDP.MKTP.CD", base_period="2010", end_period="2022",
                            ref_areas="KEN,UGA")
    d = _get_json(app)
    assert d["state"]["idno"] == "NY.GDP.MKTP.CD"
    assert d["state"]["base_period"] == "2010"
    assert d["state"]["end_period"] == "2022"
    assert d["state"]["ref_areas"] == "KEN,UGA"
    assert "ref_areas_list" not in d["state"]


@pytest.mark.asyncio
async def test_growth_app_empty_ref_areas():
    from nada_ai.mcp_server.apps.growth_app import show_growth

    app = await show_growth(idno="X", base_period="2000", end_period="2010")
    d = _get_json(app)
    assert d["state"]["ref_areas"] == ""
    assert "ref_areas_list" not in d["state"]


# ---------------------------------------------------------------------------
# All apps produce serialisable JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("factory,kwargs", [
    ("nada_ai.mcp_server.search_app.search_catalog_ui", {}),
    ("nada_ai.mcp_server.apps.schema_app.explore_indicator_schema", {"idno": "TEST"}),
    ("nada_ai.mcp_server.apps.rank_app.rank_ref_areas", {"idno": "TEST", "period": "2020"}),
    ("nada_ai.mcp_server.apps.compare_app.compare_ref_areas", {"idno": "TEST", "ref_areas": "A,B"}),
    ("nada_ai.mcp_server.apps.summarize_app.summarize_indicator", {"idno": "TEST", "period": "2020"}),
    ("nada_ai.mcp_server.apps.growth_app.show_growth",
     {"idno": "TEST", "base_period": "2010", "end_period": "2020"}),
    ("nada_ai.mcp_server.apps.extremes_app.show_extremes", {"idno": "TEST"}),
])
async def test_app_to_json_is_dict(factory, kwargs):
    import importlib
    module_path, fn_name = factory.rsplit(".", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    result = fn(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    d = _get_json(result)
    assert isinstance(d, dict), f"{factory} returned non-dict from to_json()"
    assert "state" in d


# ---------------------------------------------------------------------------
# ref_areas live-binding fix — regression tests for all 6 affected apps
#
# Prior bug: each app snapshotted the parsed ref_areas list once into a
# separate "ref_areas_list" state key at initial render. The button action
# referenced that snapshot instead of the live "ref_areas" text input, so
# editing the field after the app opened had no effect on the next click.
# Fix: do_X now accepts the raw comma-separated string (same state key the
# Input writes to) and parses it internally on every call.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("module_path,fn_name,backing_fn,call_kwargs,expected", [
    ("nada_ai.mcp_server.apps.compare_app", "do_compare", "_nada_compare",
     {"idno": "X", "ref_areas": "KEN, UGA ,TZA"}, ["KEN", "UGA", "TZA"]),
    ("nada_ai.mcp_server.apps.join_app", "do_join", "_nada_join",
     {"idno1": "X", "idno2": "Y", "ref_areas": "KEN, UGA"}, ["KEN", "UGA"]),
    ("nada_ai.mcp_server.apps.benchmark_app", "do_benchmark", "_nada_benchmark",
     {"idno": "X", "period": "2022", "ref_areas": "KEN, UGA"}, ["KEN", "UGA"]),
    ("nada_ai.mcp_server.apps.trend_app", "do_trend", "_nada_trend",
     {"idno": "X", "ref_areas": "KEN, UGA"}, ["KEN", "UGA"]),
    ("nada_ai.mcp_server.apps.growth_app", "do_growth", "_nada_growth",
     {"idno": "X", "base_period": "2010", "end_period": "2020", "ref_areas": "KEN, UGA"}, ["KEN", "UGA"]),
    ("nada_ai.mcp_server.apps.aggregate_app", "do_aggregate", "_nada_aggregate",
     {"idno": "X", "ref_areas": "KEN, UGA"}, ["KEN", "UGA"]),
])
async def test_do_x_parses_live_ref_areas_string(module_path, fn_name, backing_fn, call_kwargs, expected):
    import importlib
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    with patch(f"{module_path}.{backing_fn}", new=AsyncMock(return_value="ok")) as mock_backing:
        await fn(**call_kwargs)
        _, kwargs = mock_backing.call_args
        assert kwargs["ref_areas"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("module_path,fn_name,backing_fn,call_kwargs", [
    ("nada_ai.mcp_server.apps.join_app", "do_join", "_nada_join", {"idno1": "X", "idno2": "Y"}),
    ("nada_ai.mcp_server.apps.trend_app", "do_trend", "_nada_trend", {"idno": "X"}),
    ("nada_ai.mcp_server.apps.growth_app", "do_growth", "_nada_growth",
     {"idno": "X", "base_period": "2010", "end_period": "2020"}),
    ("nada_ai.mcp_server.apps.aggregate_app", "do_aggregate", "_nada_aggregate", {"idno": "X"}),
])
async def test_do_x_treats_missing_ref_areas_as_none(module_path, fn_name, backing_fn, call_kwargs):
    """Optional ref_areas apps must pass None (not []) when the field is omitted,
    matching the underlying tool's 'default to all ref areas' semantics."""
    import importlib
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    with patch(f"{module_path}.{backing_fn}", new=AsyncMock(return_value="ok")) as mock_backing:
        await fn(**call_kwargs)
        _, kwargs = mock_backing.call_args
        assert kwargs["ref_areas"] is None


# ---------------------------------------------------------------------------
# extremes_app
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extremes_app_renders():
    from nada_ai.mcp_server.apps.extremes_app import show_extremes

    app = await show_extremes()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


@pytest.mark.asyncio
async def test_extremes_app_with_result():
    from nada_ai.nada.models import ExtremePoint, ExtremesResponse
    from nada_ai.mcp_server.apps.extremes_app import show_extremes

    fake = ExtremesResponse(
        idno="X", indicator_name="Test", geo_column="COUNTRY_CODE",
        time_column="TIME_PERIOD", obs_column="OBS_VALUE",
        maximum=ExtremePoint(ref_area="KEN", ref_area_label="Kenya", period="2020", value=100.0),
        minimum=ExtremePoint(ref_area="UGA", ref_area_label="Uganda", period="2010", value=1.0),
        total_observations=42,
    )
    with patch("nada_ai.mcp_server.apps.extremes_app._nada_extremes", new=AsyncMock(return_value=fake)):
        result = await show_extremes(idno="SP.POP.TOTL")
    d = _get_json(result)
    assert d["state"]["result"]["maximum"]["ref_area"] == "KEN"
    assert d["state"]["result"]["minimum"]["ref_area"] == "UGA"


# ---------------------------------------------------------------------------
# coverage_app — bar chart of coverage_pct
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coverage_app_renders_with_chart_data():
    from nada_ai.nada.models import CoverageResponse, CoverageSummary
    from nada_ai.mcp_server.apps.coverage_app import show_coverage

    fake = CoverageResponse(
        idno="X", indicator_name="Test", geo_column="COUNTRY_CODE", time_column="TIME_PERIOD",
        total_ref_areas=2, total_periods=10,
        rows=[
            CoverageSummary(ref_area="KEN", ref_area_label="Kenya", n_periods=8,
                             first_period="2010", last_period="2020", coverage_pct=80.0),
            CoverageSummary(ref_area="UGA", ref_area_label="Uganda", n_periods=5,
                             first_period="2012", last_period="2019", coverage_pct=50.0),
        ],
    )
    with patch("nada_ai.mcp_server.apps.coverage_app._nada_coverage", new=AsyncMock(return_value=fake)):
        result = await show_coverage(idno="SP.POP.TOTL")
    d = _get_json(result)
    assert d["state"]["result"]["rows"][0]["coverage_pct"] == 80.0


# ---------------------------------------------------------------------------
# correlate_app — scatter chart of value1 vs value2
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_correlate_app_renders_with_chart_data():
    from nada_ai.nada.models import CorrelatePoint, CorrelateResponse
    from nada_ai.mcp_server.apps.correlate_app import show_correlate

    fake = CorrelateResponse(
        idno1="X", idno2="Y", period="2020", pearson_r=0.87, n=2,
        rows=[
            CorrelatePoint(ref_area="KEN", ref_area_label="Kenya", value1=1.0, value2=2.0),
            CorrelatePoint(ref_area="UGA", ref_area_label="Uganda", value1=3.0, value2=4.0),
        ],
    )
    with patch("nada_ai.mcp_server.apps.correlate_app._nada_correlate", new=AsyncMock(return_value=fake)):
        result = await show_correlate(idno1="X", idno2="Y", period="2020")
    d = _get_json(result)
    assert d["state"]["result"]["pearson_r"] == 0.87
    assert d["state"]["result"]["rows"][0]["value1"] == 1.0
    assert d["state"]["result"]["rows"][0]["value2"] == 2.0
