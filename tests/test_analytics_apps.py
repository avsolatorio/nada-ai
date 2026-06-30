"""Tests for UI FastMCPApps — verify apps render without error and produce valid PrefabApp."""

from __future__ import annotations

import inspect

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

def test_search_app_renders():
    from nada_ai.mcp_server.search_app import search_catalog_ui

    app = search_catalog_ui()
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
async def test_compare_app_parses_ref_areas_list():
    from nada_ai.mcp_server.apps.compare_app import compare_ref_areas

    app = await compare_ref_areas(idno="TEST", ref_areas="KEN,UGA,TZA")
    d = _get_json(app)
    assert d["state"]["ref_areas"] == "KEN,UGA,TZA"
    assert d["state"]["ref_areas_list"] == ["KEN", "UGA", "TZA"]


@pytest.mark.asyncio
async def test_compare_app_empty_ref_areas():
    from nada_ai.mcp_server.apps.compare_app import compare_ref_areas

    app = await compare_ref_areas()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert d["state"]["ref_areas_list"] == []


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
    assert d["state"]["ref_areas_list"] == ["KEN", "UGA"]


@pytest.mark.asyncio
async def test_growth_app_empty_ref_areas_parses_to_empty_list():
    from nada_ai.mcp_server.apps.growth_app import show_growth

    app = await show_growth(idno="X", base_period="2000", end_period="2010")
    d = _get_json(app)
    assert d["state"]["ref_areas_list"] == []


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
