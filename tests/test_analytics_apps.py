"""Tests for UI FastMCPApps — verify apps render without error and produce valid PrefabApp."""

from __future__ import annotations

import pytest
from prefab_ui.app import PrefabApp


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

def test_schema_app_renders():
    from nada_ai.mcp_server.apps.schema_app import explore_indicator_schema

    app = explore_indicator_schema()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


def test_schema_app_with_idno():
    from nada_ai.mcp_server.apps.schema_app import explore_indicator_schema

    app = explore_indicator_schema(idno="SP.POP.TOTL")
    d = app.to_json()
    assert d["state"]["idno"] == "SP.POP.TOTL"


# ---------------------------------------------------------------------------
# rank_app
# ---------------------------------------------------------------------------

def test_rank_app_renders():
    from nada_ai.mcp_server.apps.rank_app import rank_ref_areas

    app = rank_ref_areas()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


def test_rank_app_with_args():
    from nada_ai.mcp_server.apps.rank_app import rank_ref_areas

    app = rank_ref_areas(idno="SP.POP.TOTL", period="2022", n=5)
    d = app.to_json()
    assert d["state"]["idno"] == "SP.POP.TOTL"
    assert d["state"]["period"] == "2022"
    assert d["state"]["n"] == 5


def test_rank_app_ascending_flag():
    from nada_ai.mcp_server.apps.rank_app import rank_ref_areas

    app = rank_ref_areas(idno="TEST", period="2020", ascending=True)
    d = app.to_json()
    assert d["state"]["ascending"] is True


# ---------------------------------------------------------------------------
# compare_app
# ---------------------------------------------------------------------------

def test_compare_app_renders():
    from nada_ai.mcp_server.apps.compare_app import compare_ref_areas

    app = compare_ref_areas()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


def test_compare_app_parses_ref_areas_list():
    from nada_ai.mcp_server.apps.compare_app import compare_ref_areas

    app = compare_ref_areas(idno="TEST", ref_areas="KEN,UGA,TZA")
    d = app.to_json()
    assert d["state"]["ref_areas"] == "KEN,UGA,TZA"
    # ref_areas_list should be the parsed list
    assert d["state"]["ref_areas_list"] == ["KEN", "UGA", "TZA"]


def test_compare_app_empty_ref_areas():
    from nada_ai.mcp_server.apps.compare_app import compare_ref_areas

    app = compare_ref_areas()
    d = app.to_json()
    assert d["state"]["ref_areas_list"] == []


# ---------------------------------------------------------------------------
# summarize_app
# ---------------------------------------------------------------------------

def test_summarize_app_renders():
    from nada_ai.mcp_server.apps.summarize_app import summarize_indicator

    app = summarize_indicator()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


def test_summarize_app_with_args():
    from nada_ai.mcp_server.apps.summarize_app import summarize_indicator

    app = summarize_indicator(idno="NY.GDP.MKTP.CD", period="2021")
    d = app.to_json()
    assert d["state"]["idno"] == "NY.GDP.MKTP.CD"
    assert d["state"]["period"] == "2021"


# ---------------------------------------------------------------------------
# growth_app
# ---------------------------------------------------------------------------

def test_growth_app_renders():
    from nada_ai.mcp_server.apps.growth_app import show_growth

    app = show_growth()
    assert isinstance(app, PrefabApp)
    d = app.to_json()
    assert "state" in d


def test_growth_app_with_args():
    from nada_ai.mcp_server.apps.growth_app import show_growth

    app = show_growth(idno="NY.GDP.MKTP.CD", base_period="2010", end_period="2022",
                      ref_areas="KEN,UGA")
    d = app.to_json()
    assert d["state"]["idno"] == "NY.GDP.MKTP.CD"
    assert d["state"]["base_period"] == "2010"
    assert d["state"]["end_period"] == "2022"
    assert d["state"]["ref_areas_list"] == ["KEN", "UGA"]


def test_growth_app_empty_ref_areas_parses_to_empty_list():
    from nada_ai.mcp_server.apps.growth_app import show_growth

    app = show_growth(idno="X", base_period="2000", end_period="2010")
    d = app.to_json()
    assert d["state"]["ref_areas_list"] == []


# ---------------------------------------------------------------------------
# All apps produce serialisable JSON
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory,kwargs", [
    ("nada_ai.mcp_server.search_app.search_catalog_ui", {}),
    ("nada_ai.mcp_server.apps.schema_app.explore_indicator_schema", {"idno": "TEST"}),
    ("nada_ai.mcp_server.apps.rank_app.rank_ref_areas", {"idno": "TEST", "period": "2020"}),
    ("nada_ai.mcp_server.apps.compare_app.compare_ref_areas", {"idno": "TEST", "ref_areas": "A,B"}),
    ("nada_ai.mcp_server.apps.summarize_app.summarize_indicator", {"idno": "TEST", "period": "2020"}),
    ("nada_ai.mcp_server.apps.growth_app.show_growth",
     {"idno": "TEST", "base_period": "2010", "end_period": "2020"}),
])
def test_app_to_json_is_dict(factory, kwargs):
    import importlib
    module_path, fn_name = factory.rsplit(".", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    app = fn(**kwargs)
    result = app.to_json()
    assert isinstance(result, dict), f"{factory} returned non-dict from to_json()"
    assert "state" in result
