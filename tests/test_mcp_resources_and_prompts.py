"""Tests for MCP resources, prompts, and tool trust/safety annotations."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

from nada_ai.mcp_server import mcp
from nada_ai.mcp_server.tool_config import get_mcp_tool_texts

_PREFIX = get_mcp_tool_texts().prefix


# ── tool annotations ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_all_analytical_tools_are_annotated_read_only():
    async with Client(mcp) as client:
        tools = await client.list_tools()

    analytical_tools = [t for t in tools if t.name.startswith(_PREFIX + "_")]
    assert len(analytical_tools) >= 17

    for tool in analytical_tools:
        assert tool.annotations is not None, f"{tool.name} has no annotations"
        assert tool.annotations.readOnlyHint is True, f"{tool.name} missing readOnlyHint"
        assert tool.annotations.destructiveHint is False, f"{tool.name} missing destructiveHint=False"
        assert tool.annotations.idempotentHint is True, f"{tool.name} missing idempotentHint"
        assert tool.annotations.openWorldHint is True, f"{tool.name} missing openWorldHint"


# ── resources ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_usage_resource_is_valid_json():
    async with Client(mcp) as client:
        result = await client.read_resource("nada://search-usage")
    payload = json.loads(result[0].text)
    assert "workflow" in payload
    assert "keyword_search" in payload


@pytest.mark.asyncio
async def test_analytics_workflow_resource_is_valid_json_and_lists_all_tools():
    async with Client(mcp) as client:
        result = await client.read_resource("nada://analytics-workflow")
    payload = json.loads(result[0].text)
    assert "workflow" in payload
    assert "tools" in payload

    expected_suffixes = [
        "rank", "extremes", "compare", "summarize", "growth", "correlate",
        "outliers", "trend", "benchmark", "coverage", "join", "aggregate",
    ]
    for suffix in expected_suffixes:
        tool_name = f"{_PREFIX}_{suffix}"
        assert tool_name in payload["tools"], f"{tool_name} missing from analytics-workflow resource"


@pytest.mark.asyncio
async def test_resources_are_discoverable_via_list():
    async with Client(mcp) as client:
        resources = await client.list_resources()
    uris = {str(r.uri) for r in resources}
    assert "nada://search-usage" in uris
    assert "nada://analytics-workflow" in uris


# ── prompts ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompts_are_discoverable_with_expected_arguments():
    async with Client(mcp) as client:
        prompts = await client.list_prompts()
    by_name = {p.name: p for p in prompts}

    assert f"{_PREFIX}_explore_indicator" in by_name
    assert [a.name for a in by_name[f"{_PREFIX}_explore_indicator"].arguments] == ["idno"]

    assert f"{_PREFIX}_compare_countries" in by_name
    assert {a.name for a in by_name[f"{_PREFIX}_compare_countries"].arguments} == {
        "idno", "ref_areas", "period",
    }

    assert f"{_PREFIX}_find_anomalies" in by_name
    anomaly_args = {a.name for a in by_name[f"{_PREFIX}_find_anomalies"].arguments}
    assert anomaly_args == {"idno", "period", "ref_area"}


@pytest.mark.asyncio
async def test_explore_indicator_prompt_renders_with_idno():
    async with Client(mcp) as client:
        result = await client.get_prompt(f"{_PREFIX}_explore_indicator", {"idno": "SP.POP.TOTL"})
    text = result.messages[0].content.text
    assert "SP.POP.TOTL" in text
    assert f"{_PREFIX}_get_schema" in text
    assert f"{_PREFIX}_coverage" in text
    assert f"{_PREFIX}_trend" in text


@pytest.mark.asyncio
async def test_compare_countries_prompt_splits_ref_areas():
    async with Client(mcp) as client:
        result = await client.get_prompt(
            f"{_PREFIX}_compare_countries",
            {"idno": "SP.POP.TOTL", "ref_areas": "KEN, UGA ,TZA", "period": "2022"},
        )
    text = result.messages[0].content.text
    assert "'KEN'" in text
    assert "'UGA'" in text
    assert "'TZA'" in text
    assert "2022" in text


@pytest.mark.asyncio
async def test_find_anomalies_prompt_cross_section_mode():
    async with Client(mcp) as client:
        result = await client.get_prompt(
            f"{_PREFIX}_find_anomalies", {"idno": "SP.POP.TOTL", "period": "2022"}
        )
    text = result.messages[0].content.text
    assert "cross-section" in text
    assert "period='2022'" in text


@pytest.mark.asyncio
async def test_find_anomalies_prompt_longitudinal_mode():
    async with Client(mcp) as client:
        result = await client.get_prompt(
            f"{_PREFIX}_find_anomalies", {"idno": "SP.POP.TOTL", "ref_area": "KEN"}
        )
    text = result.messages[0].content.text
    assert "longitudinal" in text
    assert "ref_area='KEN'" in text
