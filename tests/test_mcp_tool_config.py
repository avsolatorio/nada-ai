"""Tests for configurable MCP tool names and descriptions."""

from nada_ai.mcp_server.tool_config import (
    is_allowed_mcp_tool_name,
    mcp_tool_name,
    normalize_tool_prefix,
    resolve_mcp_tool_texts,
)
from nada_ai.settings import MCPServerSettings


def test_default_tool_names_and_workflow_text():
    texts = resolve_mcp_tool_texts(MCPServerSettings())
    assert texts.search_tool_name == "nada_search_catalog"
    assert texts.get_metadata_tool_name == "nada_get_metadata"
    assert "STEP 1 of 2" in texts.search_description
    assert "STEP 2 of 2" in texts.get_metadata_description
    assert "nada_search_catalog" in texts.get_metadata_description
    assert "nada_get_metadata" in texts.search_description


def test_custom_prefix_and_catalog_name():
    texts = resolve_mcp_tool_texts(
        MCPServerSettings(
            tool_prefix="wdr",
            catalog_name="World Development Report catalog",
        )
    )
    assert texts.search_tool_name == "wdr_search_catalog"
    assert texts.get_metadata_tool_name == "wdr_get_metadata"
    assert "World Development Report catalog" in texts.search_description
    assert "wdr_search_catalog" in texts.get_metadata_description


def test_description_overrides_replace_defaults():
    texts = resolve_mcp_tool_texts(
        MCPServerSettings(
            search_catalog_description="Custom search instructions.",
            get_metadata_description="Custom metadata instructions.",
        )
    )
    assert texts.search_description == "Custom search instructions."
    assert texts.get_metadata_description == "Custom metadata instructions."


def test_is_allowed_mcp_tool_name_respects_prefix():
    assert is_allowed_mcp_tool_name("wdr_search_catalog", prefix="wdr")
    assert not is_allowed_mcp_tool_name("nada_search_catalog", prefix="wdr")


def test_normalize_tool_prefix_rejects_invalid_values():
    try:
        normalize_tool_prefix("123bad")
    except ValueError as exc:
        assert "prefix" in str(exc).lower()
    else:
        raise AssertionError("expected invalid prefix to raise")

    assert normalize_tool_prefix("NADA") == "nada"
    assert normalize_tool_prefix("  wdi_2026  ") == "wdi_2026"
    assert mcp_tool_name("wdi", "search_catalog") == "wdi_search_catalog"
