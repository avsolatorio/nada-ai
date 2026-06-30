"""MCP tool naming and LLM-facing descriptions (defaults + org overrides)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from nada_ai.settings import MCPServerSettings, get_mcp_server_settings

_TOOL_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class MCPToolTexts:
    """Resolved MCP tool names and descriptions for the active deployment."""

    prefix: str
    catalog_name: str
    search_tool_name: str
    get_metadata_tool_name: str
    search_description: str
    get_metadata_description: str


def normalize_tool_prefix(prefix: str) -> str:
    normalized = prefix.strip().lower()
    if not normalized or not _TOOL_PREFIX_RE.fullmatch(normalized):
        raise ValueError(
            "MCP tool prefix must start with a letter and contain only lowercase letters, digits, and underscores"
        )
    return normalized


def mcp_tool_name(prefix: str, suffix: str) -> str:
    return f"{normalize_tool_prefix(prefix)}_{suffix}"


def _format_default_search_description(*, prefix: str, catalog_name: str) -> str:
    get_metadata_tool = mcp_tool_name(prefix, "get_metadata")
    search_ui_tool = mcp_tool_name(prefix, "search_catalog_ui")
    return f"""Programmatic catalog search — returns raw JSON results for further processing. For interactive use, prefer {search_ui_tool} instead (opens a search UI the user can browse directly).

Use this tool when:
- You need to iterate over results programmatically (e.g. loop over idnos, extract fields, pass to other tools)
- You already know the user wants data processed rather than browsed
- You need facets, pagination control, or multiple pages in a single workflow

For all other cases — user wants to find or browse indicators, datasets, or surveys — call {search_ui_tool} with the user's keywords instead. It opens an interactive UI that pre-fills the search and shows the results immediately.

Each result includes `idno` (required for {get_metadata_tool}), `title`, and often `abstract`. Paginate with `page` / `page_size`; continue while `has_more` is true.

Query tips: `keywords` uses semantic search over titles, abstracts, definitions, and methodology text. Use the concept from the user's question — not filler like "data" or "survey". Narrow with `type`, `country`, `from_year`, `to_year`. Set `include_facets=true` when counts by type or country would help."""


def _format_default_get_metadata_description(*, prefix: str, catalog_name: str) -> str:
    search_tool = mcp_tool_name(prefix, "search_catalog")
    get_metadata_tool = mcp_tool_name(prefix, "get_metadata")
    return f"""STEP 2 of 2 — Fetch full metadata for one catalog item. Requires an `idno` from {search_tool}.

Use when you need the authoritative catalog record for a study, indicator, survey, document, or dataset — especially fields that search snippets do not include.

Rich content available (varies by `type`):
- Indicators / timeseries: `definition`, dimensions, methodology notes, coverage, source
- Documents: `abstract`, authorship, scope
- Surveys / microdata: sampling, collection methodology, geographic and temporal coverage
- All types: `title`, `abstract`, `year_start` / `year_end`, `authoring_entity`, links, typed `metadata` block

When to call:
- You have a concrete `idno` from a prior {search_tool} result (`items[].idno`)
- The user asks what an indicator means, how it is measured, or how a dataset was produced — after {search_tool} finds the best match
- You need methodology, coverage, producers, or document details to answer the question from catalog metadata

When NOT to call:
- You do not have an `idno` yet → call {search_tool} first and pick the best match
- The user only wants to browse or compare many datasets → stay on {search_tool}
- Never invent or guess an `idno`; it must come from search results or explicit user input

Workflow: SECOND. For definition or methodology questions: {search_tool} with a concept-focused `keywords` query → {get_metadata_tool} with the chosen `idno` → answer from returned fields when present (prefer catalog text over general knowledge). Pass the exact `idno` string from search results."""


def resolve_mcp_tool_texts(settings: MCPServerSettings | None = None) -> MCPToolTexts:
    settings = settings or get_mcp_server_settings()
    prefix = normalize_tool_prefix(settings.tool_prefix)
    catalog_name = settings.catalog_name.strip() or "NADA catalog"
    search_tool_name = mcp_tool_name(prefix, "search_catalog")
    get_metadata_tool_name = mcp_tool_name(prefix, "get_metadata")

    if settings.search_catalog_description:
        search_description = settings.search_catalog_description
    else:
        search_description = _format_default_search_description(prefix=prefix, catalog_name=catalog_name)

    if settings.get_metadata_description:
        get_metadata_description = settings.get_metadata_description
    else:
        get_metadata_description = _format_default_get_metadata_description(
            prefix=prefix,
            catalog_name=catalog_name,
        )

    return MCPToolTexts(
        prefix=prefix,
        catalog_name=catalog_name,
        search_tool_name=search_tool_name,
        get_metadata_tool_name=get_metadata_tool_name,
        search_description=search_description,
        get_metadata_description=get_metadata_description,
    )


@lru_cache(maxsize=1)
def get_mcp_tool_texts() -> MCPToolTexts:
    return resolve_mcp_tool_texts()


def clear_tool_texts_cache() -> None:
    """Clear cached MCPToolTexts. Call after changing MCP_TOOL_PREFIX in tests."""
    get_mcp_tool_texts.cache_clear()


def is_allowed_mcp_tool_name(tool_name: str, *, prefix: str | None = None) -> bool:
    active_prefix = normalize_tool_prefix(prefix or get_mcp_server_settings().tool_prefix)
    return tool_name.startswith(f"{active_prefix}_")
