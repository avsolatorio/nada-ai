"""Resources for the NADA MCP Server.

These resources provide static context to help LLMs understand the NADA system.
"""

import json

from ._server_definition import mcp
from .tool_config import get_mcp_tool_texts


def _search_usage_payload() -> dict:
    texts = get_mcp_tool_texts()
    search_tool = texts.search_tool_name
    get_metadata_tool = texts.get_metadata_tool_name
    return {
        "catalog_name": texts.catalog_name,
        "workflow": [
            f"1. Call {search_tool} first to discover datasets and obtain an `idno`",
            f"2. Call {get_metadata_tool} with that `idno` for full study metadata",
            "3. Use the `type` filter on search (survey, document, timeseries, etc.) to narrow results",
            "4. For definitions or methodology: search by concept → get_metadata on best match → answer from catalog fields",
        ],
        "keyword_search": {
            "example": f"{search_tool}(keywords='population census', type='survey', page_size=10)",
            "note": "Semantic keyword search over titles, abstracts, definitions, and methodology; prefer sort_by='relevance' when keywords are set",
        },
        "browse_by_type": {
            "example": f"{search_tool}(type='timeseries', sort_by='popularity', sort_order='desc', page_size=15)",
            "note": "Browse without keywords; omit keywords for filtered catalog browse",
        },
        "with_facets": {
            "example": f"{search_tool}(keywords='health', include_facets=True)",
            "note": "Include facet counts by dataset type, country, topic, etc.",
        },
        "pagination": {
            "example": f"{search_tool}(keywords='Rwanda', page=2, page_size=15)",
            "note": "Check has_more and next_page in the response for additional pages",
        },
        "metadata_follow_up": {
            "example": f"{get_metadata_tool}(idno='SI.POV.DDAY')",
            "note": f"Requires `idno` from a prior {search_tool} result",
        },
        "definition_or_methodology": {
            "example": f"{search_tool}(keywords='prevalence of stunting definition', sort_by='relevance') → {get_metadata_tool}(idno='<from search>')",
            "note": "Search by concept, then fetch full metadata to read definition, methodology, abstract, and related fields",
        },
    }


@mcp.resource("nada://search-usage")
async def search_usage_resource() -> str:
    """Search tool usage guidance."""
    return json.dumps(_search_usage_payload(), indent=2)


def _analytics_workflow_payload() -> dict:
    texts = get_mcp_tool_texts()
    prefix = texts.prefix
    schema_tool = f"{prefix}_get_schema"
    codelist_tool = f"{prefix}_get_codelist"
    return {
        "catalog_name": texts.catalog_name,
        "workflow": [
            f"1. Call {schema_tool} first for any timeseries `idno` — every analytical tool below "
            "assumes you already know the column names and available dimensions.",
            f"2. If the indicator has disaggregation dimensions (SEX, AGE_GROUP, etc.), call "
            f"{codelist_tool} to discover valid values before filtering with `dimensions`.",
            "3. Pick the analytical tool that matches the question (table below), pass the "
            "`idno` plus any `dimensions`/`ref_areas`/`period` the tool requires.",
            "4. All analytical tools auto-paginate the underlying data fetch — no manual "
            "pagination needed once you're past step 1.",
        ],
        "tools": {
            f"{prefix}_rank": "Top/bottom-N ref areas for one period — 'which countries had the highest X in 2022?'",
            f"{prefix}_extremes": "Global max/min across all periods and ref areas — 'what was the highest X ever recorded?'",
            f"{prefix}_compare": "Pivoted time series for a set of ref areas — trend comparison across countries.",
            f"{prefix}_summarize": "Descriptive stats (min/max/mean/median/std) across ref areas for one period.",
            f"{prefix}_growth": "Period-over-period absolute and percentage change per ref area.",
            f"{prefix}_correlate": "Pearson correlation between two indicators — 'does X correlate with Y?'",
            f"{prefix}_outliers": "Z-score / IQR / trend-residual outlier detection, cross-section or longitudinal.",
            f"{prefix}_trend": "Linear regression per ref area — slope, R², improving/declining/stable.",
            f"{prefix}_benchmark": "Percentile rank and deviation from peer mean/median for specific ref areas.",
            f"{prefix}_coverage": "Data availability per ref area — periods covered, gaps, coverage %.",
            f"{prefix}_join": "Align two indicators by (ref_area, period) into one merged table.",
            f"{prefix}_aggregate": "Group-level statistics (mean/median/total/min/max/std) per period for a custom ref area set.",
        },
        "dimensions_filter": {
            "example": "dimensions={'SEX': 'F'}",
            "note": f"Only valid after confirming the code with {codelist_tool} — do not guess dimension values.",
        },
    }


@mcp.resource("nada://analytics-workflow")
async def analytics_workflow_resource() -> str:
    """Analytical tool catalog and the schema-first workflow they share."""
    return json.dumps(_analytics_workflow_payload(), indent=2)
