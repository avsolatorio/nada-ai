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
