"""Resources for the NADA MCP Server.

These resources provide static context to help LLMs understand the NADA system.
"""

import json

from ._server_definition import mcp

SEARCH_USAGE = {
    "keyword_search": {
        "example": "nada_search_catalog(keywords='population census', type='survey', page_size=10)",
        "note": "Full-text search with optional type filter",
    },
    "browse_by_type": {
        "example": "nada_search_catalog(type='timeseries', sort_by='popularity', sort_order='desc', page_size=15)",
        "note": "Browse without keywords; omit keywords for filtered catalog browse",
    },
    "with_facets": {
        "example": "nada_search_catalog(keywords='health', include_facets=True)",
        "note": "Include facet counts by dataset type, country, topic, etc.",
    },
    "pagination": {
        "example": "nada_search_catalog(keywords='Rwanda', page=2, page_size=15)",
        "note": "Check has_more and next_page in the response for additional pages",
    },
    "workflow": [
        "1. Search catalog with nada_search_catalog to find relevant studies",
        "2. Note idno and url from results for follow-up metadata or data access",
        "3. Use type filter (survey, document, timeseries) to narrow results",
    ],
}


@mcp.resource("nada://search-usage")
async def search_usage_resource() -> str:
    """Search tool usage guidance."""
    return json.dumps(SEARCH_USAGE, indent=2)
