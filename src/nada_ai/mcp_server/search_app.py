"""Catalog Search FastMCPApp — interactive UI entry point for the NADA catalog.

Provides a `FastMCPApp` with:
- ``@app.ui()`` entry point  : ``search_catalog_ui`` — renders the search form
- ``@app.tool()`` backend     : ``do_search`` — executes the actual catalog query
"""

from __future__ import annotations

from fastmcp import FastMCPApp
from prefab_ui.app import PrefabApp
from prefab_ui.actions import SetState
from prefab_ui.actions.mcp import CallTool
from prefab_ui.components import (
    Badge,
    Button,
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    Column,
    ForEach,
    H3,
    If,
    Input,
    Link,
    Loader,
    Muted,
    Row,
    Select,
    SelectOption,
    Separator,
    Small,
    Text,
)
from prefab_ui.rx import STATE

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import (
    CatalogDataAccessType,
    CatalogSearchRequest,
    CatalogSearchResponse,
    CatalogSortBy,
    CatalogSortOrder,
    CatalogStudyRow,
)
from nada_ai.mcp_server.tool_config import get_mcp_tool_texts

_TOOL_TEXTS = get_mcp_tool_texts()

search_app = FastMCPApp("CatalogSearch")

# ---------------------------------------------------------------------------
# Backend tool — called by the UI via CallTool
# ---------------------------------------------------------------------------

@search_app.tool()
async def do_search(
    keywords: str = "",
    type: str = "timeseries",
    from_year: int | None = None,
    to_year: int | None = None,
    country: str = "",
    page: int = 1,
    page_size: int = 15,
    sort_by: CatalogSortBy = "relevance",
    sort_order: CatalogSortOrder = "desc",
) -> CatalogSearchResponse:
    """Execute a catalog search and return paged results."""
    request = CatalogSearchRequest(
        keywords=keywords or None,
        type=type or "timeseries",
        from_year=from_year,
        to_year=to_year,
        country=country or None,
        page=page,
        page_size=page_size,
        sort_by=sort_by if keywords else "title",
        sort_order=sort_order,
        include_facets=True,
    )
    return await nada_api.search_catalog(request)


# ---------------------------------------------------------------------------
# UI entry point — the model calls this to open the search app
# ---------------------------------------------------------------------------

@search_app.ui(
    description=(
        f"Open an interactive catalog search UI for the {_TOOL_TEXTS.catalog_name}. "
        "Use this when the user wants to browse or search datasets interactively."
    ),
    title="Catalog Search",
)
def search_catalog_ui() -> PrefabApp:
    """Render the interactive catalog search interface."""

    _TYPE_OPTIONS = [
        ("timeseries", "Timeseries"),
        ("survey", "Survey / Microdata"),
        ("document", "Document"),
        ("geospatial", "Geospatial"),
        ("table", "Table"),
    ]

    _SORT_OPTIONS = [
        ("relevance", "Relevance"),
        ("title", "Title"),
        ("year", "Year"),
        ("popularity", "Popularity"),
        ("created", "Date created"),
        ("changed", "Last updated"),
    ]

    _search_action = [
        SetState("loading", True),
        SetState("results", None),
        SetState("error", None),
        CallTool(
            do_search,
            arguments={
                "keywords": "{{ keywords }}",
                "type": "{{ type }}",
                "country": "{{ country }}",
                "page": 1,
                "page_size": 15,
                "sort_by": "{{ sort_by }}",
                "sort_order": "desc",
            },
            on_success=[
                SetState("results", "{{ $result }}"),
                SetState("loading", False),
            ],
            on_error=[
                SetState("error", "Search failed. Please try again."),
                SetState("loading", False),
            ],
        ),
    ]

    with PrefabApp(
        title=f"{_TOOL_TEXTS.catalog_name} — Catalog Search",
        state={
            "keywords": "",
            "type": "timeseries",
            "country": "",
            "sort_by": "relevance",
            "loading": False,
            "results": None,
            "error": None,
        },
        css_class="p-4 max-w-4xl mx-auto",
    ) as app:

        H3(f"{_TOOL_TEXTS.catalog_name}", css_class="mb-4 text-xl font-semibold")

        # Search form
        with Card(css_class="mb-4"):
            with CardContent(css_class="pt-4"):
                with Column(gap=3):
                    # Keywords row
                    with Row(gap=2, css_class="items-end"):
                        with Column(css_class="flex-1", gap=1):
                            Small("Keywords")
                            Input(
                                placeholder="Search by topic, indicator, concept…",
                                name="keywords",
                                value="{{ keywords }}",
                            )
                        Button(
                            "Search",
                            on_click=_search_action,
                            css_class="shrink-0",
                        )

                    # Filters row
                    with Row(gap=2, css_class="flex-wrap"):
                        with Column(css_class="min-w-40", gap=1):
                            Small("Dataset type")
                            with Select(name="type", value="{{ type }}"):
                                for val, label in _TYPE_OPTIONS:
                                    SelectOption(value=val, label=label)

                        with Column(css_class="min-w-40", gap=1):
                            Small("Sort by")
                            with Select(name="sort_by", value="{{ sort_by }}"):
                                for val, label in _SORT_OPTIONS:
                                    SelectOption(value=val, label=label)

                        with Column(css_class="min-w-36", gap=1):
                            Small("Country")
                            Input(
                                placeholder="e.g. Kenya",
                                name="country",
                                value="{{ country }}",
                            )

        # Loading indicator
        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Searching…", css_class="ml-2")

        # Error state
        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        # Results
        with If(STATE.results):
            with Column(gap=2):
                # Summary bar
                with Row(css_class="items-center justify-between mb-1"):
                    Small(
                        "{{ results.count }} of {{ results.total_count }} results",
                        css_class="text-muted-foreground",
                    )
                    with If(STATE.results.has_more):
                        Small(
                            "Page {{ results.page }} of {{ results.total_pages }}",
                            css_class="text-muted-foreground",
                        )

                Separator()

                # Result cards
                with ForEach("results.items") as (idx, item):
                    with Card(css_class="hover:bg-muted/30 transition-colors"):
                        with CardContent(css_class="pt-3 pb-3"):
                            with Column(gap=1):
                                with Row(css_class="items-start justify-between gap-2"):
                                    with Column(css_class="flex-1", gap=0):
                                        with If(item.url):
                                            Link(
                                                item.title,
                                                href=item.url,
                                                css_class="font-medium text-sm hover:underline",
                                            )
                                        with If(~item.url):
                                            Text(item.title, css_class="font-medium text-sm")
                                    Badge(item.type, css_class="shrink-0 text-xs")

                                with If(item.abstract):
                                    Muted(
                                        item.abstract,
                                        css_class="text-xs line-clamp-2",
                                    )

                                with Row(gap=2, css_class="flex-wrap"):
                                    with If(item.nation):
                                        Small(item.nation, css_class="text-muted-foreground")
                                    with If(item.year_start):
                                        Small(
                                            "{{ item.year_start }}–{{ item.year_end }}",
                                            css_class="text-muted-foreground",
                                        )
                                    with If(item.idno):
                                        Small(item.idno, css_class="text-muted-foreground font-mono")

    return app
