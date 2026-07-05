"""SchemaExplorerApp — interactive DSD schema viewer for a timeseries indicator."""

from __future__ import annotations

from fastmcp import FastMCPApp
from prefab_ui.actions.mcp import SendMessage
from prefab_ui.app import PrefabApp
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
    Loader,
    Muted,
    Row,
    Separator,
    Small,
    Text,
)
from prefab_ui.rx import STATE

from nada_ai.nada.models import IndicatorSchemaResponse
from nada_ai.mcp_server.tools import _nada_get_schema
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

schema_app = FastMCPApp("SchemaExplorer")

_COLUMN_TYPE_VARIANT: dict[str, str] = {
    "geography": "default",
    "time_period": "default",
    "observation_value": "default",
    "attribute": "secondary",
    "periodicity": "secondary",
    "indicator_id": "secondary",
}


@schema_app.tool()
async def do_get_schema(idno: str) -> IndicatorSchemaResponse:
    """Fetch and return the DSD schema for an indicator."""
    return await _nada_get_schema(idno)


@schema_app.ui(
    description=(
        "Open an interactive Data Structure Definition (DSD) explorer for a timeseries indicator. "
        "Shows all column names, their structural roles, codelist IDs, time period format, "
        "and reporting year bounds. Use this to discover disaggregation dimensions and valid "
        "filter values before calling analytical tools."
    ),
    title="Schema Explorer",
)
async def explore_indicator_schema(idno: str = "") -> PrefabApp:
    """Open the DSD schema explorer, pre-loaded with schema data when idno is supplied.

    Args:
        idno: Indicator idno to explore (e.g. SP.POP.TOTL). Pre-fills the form and loads the schema.
    """
    result = None
    if idno:
        result = await do_get_schema(idno=idno)

    _load_action = make_action(
        do_get_schema,
        {"idno": "{{ idno }}"},
        result_key="schema_data",
        error_msg="Failed to load schema.",
    )

    with PrefabApp(
        title="DSD Schema Explorer",
        state={
            "idno": idno,
            "loading": False,
            "schema_data": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("DSD Schema Explorer", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 items-end"):
            with Column(css_class="flex-1", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            Button("Load Schema", on_click=_load_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Loading schema…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.schema_data):
            with Column(gap=3):
                # Header info
                with Card():
                    with CardContent(css_class="pt-3 pb-3"):
                        with Row(gap=4, css_class="flex-wrap"):
                            with Column(gap=0):
                                Small("Indicator", css_class="text-muted-foreground")
                                Text("{{ schema_data.schema.idno }}", css_class="font-mono text-sm font-medium")
                            with Column(gap=0):
                                Small("Time format", css_class="text-muted-foreground")
                                Text("{{ schema_data.schema.time_period_format }}", css_class="font-mono text-sm")
                            with Column(gap=0):
                                Small("Year range", css_class="text-muted-foreground")
                                Text(
                                    "{{ schema_data.schema.reporting_year_bounds.min }}"
                                    " – {{ schema_data.schema.reporting_year_bounds.max }}",
                                    css_class="text-sm",
                                )

                Separator()

                # Dimension columns callout
                with If(STATE.schema_data.schema.dimension_columns):
                    with Card(css_class="border-amber-200 bg-amber-50 dark:bg-amber-950/20"):
                        with CardContent(css_class="pt-3 pb-3"):
                            Small(
                                "This indicator has disaggregation dimensions — filter them "
                                "before calling analytical tools.",
                                css_class="text-amber-700 dark:text-amber-400",
                            )

                # Components table
                with Card():
                    with CardHeader(css_class="pb-2"):
                        CardTitle("Components", css_class="text-base")
                    with CardContent(css_class="pt-0"):
                        with Column(gap=1):
                            with ForEach("schema_data.schema.components") as (_, comp):
                                with Row(css_class="items-center justify-between py-1 border-b last:border-0", gap=2):
                                    with Column(gap=0, css_class="flex-1"):
                                        Text(comp.name, css_class="font-mono text-xs font-medium")
                                        with If(comp.label):
                                            Muted(comp.label, css_class="text-xs")
                                    with Row(gap=1):
                                        Badge(comp.column_type, css_class="text-xs shrink-0")
                                        with If(comp.codelist_id):
                                            Badge(
                                                "codelist: {{ comp.codelist_id }}",
                                                css_class="text-xs shrink-0",
                                            )

                # Send to LLM button
                Button(
                    "Use this schema in chat",
                    on_click=SendMessage(
                        "The schema for {{ schema_data.schema.idno }} is loaded. "
                        "Geography column: {{ schema_data.schema.geo_column }}, "
                        "time column: {{ schema_data.schema.time_column }}, "
                        "obs column: {{ schema_data.schema.obs_column }}. "
                        "Dimension columns: {{ schema_data.schema.dimension_columns }}. "
                        "Year range: {{ schema_data.schema.reporting_year_bounds.min }}"
                        "–{{ schema_data.schema.reporting_year_bounds.max }}."
                    ),
                    css_class="w-full mt-2",
                )

    return ui_result(app, app_name="SchemaExplorer", result=result, params={"idno": idno})
