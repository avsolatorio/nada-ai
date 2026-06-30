"""CoverageApp — data availability summary per ref area."""

from __future__ import annotations

from fastmcp import FastMCPApp
from prefab_ui.actions import SetState
from prefab_ui.actions.mcp import CallTool
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Badge,
    Button,
    Column,
    H3,
    If,
    Input,
    Loader,
    Muted,
    Row,
    Small,
)
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import CoverageResponse
from nada_ai.mcp_server import analytics

coverage_app = FastMCPApp("Coverage")


@coverage_app.tool()
async def do_coverage(
    idno: str,
    dimensions: dict[str, str] | None = None,
) -> CoverageResponse:
    """Fetch schema + data and compute coverage summary per ref area."""
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return CoverageResponse(idno=idno, error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_
    data = await nada_api.get_all_timeseries_data(idno, dimensions=dimensions)
    if data.error:
        return CoverageResponse(idno=idno, geo_column=schema.geo_column,
                                time_column=schema.time_column, error=data.error)
    try:
        return analytics.coverage(data.data, schema, dimensions=dimensions)
    except ValueError as exc:
        return CoverageResponse(idno=idno, geo_column=schema.geo_column,
                                time_column=schema.time_column, error=str(exc))


@coverage_app.ui(
    description=(
        "Open an interactive data coverage UI for a timeseries indicator. "
        "Shows how many periods each ref area has data for, first/last periods, and coverage %."
    ),
    title="Data Coverage",
)
def show_coverage(idno: str = "") -> PrefabApp:
    """Render the coverage summary table."""

    _action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_coverage,
            arguments={"idno": "{{ idno }}"},
            on_success=[SetState("result", "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", "Coverage check failed."), SetState("loading", False)],
        ),
    ]

    with PrefabApp(
        title="Data Coverage",
        state={"idno": idno, "loading": False, "result": None, "error": None},
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Data Coverage", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            Button("Check Coverage", on_click=_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Analysing coverage…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2 flex-wrap"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Badge("{{ result.total_ref_areas }} ref areas")
                    Badge("{{ result.total_periods }} distinct periods")

                DataTable(
                    columns=[
                        DataTableColumn(key="ref_area", header="Code", sortable=True),
                        DataTableColumn(key="ref_area_label", header="Name", sortable=True),
                        DataTableColumn(key="n_periods", header="Periods",
                                        sortable=True, align="right"),
                        DataTableColumn(key="coverage_pct", header="Coverage %",
                                        sortable=True, format="number:1", align="right"),
                        DataTableColumn(key="first_period", header="First", sortable=True),
                        DataTableColumn(key="last_period", header="Last", sortable=True),
                    ],
                    rows="{{ result.rows }}",
                    paginated=True,
                    page_size=20,
                    search=True,
                )

    return app
