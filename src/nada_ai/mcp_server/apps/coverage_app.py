"""CoverageApp — data availability summary per ref area."""

from __future__ import annotations

from fastmcp import FastMCPApp
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
from prefab_ui.components.charts import BarChart, ChartSeries
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada.models import CoverageResponse
from nada_ai.mcp_server.tools import _nada_coverage
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

coverage_app = FastMCPApp("Coverage")


@coverage_app.tool()
async def do_coverage(
    idno: str,
    dimensions: dict[str, str] | None = None,
) -> CoverageResponse:
    """Fetch schema + data and compute coverage summary per ref area."""
    return await _nada_coverage(idno=idno, dimensions=dimensions)


@coverage_app.ui(
    description=(
        "Open an interactive data coverage UI for a timeseries indicator. "
        "Shows how many periods each ref area has data for, first/last periods, and coverage %."
    ),
    title="Data Coverage",
)
async def show_coverage(idno: str = "") -> PrefabApp:
    """Open the data coverage UI, pre-loaded with data when idno is supplied.

    Args:
        idno: Indicator idno to check coverage for (e.g. SP.POP.TOTL). Pre-fills the form.
    """
    result = None
    if idno:
        result = await do_coverage(idno=idno)

    _action = make_action(
        do_coverage,
        {"idno": "{{ idno }}"},
        error_msg="Coverage check failed.",
    )

    with PrefabApp(
        title="Data Coverage",
        state={
            "idno": idno, "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
        },
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

                # coverage_pct is always present regardless of how many ref areas
                # were returned, so a single fixed series is safe here (unlike a
                # per-ref-area comparison, which would need a dynamic series list).
                BarChart(
                    data="{{ result.rows }}",
                    series=[ChartSeries(data_key="coverage_pct", label="Coverage %")],
                    x_axis="ref_area",
                    horizontal=True,
                    show_legend=False,
                    value_format="number:1",
                    height=350,
                )

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

    return ui_result(app, app_name="Coverage", result=result, params={"idno": idno})
