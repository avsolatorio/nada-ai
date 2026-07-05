"""AggregateApp — group-level statistics per period for a custom ref area set."""

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
from prefab_ui.components.charts import ChartSeries, LineChart
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada.models import AggregateResponse
from nada_ai.mcp_server.tools import _nada_aggregate
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

aggregate_app = FastMCPApp("Aggregate")


@aggregate_app.tool()
async def do_aggregate(
    idno: str,
    ref_areas: list[str] | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> AggregateResponse:
    """Fetch schema + data and compute group-level statistics per period."""
    return await _nada_aggregate(idno=idno, ref_areas=ref_areas,
                                  from_year=from_year, to_year=to_year, dimensions=dimensions)


@aggregate_app.ui(
    description=(
        "Open an interactive aggregate UI for a timeseries indicator. "
        "Shows mean, median, total, min, and max per period across a custom group of ref areas."
    ),
    title="Aggregate",
)
async def show_aggregate(
    idno: str = "",
    ref_areas: str = "",
    from_year: str = "",
    to_year: str = "",
) -> PrefabApp:
    """Open the aggregate UI, pre-loaded with data when idno is supplied.

    Args:
        idno: Indicator idno to aggregate (e.g. SP.POP.TOTL). Pre-fills the form.
        ref_areas: Comma-separated ref area codes to aggregate (e.g. KEN,UGA). Optional; default is all.
        from_year: Start year filter (e.g. 2000). Optional.
        to_year: End year filter (e.g. 2022). Optional.
    """
    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []
    fy = int(from_year) if from_year and from_year.isdigit() else None
    ty = int(to_year) if to_year and to_year.isdigit() else None
    result = None
    if idno:
        result = await do_aggregate(idno=idno, ref_areas=parsed_refs or None,
                                    from_year=fy, to_year=ty)

    _action = make_action(
        do_aggregate,
        {"idno": "{{ idno }}", "ref_areas": "{{ ref_areas_list }}",
         "from_year": "{{ from_year }}", "to_year": "{{ to_year }}"},
        error_msg="Aggregation failed.",
    )

    with PrefabApp(
        title="Aggregate",
        state={
            "idno": idno, "ref_areas": ref_areas, "ref_areas_list": parsed_refs,
            "from_year": from_year, "to_year": to_year,
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
        },
        css_class="p-4 max-w-4xl mx-auto",
    ) as app:

        H3("Aggregate", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="flex-1 min-w-48", gap=1):
                Small("Ref areas (comma-separated, empty = all)")
                Input(placeholder="e.g. KEN,UGA,TZA", name="ref_areas", value="{{ ref_areas }}")
            with Column(css_class="min-w-24", gap=1):
                Small("From year")
                Input(placeholder="e.g. 2000", name="from_year", value="{{ from_year }}")
            with Column(css_class="min-w-24", gap=1):
                Small("To year")
                Input(placeholder="e.g. 2022", name="to_year", value="{{ to_year }}")
            Button("Aggregate", on_click=_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Aggregating…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-1 flex-wrap"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Badge("{{ result.rows | length }} periods")

                # Mean + median trend lines — series keys are fixed so the chart works reactively
                LineChart(
                    data="{{ result.rows }}",
                    series=[
                        ChartSeries(data_key="mean", label="Mean"),
                        ChartSeries(data_key="median", label="Median"),
                    ],
                    x_axis="period",
                    curve="smooth",
                    show_dots=False,
                    value_format="compact",
                    height=300,
                )

                DataTable(
                    columns=[
                        DataTableColumn(key="period", header="Period", sortable=True),
                        DataTableColumn(key="n_ref_areas", header="N",
                                        sortable=True, align="right"),
                        DataTableColumn(key="mean", header="Mean",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="median", header="Median",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="total", header="Total",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="min_value", header="Min",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="max_value", header="Max",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="std", header="Std",
                                        sortable=True, format="number:2", align="right"),
                    ],
                    rows="{{ result.rows }}",
                    paginated=True,
                    page_size=20,
                )

    return ui_result(app, app_name="Aggregate", result=result,
                     params={"idno": idno, "ref_areas": ref_areas or None,
                             "from_year": from_year or None, "to_year": to_year or None})
