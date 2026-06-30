"""AggregateApp — group-level statistics per period for a custom ref area set."""

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
from prefab_ui.components.charts import ChartSeries, LineChart
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import AggregateResponse
from nada_ai.mcp_server import analytics

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
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return AggregateResponse(idno=idno, error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_
    data = await nada_api.get_all_timeseries_data(
        idno, from_year=from_year, to_year=to_year,
        country_codes=ref_areas, geo_column=schema.geo_column or "COUNTRY_CODE",
        dimensions=dimensions,
    )
    if data.error:
        return AggregateResponse(idno=idno, geo_column=schema.geo_column,
                                 obs_column=schema.obs_column, error=data.error)
    try:
        return analytics.aggregate(data.data, schema, ref_areas=ref_areas, dimensions=dimensions)
    except ValueError as exc:
        return AggregateResponse(idno=idno, geo_column=schema.geo_column,
                                 obs_column=schema.obs_column, error=str(exc))


@aggregate_app.ui(
    description=(
        "Open an interactive aggregate UI for a timeseries indicator. "
        "Shows mean, median, total, min, and max per period across a custom group of ref areas."
    ),
    title="Aggregate",
)
def show_aggregate(
    idno: str = "",
    ref_areas: str = "",
    from_year: str = "",
    to_year: str = "",
) -> PrefabApp:
    """Render the aggregate time-series chart and stats table."""

    _action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_aggregate,
            arguments={
                "idno": "{{ idno }}",
                "ref_areas": "{{ ref_areas_list }}",
                "from_year": "{{ from_year }}",
                "to_year": "{{ to_year }}",
            },
            on_success=[SetState("result", "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", "Aggregation failed."), SetState("loading", False)],
        ),
    ]

    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []

    with PrefabApp(
        title="Aggregate",
        state={
            "idno": idno, "ref_areas": ref_areas, "ref_areas_list": parsed_refs,
            "from_year": from_year, "to_year": to_year,
            "loading": False, "result": None, "error": None,
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

    return app
