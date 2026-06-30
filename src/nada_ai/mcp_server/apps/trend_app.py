"""TrendApp — linear trend (OLS) per ref area over available periods."""

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
from nada_ai.nada.models import TrendResponse
from nada_ai.mcp_server import analytics

trend_app = FastMCPApp("Trend")


@trend_app.tool()
async def do_trend(
    idno: str,
    ref_areas: list[str] | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> TrendResponse:
    """Fetch schema + data and compute linear trend per ref area."""
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return TrendResponse(idno=idno, error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_
    data = await nada_api.get_all_timeseries_data(
        idno, from_year=from_year, to_year=to_year,
        country_codes=ref_areas, geo_column=schema.geo_column or "COUNTRY_CODE",
        dimensions=dimensions,
    )
    if data.error:
        return TrendResponse(idno=idno, geo_column=schema.geo_column,
                             obs_column=schema.obs_column, error=data.error)
    try:
        return analytics.trend(data.data, schema, ref_areas=ref_areas, dimensions=dimensions)
    except ValueError as exc:
        return TrendResponse(idno=idno, geo_column=schema.geo_column,
                             obs_column=schema.obs_column, error=str(exc))


@trend_app.ui(
    description=(
        "Open an interactive trend analysis UI for a timeseries indicator. "
        "Shows linear slope, R², and direction (improving/declining/stable) per ref area."
    ),
    title="Trend Analysis",
)
def show_trend(
    idno: str = "",
    ref_areas: str = "",
    from_year: str = "",
    to_year: str = "",
) -> PrefabApp:
    """Render the trend analysis table."""

    _action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_trend,
            arguments={
                "idno": "{{ idno }}",
                "ref_areas": "{{ ref_areas_list }}",
                "from_year": "{{ from_year }}",
                "to_year": "{{ to_year }}",
            },
            on_success=[SetState("result", "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", "Trend analysis failed."), SetState("loading", False)],
        ),
    ]

    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []

    with PrefabApp(
        title="Trend Analysis",
        state={
            "idno": idno, "ref_areas": ref_areas, "ref_areas_list": parsed_refs,
            "from_year": from_year, "to_year": to_year,
            "loading": False, "result": None, "error": None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Trend Analysis", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="flex-1 min-w-40", gap=1):
                Small("Ref areas (optional)")
                Input(placeholder="e.g. KEN,UGA", name="ref_areas", value="{{ ref_areas }}")
            with Column(css_class="min-w-24", gap=1):
                Small("From year")
                Input(placeholder="e.g. 2010", name="from_year", value="{{ from_year }}")
            with Column(css_class="min-w-24", gap=1):
                Small("To year")
                Input(placeholder="e.g. 2022", name="to_year", value="{{ to_year }}")
            Button("Analyse", on_click=_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Fitting trends…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Badge("{{ result.rows | length }} ref areas")

                DataTable(
                    columns=[
                        DataTableColumn(key="ref_area", header="Code", sortable=True),
                        DataTableColumn(key="ref_area_label", header="Name", sortable=True),
                        DataTableColumn(key="direction", header="Direction", sortable=True),
                        DataTableColumn(key="slope", header="Slope",
                                        sortable=True, format="number:4", align="right"),
                        DataTableColumn(key="r_squared", header="R²",
                                        sortable=True, format="number:4", align="right"),
                        DataTableColumn(key="n_periods", header="Periods",
                                        sortable=True, align="right"),
                        DataTableColumn(key="first_period", header="From", sortable=True),
                        DataTableColumn(key="last_period", header="To", sortable=True),
                    ],
                    rows="{{ result.rows }}",
                    paginated=True,
                    page_size=15,
                    search=True,
                )

    return app
