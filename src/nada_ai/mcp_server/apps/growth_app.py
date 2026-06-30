"""GrowthApp — period-over-period change per ref area."""

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
from nada_ai.nada.models import GrowthResponse
from nada_ai.mcp_server import analytics

growth_app = FastMCPApp("Growth")


@growth_app.tool()
async def do_growth(
    idno: str,
    base_period: str,
    end_period: str,
    ref_areas: list[str] | None = None,
    dimensions: dict[str, str] | None = None,
) -> GrowthResponse:
    """Fetch schema + data and compute period-over-period change."""
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return GrowthResponse(idno=idno, base_period=base_period, end_period=end_period,
                              geo_column=None, obs_column=None,
                              error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_
    data = await nada_api.get_all_timeseries_data(
        idno, country_codes=ref_areas,
        geo_column=schema.geo_column or "COUNTRY_CODE",
        dimensions=dimensions,
    )
    if data.error:
        return GrowthResponse(idno=idno, base_period=base_period, end_period=end_period,
                              geo_column=schema.geo_column, obs_column=schema.obs_column,
                              error=data.error)
    return analytics.growth(data.data, schema, ref_areas=ref_areas,
                            base_period=base_period, end_period=end_period, dimensions=dimensions)


@growth_app.ui(
    description=(
        "Open an interactive period-over-period growth UI for a timeseries indicator. "
        "Shows absolute and percentage change per ref area between two periods."
    ),
    title="Growth / Change",
)
def show_growth(
    idno: str = "",
    base_period: str = "",
    end_period: str = "",
    ref_areas: str = "",
) -> PrefabApp:
    """Render the growth table."""

    _growth_action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_growth,
            arguments={
                "idno": "{{ idno }}",
                "base_period": "{{ base_period }}",
                "end_period": "{{ end_period }}",
                "ref_areas": "{{ ref_areas_list }}",
            },
            on_success=[
                SetState("result", "{{ $result }}"),
                SetState("loading", False),
            ],
            on_error=[
                SetState("error", "Growth calculation failed."),
                SetState("loading", False),
            ],
        ),
    ]

    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []

    with PrefabApp(
        title="Growth / Change",
        state={
            "idno": idno,
            "base_period": str(base_period),
            "end_period": str(end_period),
            "ref_areas": ref_areas,
            "ref_areas_list": parsed_refs,
            "loading": False,
            "result": None,
            "error": None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Growth / Change", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="min-w-28", gap=1):
                Small("Base period")
                Input(placeholder="e.g. 2010", name="base_period", value="{{ base_period }}")
            with Column(css_class="min-w-28", gap=1):
                Small("End period")
                Input(placeholder="e.g. 2022", name="end_period", value="{{ end_period }}")
            with Column(css_class="min-w-48", gap=1):
                Small("Ref areas (optional, comma-separated)")
                Input(placeholder="e.g. KEN,UGA", name="ref_areas", value="{{ ref_areas }}")
            Button("Calculate", on_click=_growth_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Calculating…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Badge("{{ result.base_period }} → {{ result.end_period }}")

                DataTable(
                    columns=[
                        DataTableColumn(key="ref_area", header="Code", sortable=True),
                        DataTableColumn(key="ref_area_label", header="Name", sortable=True),
                        DataTableColumn(key="base_value", header="Base value",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="end_value", header="End value",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="absolute_change", header="Abs. change",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="pct_change", header="% change",
                                        sortable=True, format="number:1", align="right"),
                    ],
                    rows="{{ result.rows }}",
                    paginated=True,
                    page_size=15,
                    search=True,
                )

    return app
