"""CompareApp — side-by-side time series across multiple ref areas."""

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
    Text,
)
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import CompareResponse
from nada_ai.mcp_server import analytics

compare_app = FastMCPApp("Compare")


@compare_app.tool()
async def do_compare(
    idno: str,
    ref_areas: list[str],
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> CompareResponse:
    """Fetch schema + data and build a pivoted comparison."""
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return CompareResponse(idno=idno, geo_column=None, time_column=None, obs_column=None,
                               error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_
    data = await nada_api.get_all_timeseries_data(
        idno, from_year=from_year, to_year=to_year,
        country_codes=ref_areas, geo_column=schema.geo_column or "COUNTRY_CODE",
        dimensions=dimensions,
    )
    if data.error:
        return CompareResponse(idno=idno, geo_column=schema.geo_column,
                               time_column=schema.time_column,
                               obs_column=schema.obs_column, error=data.error)
    return analytics.compare(data.data, schema, ref_areas=ref_areas, dimensions=dimensions)


@compare_app.ui(
    description=(
        "Open an interactive time-series comparison UI for a timeseries indicator. "
        "Shows trend lines for multiple ref areas (countries, provinces, etc.) "
        "over a selected period range."
    ),
    title="Compare Ref Areas",
)
def compare_ref_areas(
    idno: str = "",
    ref_areas: str = "",
    from_year: str = "",
    to_year: str = "",
) -> PrefabApp:
    """Render the interactive comparison line chart + table."""

    _compare_action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_compare,
            arguments={
                "idno": "{{ idno }}",
                "ref_areas": "{{ ref_areas_list }}",
                "from_year": "{{ from_year }}",
                "to_year": "{{ to_year }}",
            },
            on_success=[
                SetState("result", "{{ $result }}"),
                SetState("loading", False),
            ],
            on_error=[
                SetState("error", "Comparison failed. Check the idno and ref area codes."),
                SetState("loading", False),
            ],
        ),
    ]

    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []

    with PrefabApp(
        title="Compare Ref Areas",
        state={
            "idno": idno,
            "ref_areas": ref_areas,
            "ref_areas_list": parsed_refs,
            "from_year": from_year,
            "to_year": to_year,
            "loading": False,
            "result": None,
            "error": None,
        },
        css_class="p-4 max-w-4xl mx-auto",
    ) as app:

        H3("Compare Ref Areas", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="flex-1 min-w-48", gap=1):
                Small("Ref areas (comma-separated codes)")
                Input(placeholder="e.g. KEN,UGA,TZA", name="ref_areas", value="{{ ref_areas }}")
            with Column(css_class="min-w-24", gap=1):
                Small("From year")
                Input(placeholder="e.g. 2010", name="from_year", value="{{ from_year }}")
            with Column(css_class="min-w-24", gap=1):
                Small("To year")
                Input(placeholder="e.g. 2022", name="to_year", value="{{ to_year }}")
            Button("Compare", on_click=_compare_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Loading comparison…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-1 flex-wrap"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Text(
                        "{{ result.ref_areas | join(', ') }}",
                        css_class="text-sm text-muted-foreground",
                    )

                # Use unpivoted rows so the table works regardless of which ref_areas were queried
                DataTable(
                    columns=[
                        DataTableColumn(key="period", header="Period", sortable=True),
                        DataTableColumn(key="ref_area", header="Ref Area", sortable=True),
                        DataTableColumn(key="value", header="Value",
                                        sortable=True, format="number:2", align="right"),
                    ],
                    rows="{{ result.rows_unpivoted }}",
                    paginated=True,
                    page_size=20,
                    search=True,
                )

    return app
