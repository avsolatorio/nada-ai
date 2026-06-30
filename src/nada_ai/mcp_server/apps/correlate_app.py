"""CorrelateApp — Pearson correlation between two indicators for a period."""

from __future__ import annotations

import asyncio

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
from nada_ai.nada.models import CorrelateResponse
from nada_ai.mcp_server import analytics

correlate_app = FastMCPApp("Correlate")


@correlate_app.tool()
async def do_correlate(
    idno1: str,
    idno2: str,
    period: str,
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions1: dict[str, str] | None = None,
    dimensions2: dict[str, str] | None = None,
) -> CorrelateResponse:
    """Fetch both indicators and compute Pearson correlation across ref areas."""
    schema1_resp, schema2_resp = await asyncio.gather(
        nada_api.get_indicator_schema(idno1),
        nada_api.get_indicator_schema(idno2),
    )
    if schema1_resp.error or not schema1_resp.schema_:
        return CorrelateResponse(idno1=idno1, idno2=idno2, period=period,
                                 error=schema1_resp.error or "Schema unavailable for idno1")
    if schema2_resp.error or not schema2_resp.schema_:
        return CorrelateResponse(idno1=idno1, idno2=idno2, period=period,
                                 error=schema2_resp.error or "Schema unavailable for idno2")

    data1, data2 = await asyncio.gather(
        nada_api.get_all_timeseries_data(idno1, from_year=from_year, to_year=to_year, dimensions=dimensions1),
        nada_api.get_all_timeseries_data(idno2, from_year=from_year, to_year=to_year, dimensions=dimensions2),
    )
    if data1.error:
        return CorrelateResponse(idno1=idno1, idno2=idno2, period=period, error=data1.error)
    if data2.error:
        return CorrelateResponse(idno1=idno1, idno2=idno2, period=period, error=data2.error)

    try:
        return analytics.correlate(
            data1.data, schema1_resp.schema_, data2.data, schema2_resp.schema_,
            period=period, dimensions1=dimensions1, dimensions2=dimensions2,
        )
    except ValueError as exc:
        return CorrelateResponse(idno1=idno1, idno2=idno2, period=period, error=str(exc))


@correlate_app.ui(
    description=(
        "Open an interactive correlation UI between two timeseries indicators. "
        "Shows Pearson r and a scatter table of ref area values for a given period."
    ),
    title="Correlate Indicators",
)
def show_correlate(
    idno1: str = "",
    idno2: str = "",
    period: str = "",
) -> PrefabApp:
    """Render the correlation scatter table."""

    _action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_correlate,
            arguments={
                "idno1": "{{ idno1 }}",
                "idno2": "{{ idno2 }}",
                "period": "{{ period }}",
            },
            on_success=[SetState("result", "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", "Correlation failed."), SetState("loading", False)],
        ),
    ]

    with PrefabApp(
        title="Correlate Indicators",
        state={
            "idno1": idno1, "idno2": idno2, "period": str(period),
            "loading": False, "result": None, "error": None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Correlate Indicators", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator 1 idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno1", value="{{ idno1 }}")
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator 2 idno")
                Input(placeholder="e.g. NY.GDP.PCAP.CD", name="idno2", value="{{ idno2 }}")
            with Column(css_class="min-w-28", gap=1):
                Small("Period")
                Input(placeholder="e.g. 2022", name="period", value="{{ period }}")
            Button("Correlate", on_click=_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Computing correlation…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2 flex-wrap"):
                    Small("{{ result.indicator_name1 }}", css_class="text-muted-foreground")
                    Text("vs", css_class="text-muted-foreground")
                    Small("{{ result.indicator_name2 }}", css_class="text-muted-foreground")
                    Badge("{{ result.period }}")
                with Row(css_class="items-center gap-4 mb-2"):
                    Small("Pearson r: ", css_class="font-medium")
                    Badge("{{ result.pearson_r }}")
                    Small("n = {{ result.n }} ref areas", css_class="text-muted-foreground")

                DataTable(
                    columns=[
                        DataTableColumn(key="ref_area", header="Ref Area", sortable=True),
                        DataTableColumn(key="ref_area_label", header="Name", sortable=True),
                        DataTableColumn(key="value1", header="{{ result.idno1 }}",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="value2", header="{{ result.idno2 }}",
                                        sortable=True, format="number:2", align="right"),
                    ],
                    rows="{{ result.rows }}",
                    paginated=True,
                    page_size=15,
                    search=True,
                )

    return app
