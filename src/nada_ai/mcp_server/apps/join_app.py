"""JoinApp — row-aligned merge of two indicators."""

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
)
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import JoinResponse
from nada_ai.mcp_server import analytics

join_app = FastMCPApp("Join")


@join_app.tool()
async def do_join(
    idno1: str,
    idno2: str,
    from_year: int | None = None,
    to_year: int | None = None,
    ref_areas: list[str] | None = None,
    dimensions1: dict[str, str] | None = None,
    dimensions2: dict[str, str] | None = None,
) -> JoinResponse:
    """Fetch both indicators and align them by (ref_area, period)."""
    schema1_resp, schema2_resp = await asyncio.gather(
        nada_api.get_indicator_schema(idno1),
        nada_api.get_indicator_schema(idno2),
    )
    if schema1_resp.error or not schema1_resp.schema_:
        return JoinResponse(idno1=idno1, idno2=idno2,
                            error=schema1_resp.error or "Schema unavailable for idno1")
    if schema2_resp.error or not schema2_resp.schema_:
        return JoinResponse(idno1=idno1, idno2=idno2,
                            error=schema2_resp.error or "Schema unavailable for idno2")

    schema1, schema2 = schema1_resp.schema_, schema2_resp.schema_
    data1, data2 = await asyncio.gather(
        nada_api.get_all_timeseries_data(
            idno1, from_year=from_year, to_year=to_year,
            country_codes=ref_areas, geo_column=schema1.geo_column or "COUNTRY_CODE",
            dimensions=dimensions1,
        ),
        nada_api.get_all_timeseries_data(
            idno2, from_year=from_year, to_year=to_year,
            country_codes=ref_areas, geo_column=schema2.geo_column or "COUNTRY_CODE",
            dimensions=dimensions2,
        ),
    )
    if data1.error:
        return JoinResponse(idno1=idno1, idno2=idno2, error=data1.error)
    if data2.error:
        return JoinResponse(idno1=idno1, idno2=idno2, error=data2.error)

    try:
        return analytics.join_indicators(
            data1.data, schema1, data2.data, schema2,
            dimensions1=dimensions1, dimensions2=dimensions2,
        )
    except ValueError as exc:
        return JoinResponse(idno1=idno1, idno2=idno2, error=str(exc))


@join_app.ui(
    description=(
        "Open an interactive two-indicator join UI. "
        "Aligns both indicators by (ref_area, period) into a single merged table."
    ),
    title="Join Indicators",
)
def show_join(
    idno1: str = "",
    idno2: str = "",
    from_year: str = "",
    to_year: str = "",
    ref_areas: str = "",
) -> PrefabApp:
    """Render the joined data table."""

    _action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_join,
            arguments={
                "idno1": "{{ idno1 }}",
                "idno2": "{{ idno2 }}",
                "from_year": "{{ from_year }}",
                "to_year": "{{ to_year }}",
                "ref_areas": "{{ ref_areas_list }}",
            },
            on_success=[SetState("result", "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", "Join failed."), SetState("loading", False)],
        ),
    ]

    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []

    with PrefabApp(
        title="Join Indicators",
        state={
            "idno1": idno1, "idno2": idno2,
            "from_year": from_year, "to_year": to_year,
            "ref_areas": ref_areas, "ref_areas_list": parsed_refs,
            "loading": False, "result": None, "error": None,
        },
        css_class="p-4 max-w-4xl mx-auto",
    ) as app:

        H3("Join Indicators", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator 1 idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno1", value="{{ idno1 }}")
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator 2 idno")
                Input(placeholder="e.g. NY.GDP.PCAP.CD", name="idno2", value="{{ idno2 }}")
            with Column(css_class="min-w-24", gap=1):
                Small("From year")
                Input(placeholder="e.g. 2010", name="from_year", value="{{ from_year }}")
            with Column(css_class="min-w-24", gap=1):
                Small("To year")
                Input(placeholder="e.g. 2022", name="to_year", value="{{ to_year }}")
            with Column(css_class="flex-1 min-w-40", gap=1):
                Small("Ref areas (optional)")
                Input(placeholder="e.g. KEN,UGA", name="ref_areas", value="{{ ref_areas }}")
            Button("Join", on_click=_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Joining indicators…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2 flex-wrap"):
                    Small("{{ result.indicator_name1 }}", css_class="text-muted-foreground")
                    Small("×", css_class="text-muted-foreground")
                    Small("{{ result.indicator_name2 }}", css_class="text-muted-foreground")
                    Badge("{{ result.n_matched }} matched rows")

                DataTable(
                    columns=[
                        DataTableColumn(key="ref_area", header="Ref Area", sortable=True),
                        DataTableColumn(key="ref_area_label", header="Name", sortable=True),
                        DataTableColumn(key="period", header="Period", sortable=True),
                        DataTableColumn(key="value1", header="{{ result.idno1 }}",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="value2", header="{{ result.idno2 }}",
                                        sortable=True, format="number:2", align="right"),
                    ],
                    rows="{{ result.rows }}",
                    paginated=True,
                    page_size=20,
                    search=True,
                )

    return app
