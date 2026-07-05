"""JoinApp — row-aligned merge of two indicators."""

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
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada.models import JoinResponse
from nada_ai.mcp_server.tools import _nada_join
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

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
    return await _nada_join(idno1=idno1, idno2=idno2, from_year=from_year, to_year=to_year,
                             ref_areas=ref_areas, dimensions1=dimensions1, dimensions2=dimensions2)


@join_app.ui(
    description=(
        "Open an interactive two-indicator join UI. "
        "Aligns both indicators by (ref_area, period) into a single merged table."
    ),
    title="Join Indicators",
)
async def show_join(
    idno1: str = "",
    idno2: str = "",
    from_year: str = "",
    to_year: str = "",
    ref_areas: str = "",
) -> PrefabApp:
    """Open the join UI, pre-loaded with data when both idnos are supplied.

    Args:
        idno1: First indicator idno (e.g. SP.POP.TOTL). Pre-fills the form.
        idno2: Second indicator idno (e.g. SP.POP.TOTL.FE.IN). Pre-fills the form.
        from_year: Start year filter (e.g. 2010). Optional.
        to_year: End year filter (e.g. 2022). Optional.
        ref_areas: Comma-separated ref area codes to include (e.g. KEN,UGA). Optional.
    """
    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []
    fy = int(from_year) if from_year and from_year.isdigit() else None
    ty = int(to_year) if to_year and to_year.isdigit() else None
    result = None
    if idno1 and idno2:
        result = await do_join(idno1=idno1, idno2=idno2, from_year=fy, to_year=ty,
                               ref_areas=parsed_refs or None)

    _action = make_action(
        do_join,
        {"idno1": "{{ idno1 }}", "idno2": "{{ idno2 }}", "from_year": "{{ from_year }}",
         "to_year": "{{ to_year }}", "ref_areas": "{{ ref_areas_list }}"},
        error_msg="Join failed.",
    )

    with PrefabApp(
        title="Join Indicators",
        state={
            "idno1": idno1, "idno2": idno2,
            "from_year": from_year, "to_year": to_year,
            "ref_areas": ref_areas, "ref_areas_list": parsed_refs,
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
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

    return ui_result(app, app_name="Join", result=result,
                     params={"idno1": idno1, "idno2": idno2,
                             "from_year": from_year or None, "to_year": to_year or None,
                             "ref_areas": ref_areas or None})
