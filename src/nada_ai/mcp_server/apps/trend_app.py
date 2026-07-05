"""TrendApp — linear trend (OLS) per ref area over available periods."""

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

from nada_ai.nada.models import TrendResponse
from nada_ai.mcp_server.tools import _nada_trend
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

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
    return await _nada_trend(idno=idno, ref_areas=ref_areas,
                              from_year=from_year, to_year=to_year, dimensions=dimensions)


@trend_app.ui(
    description=(
        "Open an interactive trend analysis UI for a timeseries indicator. "
        "Shows linear slope, R², and direction (improving/declining/stable) per ref area."
    ),
    title="Trend Analysis",
)
async def show_trend(
    idno: str = "",
    ref_areas: str = "",
    from_year: str = "",
    to_year: str = "",
) -> PrefabApp:
    """Open the trend analysis UI, pre-loaded with data when idno is supplied.

    Args:
        idno: Indicator idno to analyse (e.g. SP.POP.TOTL). Pre-fills the form.
        ref_areas: Comma-separated ref area codes to include (e.g. KEN,UGA). Optional.
        from_year: Start year to narrow the trend window (e.g. 2000). Optional.
        to_year: End year to narrow the trend window (e.g. 2022). Optional.
    """
    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []
    fy = int(from_year) if from_year and from_year.isdigit() else None
    ty = int(to_year) if to_year and to_year.isdigit() else None
    result = None
    if idno:
        result = await do_trend(idno=idno, ref_areas=parsed_refs or None,
                                from_year=fy, to_year=ty)

    _action = make_action(
        do_trend,
        {"idno": "{{ idno }}", "ref_areas": "{{ ref_areas_list }}",
         "from_year": "{{ from_year }}", "to_year": "{{ to_year }}"},
        error_msg="Trend analysis failed.",
    )

    with PrefabApp(
        title="Trend Analysis",
        state={
            "idno": idno, "ref_areas": ref_areas, "ref_areas_list": parsed_refs,
            "from_year": from_year, "to_year": to_year,
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
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

    return ui_result(app, app_name="Trend", result=result,
                     params={"idno": idno, "ref_areas": ref_areas or None,
                             "from_year": from_year or None, "to_year": to_year or None})
