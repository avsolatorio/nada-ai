"""RankApp — top/bottom N ref areas for an indicator in a given period."""

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
    Select,
    SelectOption,
    Small,
    Switch,
)
from prefab_ui.components.charts import BarChart, ChartSeries
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada.models import RankResponse
from nada_ai.mcp_server.tools import _nada_rank
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

rank_app = FastMCPApp("Rank")


@rank_app.tool()
async def do_rank(
    idno: str,
    period: str,
    n: int = 10,
    ascending: bool = False,
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> RankResponse:
    """Fetch schema + data and compute ranked ref areas."""
    return await _nada_rank(idno=idno, period=period, n=n, ascending=ascending,
                            from_year=from_year, to_year=to_year, dimensions=dimensions)


@rank_app.ui(
    description=(
        "Open an interactive ranking UI for a timeseries indicator. "
        "Shows the top or bottom N ref areas (countries, provinces, etc.) "
        "sorted by observation value for a given period."
    ),
    title="Rank Ref Areas",
)
async def rank_ref_areas(
    idno: str = "",
    period: str = "",
    n: int = 10,
    ascending: bool = False,
) -> PrefabApp:
    """Open the ranking UI, pre-loaded with data when idno and period are supplied.

    Args:
        idno: Indicator idno to rank (e.g. SP.POP.TOTL). Pre-fills the form.
        period: Time period to rank within (e.g. 2022). Pre-fills the form.
        n: Number of top/bottom ref areas to show (default 10).
        ascending: If True show bottom-N (lowest first). Default False (top-N).
    """
    result = None
    if idno and period:
        result = await do_rank(idno=idno, period=period, n=n, ascending=ascending,
                               from_year=int(period) if period.isdigit() else None,
                               to_year=int(period) if period.isdigit() else None)

    _search_action = make_action(
        do_rank,
        {"idno": "{{ idno }}", "period": "{{ period }}", "n": "{{ n }}", "ascending": "{{ ascending }}"},
        error_msg="Ranking failed. Please check the idno and period.",
    )

    with PrefabApp(
        title="Rank Ref Areas",
        state={
            "idno": idno,
            "period": str(period),
            "n": n,
            "ascending": ascending,
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Rank Ref Areas", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="min-w-28", gap=1):
                Small("Period")
                Input(placeholder="e.g. 2022", name="period", value="{{ period }}")
            with Column(css_class="min-w-20", gap=1):
                Small("Top N")
                with Select(name="n", value="{{ n }}"):
                    for v in [5, 10, 20, 50]:
                        SelectOption(value=str(v), label=str(v))
            with Column(gap=1):
                Small("Order")
                with Select(name="ascending", value="{{ ascending }}"):
                    SelectOption(value="false", label="Top (highest first)")
                    SelectOption(value="true", label="Bottom (lowest first)")
            Button("Rank", on_click=_search_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Ranking…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-1"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Badge("{{ result.period }}")
                    Small(
                        "{{ result.total_ref_areas }} ref areas with data",
                        css_class="text-muted-foreground",
                    )

                BarChart(
                    data="{{ result.rows }}",
                    series=[ChartSeries(data_key="value", label="Value")],
                    x_axis="ref_area",
                    horizontal=True,
                    show_legend=False,
                    value_format="compact",
                    height=350,
                )

                DataTable(
                    columns=[
                        DataTableColumn(key="rank", header="#", width="50px", sortable=True),
                        DataTableColumn(key="ref_area", header="Code", sortable=True),
                        DataTableColumn(key="ref_area_label", header="Name", sortable=True),
                        DataTableColumn(key="value", header="Value", sortable=True,
                                        format="number:2", align="right"),
                    ],
                    rows="{{ result.rows }}",
                    paginated=True,
                    page_size=10,
                )

    return ui_result(app, app_name="Rank", result=result,
                     params={"idno": idno, "period": period, "n": n, "ascending": ascending})
