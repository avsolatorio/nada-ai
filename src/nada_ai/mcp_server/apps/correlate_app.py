"""CorrelateApp — Pearson correlation between two indicators for a period."""

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
    Text,
)
from prefab_ui.components.charts import ChartSeries, ScatterChart
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada.models import CorrelateResponse
from nada_ai.mcp_server.tools import _nada_correlate
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

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
    return await _nada_correlate(idno1=idno1, idno2=idno2, period=period,
                                  from_year=from_year, to_year=to_year,
                                  dimensions1=dimensions1, dimensions2=dimensions2)


@correlate_app.ui(
    description=(
        "Open an interactive correlation UI between two timeseries indicators. "
        "Shows Pearson r, a scatter plot of value1 vs value2, and the underlying "
        "per-ref-area values table for a given period."
    ),
    title="Correlate Indicators",
)
async def show_correlate(
    idno1: str = "",
    idno2: str = "",
    period: str = "",
) -> PrefabApp:
    """Open the correlation UI, pre-loaded with data when both idnos and period are supplied.

    Args:
        idno1: First indicator idno (e.g. SP.POP.TOTL). Pre-fills the form.
        idno2: Second indicator idno (e.g. NY.GDP.PCAP.CD). Pre-fills the form.
        period: Time period to correlate within (e.g. 2022). Pre-fills the form.
    """
    result = None
    if idno1 and idno2 and period:
        result = await do_correlate(idno1=idno1, idno2=idno2, period=period,
                                    from_year=int(period) if period.isdigit() else None,
                                    to_year=int(period) if period.isdigit() else None)

    _action = make_action(
        do_correlate,
        {"idno1": "{{ idno1 }}", "idno2": "{{ idno2 }}", "period": "{{ period }}"},
        error_msg="Correlation failed.",
    )

    with PrefabApp(
        title="Correlate Indicators",
        state={
            "idno1": idno1, "idno2": idno2, "period": str(period),
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
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

                # value1/value2 are always the fixed column names for this response
                # shape, so a single scatter series is safe regardless of n ref areas.
                ScatterChart(
                    data="{{ result.rows }}",
                    series=[ChartSeries(data_key="value2", label="Ref areas")],
                    x_axis="value1",
                    y_axis="value2",
                    height=350,
                )

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

    return ui_result(app, app_name="Correlate", result=result,
                     params={"idno1": idno1, "idno2": idno2, "period": period})
