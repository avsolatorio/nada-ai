"""SummarizeApp — descriptive statistics across ref areas for a period."""

from __future__ import annotations

from fastmcp import FastMCPApp
from prefab_ui.actions import SetState
from prefab_ui.actions.mcp import CallTool
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Badge,
    Button,
    Card,
    CardContent,
    Column,
    Grid,
    GridItem,
    H3,
    If,
    Input,
    Loader,
    Metric,
    Muted,
    Row,
    Small,
)
from prefab_ui.rx import STATE

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import SummarizeResponse
from nada_ai.mcp_server import analytics

summarize_app = FastMCPApp("Summarize")


@summarize_app.tool()
async def do_summarize(
    idno: str,
    period: str,
    dimensions: dict[str, str] | None = None,
) -> SummarizeResponse:
    """Fetch schema + data and compute descriptive stats for a period."""
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return SummarizeResponse(idno=idno, period=period, geo_column="", obs_column="",
                                 error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_
    data = await nada_api.get_all_timeseries_data(idno, dimensions=dimensions)
    if data.error:
        return SummarizeResponse(idno=idno, period=period, geo_column=schema.geo_column or "",
                                 obs_column=schema.obs_column or "", error=data.error)
    return analytics.summarize(data.data, schema, period=period, dimensions=dimensions)


@summarize_app.ui(
    description=(
        "Open an interactive summary statistics UI for a timeseries indicator. "
        "Shows min, max, mean, median, and std deviation across all ref areas for a given period."
    ),
    title="Summarize Indicator",
)
def summarize_indicator(idno: str = "", period: str = "") -> PrefabApp:
    """Render the summary stats cards."""

    _summarize_action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_summarize,
            arguments={"idno": "{{ idno }}", "period": "{{ period }}"},
            on_success=[
                SetState("result", "{{ $result }}"),
                SetState("loading", False),
            ],
            on_error=[
                SetState("error", "Summary failed. Check the idno and period."),
                SetState("loading", False),
            ],
        ),
    ]

    with PrefabApp(
        title="Summarize Indicator",
        state={
            "idno": idno,
            "period": str(period),
            "loading": False,
            "result": None,
            "error": None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Summarize Indicator", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="min-w-28", gap=1):
                Small("Period")
                Input(placeholder="e.g. 2022", name="period", value="{{ period }}")
            Button("Summarize", on_click=_summarize_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Computing statistics…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Badge("{{ result.period }}")
                    Small(
                        "{{ result.stats.count }} ref areas",
                        css_class="text-muted-foreground",
                    )

                with Grid(css_class="grid-cols-2 md:grid-cols-3 gap-3"):
                    with GridItem():
                        with Card():
                            with CardContent(css_class="pt-4"):
                                Metric(
                                    label="Maximum",
                                    value="{{ result.stats.max_value }}",
                                    description="{{ result.stats.max_ref_area }}",
                                )
                    with GridItem():
                        with Card():
                            with CardContent(css_class="pt-4"):
                                Metric(
                                    label="Minimum",
                                    value="{{ result.stats.min_value }}",
                                    description="{{ result.stats.min_ref_area }}",
                                )
                    with GridItem():
                        with Card():
                            with CardContent(css_class="pt-4"):
                                Metric(label="Mean", value="{{ result.stats.mean }}")
                    with GridItem():
                        with Card():
                            with CardContent(css_class="pt-4"):
                                Metric(label="Median", value="{{ result.stats.median }}")
                    with GridItem():
                        with Card():
                            with CardContent(css_class="pt-4"):
                                Metric(label="Std Dev", value="{{ result.stats.std }}")
                    with GridItem():
                        with Card():
                            with CardContent(css_class="pt-4"):
                                Metric(label="Count", value="{{ result.stats.count }}")

    return app
