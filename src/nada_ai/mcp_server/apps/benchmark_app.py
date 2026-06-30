"""BenchmarkApp — ref area(s) vs peer group for a given period."""

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
from nada_ai.nada.models import BenchmarkResponse
from nada_ai.mcp_server import analytics

benchmark_app = FastMCPApp("Benchmark")


@benchmark_app.tool()
async def do_benchmark(
    idno: str,
    period: str,
    ref_areas: list[str],
    dimensions: dict[str, str] | None = None,
) -> BenchmarkResponse:
    """Fetch schema + data and benchmark ref areas against all peers."""
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return BenchmarkResponse(idno=idno, period=period,
                                 error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_
    data = await nada_api.get_all_timeseries_data(idno, dimensions=dimensions)
    if data.error:
        return BenchmarkResponse(idno=idno, period=period,
                                 geo_column=schema.geo_column, obs_column=schema.obs_column,
                                 error=data.error)
    try:
        return analytics.benchmark(data.data, schema, ref_areas=ref_areas,
                                   period=period, dimensions=dimensions)
    except ValueError as exc:
        return BenchmarkResponse(idno=idno, period=period,
                                 geo_column=schema.geo_column, obs_column=schema.obs_column,
                                 error=str(exc))


@benchmark_app.ui(
    description=(
        "Open an interactive benchmarking UI for a timeseries indicator. "
        "Shows how selected ref areas rank within their peers: percentile rank, Z-score, "
        "and deviation from peer mean and median."
    ),
    title="Benchmark",
)
def show_benchmark(
    idno: str = "",
    period: str = "",
    ref_areas: str = "",
) -> PrefabApp:
    """Render the benchmark comparison table."""

    _action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_benchmark,
            arguments={
                "idno": "{{ idno }}",
                "period": "{{ period }}",
                "ref_areas": "{{ ref_areas_list }}",
            },
            on_success=[SetState("result", "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", "Benchmark failed."), SetState("loading", False)],
        ),
    ]

    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []

    with PrefabApp(
        title="Benchmark",
        state={
            "idno": idno, "period": str(period),
            "ref_areas": ref_areas, "ref_areas_list": parsed_refs,
            "loading": False, "result": None, "error": None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Benchmark", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="min-w-28", gap=1):
                Small("Period")
                Input(placeholder="e.g. 2022", name="period", value="{{ period }}")
            with Column(css_class="flex-1 min-w-40", gap=1):
                Small("Ref areas to benchmark")
                Input(placeholder="e.g. KEN,UGA,TZA", name="ref_areas", value="{{ ref_areas }}")
            Button("Benchmark", on_click=_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Benchmarking…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2 flex-wrap"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Badge("{{ result.period }}")
                    Small("{{ result.peer_count }} peers", css_class="text-muted-foreground")
                with Row(css_class="gap-4 mb-2"):
                    Small("Peer mean: {{ result.peer_mean }}", css_class="text-muted-foreground")
                    Small("Peer median: {{ result.peer_median }}", css_class="text-muted-foreground")
                    Small("Peer std: {{ result.peer_std }}", css_class="text-muted-foreground")

                DataTable(
                    columns=[
                        DataTableColumn(key="ref_area", header="Code", sortable=True),
                        DataTableColumn(key="ref_area_label", header="Name", sortable=True),
                        DataTableColumn(key="value", header="Value",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="percentile_rank", header="Percentile",
                                        sortable=True, format="number:1", align="right"),
                        DataTableColumn(key="z_score", header="Z-score",
                                        sortable=True, format="number:3", align="right"),
                        DataTableColumn(key="vs_mean", header="vs Mean",
                                        sortable=True, format="number:2", align="right"),
                        DataTableColumn(key="vs_median", header="vs Median",
                                        sortable=True, format="number:2", align="right"),
                    ],
                    rows="{{ result.rows }}",
                    paginated=False,
                )

    return app
