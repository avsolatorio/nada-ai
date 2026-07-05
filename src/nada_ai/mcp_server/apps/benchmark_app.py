"""BenchmarkApp — ref area(s) vs peer group for a given period."""

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

from nada_ai.nada.models import BenchmarkResponse
from nada_ai.mcp_server.tools import _nada_benchmark
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

benchmark_app = FastMCPApp("Benchmark")


@benchmark_app.tool()
async def do_benchmark(
    idno: str,
    period: str,
    ref_areas: list[str],
    dimensions: dict[str, str] | None = None,
) -> BenchmarkResponse:
    """Fetch schema + data and benchmark ref areas against all peers."""
    return await _nada_benchmark(idno=idno, period=period, ref_areas=ref_areas,
                                  dimensions=dimensions)


@benchmark_app.ui(
    description=(
        "Open an interactive benchmarking UI for a timeseries indicator. "
        "Shows how selected ref areas rank within their peers: percentile rank, Z-score, "
        "and deviation from peer mean and median."
    ),
    title="Benchmark",
)
async def show_benchmark(
    idno: str = "",
    period: str = "",
    ref_areas: str = "",
) -> PrefabApp:
    """Open the benchmark UI, pre-loaded with data when idno, period and ref areas are supplied.

    Args:
        idno: Indicator idno to benchmark (e.g. SP.POP.TOTL). Pre-fills the form.
        period: Time period to benchmark in (e.g. 2022). Pre-fills the form.
        ref_areas: Comma-separated ref area codes to benchmark (e.g. KEN,UGA). Pre-fills the form.
    """
    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else []
    result = None
    if idno and period and parsed_refs:
        result = await do_benchmark(idno=idno, period=period, ref_areas=parsed_refs)

    _action = make_action(
        do_benchmark,
        {"idno": "{{ idno }}", "period": "{{ period }}", "ref_areas": "{{ ref_areas_list }}"},
        error_msg="Benchmark failed.",
    )

    with PrefabApp(
        title="Benchmark",
        state={
            "idno": idno, "period": str(period),
            "ref_areas": ref_areas, "ref_areas_list": parsed_refs,
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
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

    return ui_result(app, app_name="Benchmark", result=result,
                     params={"idno": idno, "period": period, "ref_areas": ref_areas or None})
