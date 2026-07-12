"""VisualizeApp — chart one to three indicators over time, aggregated across ref areas.

Unlike compare/join/benchmark/growth/trend (which compare an arbitrary,
user-typed number of ref areas and therefore can't safely chart per-ref-area
series — see the other apps' do_X docstrings), this app charts a small,
FIXED number of indicator slots (idno, idno2, idno3). The chart's series
list is always declared with exactly 3 entries regardless of which slots
are filled in, so it stays correct and reactive across button clicks.
"""

from __future__ import annotations

import asyncio
from typing import Any

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
)
from prefab_ui.components.charts import AreaChart, BarChart, ChartSeries, LineChart
from prefab_ui.rx import STATE

from nada_ai.mcp_server.tools import _nada_aggregate
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

visualize_app = FastMCPApp("Visualize")

_MAX_INDICATORS = 3

# Series slots are always declared, one per possible indicator. Labels are
# live-bound to the corresponding idno state field, so they track whatever
# the user last typed without needing the series list itself to change.
_SERIES = [
    ChartSeries(data_key="value_1", label="{{ idno }}"),
    ChartSeries(data_key="value_2", label="{{ idno2 }}"),
    ChartSeries(data_key="value_3", label="{{ idno3 }}"),
]


def _merge_by_period(responses: list) -> list[dict[str, Any]]:
    by_period: dict[str, dict[str, Any]] = {}
    for i, resp in enumerate(responses, start=1):
        if resp is None or resp.error:
            continue
        for row in resp.rows:
            entry = by_period.setdefault(row.period, {"period": row.period})
            entry[f"value_{i}"] = row.mean
    return sorted(by_period.values(), key=lambda r: r["period"])


@visualize_app.tool()
async def do_visualize(
    idno: str,
    idno2: str = "",
    idno3: str = "",
    ref_areas: str | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
) -> dict[str, Any]:
    """Fetch mean-per-period aggregate series for 1-3 indicators and merge by period.

    ref_areas is a comma-separated string, parsed here — see do_compare's
    docstring in compare_app.py for why (avoids a stale pre-parsed list).
    """
    parsed_refs = [r.strip() for r in ref_areas.split(",") if r.strip()] if ref_areas else None
    idnos = [idno, idno2 or None, idno3 or None]
    responses = await asyncio.gather(*[
        _nada_aggregate(idno=i, ref_areas=parsed_refs, from_year=from_year, to_year=to_year)
        if i else asyncio.sleep(0, result=None)
        for i in idnos
    ])
    errors = [r.error for r in responses if r is not None and r.error]
    rows = _merge_by_period(responses)
    return {"rows": rows, "errors": errors}


@visualize_app.ui(
    description=(
        "Open an interactive chart of one to three timeseries indicators over time, "
        "aggregated (mean) across a group of ref areas. Switch between line, area, "
        "and bar chart types. Use this for plotting/visualizing indicator trends — "
        "different from aggregate/compare, which focus on tabular group statistics."
    ),
    title="Visualize Indicators",
)
async def visualize_indicators(
    idno: str = "",
    idno2: str = "",
    idno3: str = "",
    ref_areas: str = "",
    from_year: str = "",
    to_year: str = "",
    chart_type: str = "line",
) -> PrefabApp:
    """Open the visualization UI, pre-loaded when at least idno is supplied.

    Args:
        idno: Primary indicator idno to chart (e.g. SP.POP.TOTL). Pre-fills the form.
        idno2: Optional second indicator to overlay for comparison.
        idno3: Optional third indicator to overlay for comparison.
        ref_areas: Comma-separated ref area codes to aggregate over (e.g. KEN,UGA). Optional; default is all.
        from_year: Start year filter (e.g. 2000). Optional.
        to_year: End year filter (e.g. 2022). Optional.
        chart_type: Initial chart type — "line", "area", or "bar". Default "line".
    """
    fy = int(from_year) if from_year and from_year.isdigit() else None
    ty = int(to_year) if to_year and to_year.isdigit() else None
    result = None
    if idno:
        result = await do_visualize(idno=idno, idno2=idno2, idno3=idno3,
                                    ref_areas=ref_areas or None, from_year=fy, to_year=ty)

    _action = make_action(
        do_visualize,
        {
            "idno": "{{ idno }}", "idno2": "{{ idno2 }}", "idno3": "{{ idno3 }}",
            "ref_areas": "{{ ref_areas }}", "from_year": "{{ from_year }}", "to_year": "{{ to_year }}",
        },
        error_msg="Visualization failed. Check the idno(s).",
    )

    has_data = bool(result and result.get("rows"))

    with PrefabApp(
        title="Visualize Indicators",
        state={
            "idno": idno, "idno2": idno2, "idno3": idno3,
            "ref_areas": ref_areas, "from_year": from_year, "to_year": to_year,
            "chart_type": chart_type,
            "loading": False,
            "result": result if has_data else None,
            "error": None if has_data else ("; ".join(result["errors"]) if result and result.get("errors") else None),
        },
        css_class="p-4 max-w-4xl mx-auto",
    ) as app:

        H3("Visualize Indicators", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator 2 (optional)")
                Input(placeholder="e.g. NY.GDP.PCAP.CD", name="idno2", value="{{ idno2 }}")
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator 3 (optional)")
                Input(placeholder="e.g. SP.DYN.LE00.IN", name="idno3", value="{{ idno3 }}")
            with Column(css_class="min-w-36", gap=1):
                Small("Ref areas (optional)")
                Input(placeholder="e.g. KEN,UGA,TZA", name="ref_areas", value="{{ ref_areas }}")
            with Column(css_class="min-w-24", gap=1):
                Small("From year")
                Input(placeholder="e.g. 2000", name="from_year", value="{{ from_year }}")
            with Column(css_class="min-w-24", gap=1):
                Small("To year")
                Input(placeholder="e.g. 2022", name="to_year", value="{{ to_year }}")
            Button("Plot", on_click=_action, css_class="shrink-0")

        with Row(gap=2, css_class="mb-4 items-center"):
            Small("Chart type", css_class="text-muted-foreground")
            with Select(name="chart_type", value="{{ chart_type }}", css_class="w-32"):
                SelectOption(value="line", label="Line")
                SelectOption(value="area", label="Area")
                SelectOption(value="bar", label="Bar")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Fetching series…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2 flex-wrap"):
                    Badge("{{ idno }}")
                    with If(STATE.idno2):
                        Badge("{{ idno2 }}")
                    with If(STATE.idno3):
                        Badge("{{ idno3 }}")

                with If(STATE.chart_type == "line"):
                    LineChart(
                        data="{{ result.rows }}",
                        series=_SERIES,
                        x_axis="period",
                        curve="smooth",
                        show_dots=False,
                        value_format="compact",
                        height=350,
                    )
                with If(STATE.chart_type == "area"):
                    AreaChart(
                        data="{{ result.rows }}",
                        series=_SERIES,
                        x_axis="period",
                        curve="smooth",
                        value_format="compact",
                        height=350,
                    )
                with If(STATE.chart_type == "bar"):
                    BarChart(
                        data="{{ result.rows }}",
                        series=_SERIES,
                        x_axis="period",
                        value_format="compact",
                        height=350,
                    )

    return ui_result(app, app_name="Visualize", result=result, params={
        "idno": idno, "idno2": idno2 or None, "idno3": idno3 or None,
        "ref_areas": ref_areas or None,
        "from_year": from_year or None, "to_year": to_year or None,
    })
