"""OutliersApp — Z-score outlier detection for a timeseries indicator.

Supports two modes:
- Cross-section: outliers across ref areas for a given period.
- Longitudinal: outliers across time for a given ref area.
"""

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
    Select,
    SelectOption,
    Small,
)
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import OutliersResponse
from nada_ai.mcp_server import analytics
from nada_ai.mcp_server.apps._ui_result import ui_result

outliers_app = FastMCPApp("Outliers")


@outliers_app.tool()
async def do_outliers(
    idno: str,
    period: str | None = None,
    ref_area: str | None = None,
    threshold: float = 2.0,
    dimensions: dict[str, str] | None = None,
) -> OutliersResponse:
    """Fetch schema + data and detect Z-score outliers (cross-section or longitudinal)."""
    if (period is None) == (ref_area is None):
        return OutliersResponse(
            idno=idno, threshold=threshold,
            error="Provide exactly one of 'period' or 'ref_area'.",
        )
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return OutliersResponse(idno=idno, period=period, ref_area=ref_area, threshold=threshold,
                                error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_
    data = await nada_api.get_all_timeseries_data(idno, dimensions=dimensions)
    if data.error:
        return OutliersResponse(idno=idno, period=period, ref_area=ref_area, threshold=threshold,
                                geo_column=schema.geo_column, obs_column=schema.obs_column,
                                error=data.error)
    try:
        return analytics.detect_outliers(data.data, schema, period=period, ref_area=ref_area,
                                         threshold=threshold, dimensions=dimensions)
    except ValueError as exc:
        return OutliersResponse(idno=idno, period=period, ref_area=ref_area, threshold=threshold,
                                geo_column=schema.geo_column, obs_column=schema.obs_column,
                                error=str(exc))


@outliers_app.ui(
    description=(
        "Open an interactive outlier detection UI for a timeseries indicator. "
        "Cross-section mode: flags ref areas that deviate from peers in a given period. "
        "Longitudinal mode: flags unusual years in a single ref area's own history."
    ),
    title="Detect Outliers",
)
async def show_outliers(
    idno: str = "",
    period: str = "",
    ref_area: str = "",
    threshold: float = 2.0,
) -> PrefabApp:
    """Open the outlier detection UI, pre-loaded when idno and one of period/ref_area are supplied.

    Args:
        idno: Indicator idno to analyse (e.g. SP.POP.TOTL). Pre-fills the form.
        period: Time period for cross-section mode (e.g. 2022). Pre-fills the form.
        ref_area: Ref area code for longitudinal mode (e.g. KEN). Pre-fills the form.
        threshold: Z-score magnitude above which a point is flagged (default 2.0).
    """
    # Infer mode from which param is provided; period takes precedence if both
    mode = "cross_section" if period else ("longitudinal" if ref_area else "cross_section")

    result = None
    if idno and (period or ref_area):
        p = period if period else None
        r = ref_area if ref_area else None
        if (p is None) == (r is None):
            p = period or None  # at least one must differ; skip pre-fetch
        else:
            result = await do_outliers(idno=idno, period=p, ref_area=r, threshold=threshold)

    _cross_action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_outliers,
            arguments={"idno": "{{ idno }}", "period": "{{ period }}", "threshold": "{{ threshold }}"},
            on_success=[SetState("result", "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", "Outlier detection failed."), SetState("loading", False)],
        ),
    ]

    _long_action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_outliers,
            arguments={"idno": "{{ idno }}", "ref_area": "{{ ref_area }}", "threshold": "{{ threshold }}"},
            on_success=[SetState("result", "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", "Outlier detection failed."), SetState("loading", False)],
        ),
    ]

    with PrefabApp(
        title="Detect Outliers",
        state={
            "idno": idno,
            "period": period,
            "ref_area": ref_area,
            "mode": mode,
            "threshold": threshold,
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Detect Outliers", css_class="mb-4 text-xl font-semibold")

        # Mode selector
        with Row(gap=2, css_class="mb-3 items-center"):
            Small("Mode:", css_class="text-muted-foreground shrink-0")
            with Select(name="mode", value="{{ mode }}", css_class="w-48"):
                SelectOption(value="cross_section", label="Cross-section (by period)")
                SelectOption(value="longitudinal", label="Longitudinal (by ref area)")

        # Shared: idno + threshold
        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")

            # Cross-section param
            with If(STATE.mode == "cross_section"):
                with Column(css_class="min-w-28", gap=1):
                    Small("Period")
                    Input(placeholder="e.g. 2022", name="period", value="{{ period }}")

            # Longitudinal param
            with If(STATE.mode == "longitudinal"):
                with Column(css_class="min-w-28", gap=1):
                    Small("Ref area")
                    Input(placeholder="e.g. KEN", name="ref_area", value="{{ ref_area }}")

            with Column(css_class="min-w-32", gap=1):
                Small("Z-score threshold")
                with Select(name="threshold", value="{{ threshold }}"):
                    SelectOption(value="1.5", label="1.5")
                    SelectOption(value="2.0", label="2.0 (default)")
                    SelectOption(value="2.5", label="2.5")
                    SelectOption(value="3.0", label="3.0")

            with If(STATE.mode == "cross_section"):
                Button("Detect", on_click=_cross_action, css_class="shrink-0")
            with If(STATE.mode == "longitudinal"):
                Button("Detect", on_click=_long_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Detecting outliers…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2 flex-wrap"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    with If(STATE.result.period):
                        Badge("{{ result.period }}")
                    with If(STATE.result.ref_area):
                        Badge("{{ result.ref_area }}")
                    Badge("{{ result.n_outliers }} outliers")
                with Row(css_class="gap-4 mb-2"):
                    Small("Mean: {{ result.peer_mean }}", css_class="text-muted-foreground")
                    Small("Std: {{ result.peer_std }}", css_class="text-muted-foreground")

                # Cross-section table: ref areas as rows
                with If(STATE.result.mode == "cross_section"):
                    DataTable(
                        columns=[
                            DataTableColumn(key="ref_area", header="Code", sortable=True),
                            DataTableColumn(key="ref_area_label", header="Name", sortable=True),
                            DataTableColumn(key="value", header="Value",
                                            sortable=True, format="number:2", align="right"),
                            DataTableColumn(key="z_score", header="Z-score",
                                            sortable=True, format="number:3", align="right"),
                            DataTableColumn(key="is_outlier", header="Outlier", sortable=True),
                        ],
                        rows="{{ result.rows }}",
                        paginated=True, page_size=15, search=True,
                    )

                # Longitudinal table: periods as rows
                with If(STATE.result.mode == "longitudinal"):
                    DataTable(
                        columns=[
                            DataTableColumn(key="period", header="Period", sortable=True),
                            DataTableColumn(key="value", header="Value",
                                            sortable=True, format="number:2", align="right"),
                            DataTableColumn(key="z_score", header="Z-score",
                                            sortable=True, format="number:3", align="right"),
                            DataTableColumn(key="is_outlier", header="Outlier", sortable=True),
                        ],
                        rows="{{ result.rows }}",
                        paginated=True, page_size=15, search=True,
                    )

    return ui_result(app, app_name="Outliers", result=result,
                     params={"idno": idno, "period": period or None,
                             "ref_area": ref_area or None, "threshold": threshold})
