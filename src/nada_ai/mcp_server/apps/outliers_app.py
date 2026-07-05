"""OutliersApp — robust outlier detection for a timeseries indicator.

Supports two modes (cross-section / longitudinal) and three methods
(modified_zscore, iqr, trend_residual).
"""

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
    Text,
)
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada.models import OutliersResponse
from nada_ai.mcp_server.tools import _nada_outliers
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

outliers_app = FastMCPApp("Outliers")

_DEFAULT_THRESHOLDS = {"modified_zscore": 3.5, "iqr": 0.0, "trend_residual": 3.5}


@outliers_app.tool()
async def do_outliers(
    idno: str,
    period: str | None = None,
    ref_area: str | None = None,
    method: str = "modified_zscore",
    threshold: float | None = None,
    dimensions: dict[str, str] | None = None,
) -> OutliersResponse:
    """Fetch schema + data and detect outliers using the chosen method."""
    return await _nada_outliers(idno=idno, period=period, ref_area=ref_area,
                                method=method, threshold=threshold, dimensions=dimensions)


@outliers_app.ui(
    description=(
        "Open an interactive outlier detection UI for a timeseries indicator. "
        "Cross-section mode: flags ref areas that deviate from peers in a given period. "
        "Longitudinal mode: flags unusual years in a single ref area's own history. "
        "Supports modified Z-score (MAD-based), IQR fences, and LOWESS trend residuals."
    ),
    title="Detect Outliers",
)
async def show_outliers(
    idno: str = "",
    period: str = "",
    ref_area: str = "",
    method: str = "modified_zscore",
    threshold: float | None = None,
) -> PrefabApp:
    """Open the outlier detection UI, pre-loaded when idno and one of period/ref_area are given.

    Args:
        idno: Indicator idno to analyse (e.g. SP.POP.TOTL). Pre-fills the form.
        period: Time period for cross-section mode (e.g. 2022). Pre-fills the form.
        ref_area: Ref area code for longitudinal mode (e.g. KEN). Pre-fills the form.
        method: Detection method — modified_zscore (default), iqr, or trend_residual.
        threshold: Flagging threshold (method-specific default if omitted).
    """
    mode = "cross_section" if period else ("longitudinal" if ref_area else "cross_section")
    eff_threshold = threshold if threshold is not None else _DEFAULT_THRESHOLDS.get(method, 3.5)

    result = None
    p = period if period else None
    r = ref_area if ref_area else None
    if idno and (p is None) != (r is None):
        result = await do_outliers(idno=idno, period=p, ref_area=r,
                                   method=method, threshold=threshold)

    _cross_action = make_action(
        do_outliers,
        {"idno": "{{ idno }}", "period": "{{ period }}", "ref_area": None,
         "method": "{{ method }}", "threshold": "{{ threshold }}"},
        error_msg="Outlier detection failed.",
    )
    _long_action = make_action(
        do_outliers,
        {"idno": "{{ idno }}", "period": None, "ref_area": "{{ ref_area }}",
         "method": "{{ method }}", "threshold": "{{ threshold }}"},
        error_msg="Outlier detection failed.",
    )

    with PrefabApp(
        title="Detect Outliers",
        state={
            "idno": idno,
            "period": period,
            "ref_area": ref_area,
            "mode": mode,
            "method": method,
            "threshold": eff_threshold,
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Detect Outliers", css_class="mb-4 text-xl font-semibold")

        # Mode + method selectors
        with Row(gap=3, css_class="mb-3 flex-wrap items-center"):
            with Column(gap=1):
                Small("Mode")
                with Select(name="mode", value="{{ mode }}", css_class="w-52"):
                    SelectOption(value="cross_section", label="Cross-section (by period)")
                    SelectOption(value="longitudinal", label="Longitudinal (by ref area)")
            with Column(gap=1):
                Small("Method")
                with Select(name="method", value="{{ method }}", css_class="w-56"):
                    SelectOption(value="modified_zscore", label="Modified Z-score (MAD)")
                    SelectOption(value="iqr", label="IQR fences (Tukey)")
                    SelectOption(value="trend_residual", label="Trend residuals (LOWESS)")

        # Inputs
        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")

            with If(STATE.mode == "cross_section"):
                with Column(css_class="min-w-28", gap=1):
                    Small("Period")
                    Input(placeholder="e.g. 2022", name="period", value="{{ period }}")

            with If(STATE.mode == "longitudinal"):
                with Column(css_class="min-w-28", gap=1):
                    Small("Ref area")
                    Input(placeholder="e.g. KEN", name="ref_area", value="{{ ref_area }}")

            with Column(css_class="min-w-28", gap=1):
                Small("Threshold")
                Input(placeholder="e.g. 3.5", name="threshold", value="{{ threshold }}")

            with If(STATE.mode == "cross_section"):
                Button("Detect", on_click=_cross_action, css_class="shrink-0")
            with If(STATE.mode == "longitudinal"):
                Button("Detect", on_click=_long_action, css_class="shrink-0")

        # Method hint
        with If(STATE.method == "modified_zscore"):
            Muted("Modified Z-score: score = 0.6745·(x − median)/MAD. Threshold flags |score| ≥ value.",
                  css_class="text-xs mb-3")
        with If(STATE.method == "iqr"):
            Muted("IQR fences: flags values outside Q1 − 1.5·IQR or Q3 + 1.5·IQR. Score = IQR-widths outside fence.",
                  css_class="text-xs mb-3")
        with If(STATE.method == "trend_residual"):
            Muted("LOWESS trend residuals (longitudinal only): fits a smooth trend, then scores residuals by modified Z-score.",
                  css_class="text-xs mb-3")

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
                    Badge("{{ result.method }}", css_class="font-mono text-xs")
                    Badge("{{ result.n_outliers }} outliers")
                with Row(css_class="gap-4 mb-2"):
                    Small("Center: {{ result.peer_mean }}", css_class="text-muted-foreground")
                    Small("Spread: {{ result.peer_std }}", css_class="text-muted-foreground")

                # Cross-section table
                with If(STATE.result.mode == "cross_section"):
                    DataTable(
                        columns=[
                            DataTableColumn(key="ref_area", header="Code", sortable=True),
                            DataTableColumn(key="ref_area_label", header="Name", sortable=True),
                            DataTableColumn(key="value", header="Value",
                                            sortable=True, format="number:2", align="right"),
                            DataTableColumn(key="z_score", header="Score",
                                            sortable=True, format="number:3", align="right"),
                            DataTableColumn(key="is_outlier", header="Outlier", sortable=True),
                        ],
                        rows="{{ result.rows }}",
                        paginated=True, page_size=15, search=True,
                    )

                # Longitudinal table — modified_zscore / iqr
                with If(STATE.result.mode == "longitudinal"):
                    with If(STATE.result.method != "trend_residual"):
                        DataTable(
                            columns=[
                                DataTableColumn(key="period", header="Period", sortable=True),
                                DataTableColumn(key="value", header="Value",
                                                sortable=True, format="number:2", align="right"),
                                DataTableColumn(key="z_score", header="Score",
                                                sortable=True, format="number:3", align="right"),
                                DataTableColumn(key="is_outlier", header="Outlier", sortable=True),
                            ],
                            rows="{{ result.rows }}",
                            paginated=True, page_size=15, search=True,
                        )

                # Longitudinal trend_residual table — extra trend/residual columns
                with If(STATE.result.mode == "longitudinal"):
                    with If(STATE.result.method == "trend_residual"):
                        DataTable(
                            columns=[
                                DataTableColumn(key="period", header="Period", sortable=True),
                                DataTableColumn(key="value", header="Value",
                                                sortable=True, format="number:2", align="right"),
                                DataTableColumn(key="trend_value", header="Trend",
                                                sortable=True, format="number:2", align="right"),
                                DataTableColumn(key="residual", header="Residual",
                                                sortable=True, format="number:2", align="right"),
                                DataTableColumn(key="z_score", header="Score",
                                                sortable=True, format="number:3", align="right"),
                                DataTableColumn(key="is_outlier", header="Outlier", sortable=True),
                            ],
                            rows="{{ result.rows }}",
                            paginated=True, page_size=15, search=True,
                        )

    return ui_result(app, app_name="Outliers", result=result,
                     params={"idno": idno, "period": period or None,
                             "ref_area": ref_area or None,
                             "method": method, "threshold": threshold})
