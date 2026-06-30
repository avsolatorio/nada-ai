"""OutliersApp — Z-score outlier detection for an indicator in a given period."""

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

outliers_app = FastMCPApp("Outliers")


@outliers_app.tool()
async def do_outliers(
    idno: str,
    period: str,
    threshold: float = 2.0,
    dimensions: dict[str, str] | None = None,
) -> OutliersResponse:
    """Fetch schema + data and detect Z-score outliers."""
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return OutliersResponse(idno=idno, period=period, threshold=threshold,
                                error=schema_resp.error or "Schema unavailable")
    schema = schema_resp.schema_
    data = await nada_api.get_all_timeseries_data(idno, dimensions=dimensions)
    if data.error:
        return OutliersResponse(idno=idno, period=period, threshold=threshold,
                                geo_column=schema.geo_column, obs_column=schema.obs_column,
                                error=data.error)
    try:
        return analytics.detect_outliers(data.data, schema, period=period,
                                         threshold=threshold, dimensions=dimensions)
    except ValueError as exc:
        return OutliersResponse(idno=idno, period=period, threshold=threshold,
                                geo_column=schema.geo_column, obs_column=schema.obs_column,
                                error=str(exc))


@outliers_app.ui(
    description=(
        "Open an interactive outlier detection UI for a timeseries indicator. "
        "Shows Z-scores for all ref areas in a given period and flags statistical outliers."
    ),
    title="Detect Outliers",
)
def show_outliers(
    idno: str = "",
    period: str = "",
    threshold: float = 2.0,
) -> PrefabApp:
    """Render the outlier detection table."""

    _action = [
        SetState("loading", True),
        SetState("result", None),
        SetState("error", None),
        CallTool(
            do_outliers,
            arguments={
                "idno": "{{ idno }}",
                "period": "{{ period }}",
                "threshold": "{{ threshold }}",
            },
            on_success=[SetState("result", "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", "Outlier detection failed."), SetState("loading", False)],
        ),
    ]

    with PrefabApp(
        title="Detect Outliers",
        state={
            "idno": idno, "period": str(period), "threshold": threshold,
            "loading": False, "result": None, "error": None,
        },
        css_class="p-4 max-w-3xl mx-auto",
    ) as app:

        H3("Detect Outliers", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="min-w-28", gap=1):
                Small("Period")
                Input(placeholder="e.g. 2022", name="period", value="{{ period }}")
            with Column(css_class="min-w-32", gap=1):
                Small("Z-score threshold")
                with Select(name="threshold", value="{{ threshold }}"):
                    SelectOption(value="1.5", label="1.5")
                    SelectOption(value="2.0", label="2.0 (default)")
                    SelectOption(value="2.5", label="2.5")
                    SelectOption(value="3.0", label="3.0")
            Button("Detect", on_click=_action, css_class="shrink-0")

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
                    Badge("{{ result.period }}")
                    Badge("{{ result.n_outliers }} outliers")
                with Row(css_class="gap-4 mb-2"):
                    Small("Mean: {{ result.peer_mean }}", css_class="text-muted-foreground")
                    Small("Std: {{ result.peer_std }}", css_class="text-muted-foreground")

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
                    paginated=True,
                    page_size=15,
                    search=True,
                )

    return app
