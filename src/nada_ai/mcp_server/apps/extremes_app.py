"""ExtremesApp — global maximum and minimum observation for a timeseries indicator."""

from __future__ import annotations

from fastmcp import FastMCPApp
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

from nada_ai.nada.models import ExtremesResponse
from nada_ai.mcp_server.tools import _nada_extremes
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

extremes_app = FastMCPApp("Extremes")


@extremes_app.tool()
async def do_extremes(
    idno: str,
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> ExtremesResponse:
    """Fetch schema + data and find the global max/min observation."""
    return await _nada_extremes(idno=idno, from_year=from_year, to_year=to_year, dimensions=dimensions)


@extremes_app.ui(
    description=(
        "Open an interactive extremes UI for a timeseries indicator. "
        "Shows the global maximum and minimum observation across all periods and ref areas — "
        "answers 'which country had the highest X ever' or 'what was the worst year for Y'."
    ),
    title="Extremes",
)
async def show_extremes(idno: str = "", from_year: str = "", to_year: str = "") -> PrefabApp:
    """Open the extremes UI, pre-loaded with data when idno is supplied.

    Args:
        idno: Indicator idno to analyse (e.g. SP.POP.TOTL). Pre-fills the form.
        from_year: Narrow to observations from this year (inclusive). Optional.
        to_year: Narrow to observations up to this year (inclusive). Optional.
    """
    fy = int(from_year) if from_year and from_year.isdigit() else None
    ty = int(to_year) if to_year and to_year.isdigit() else None
    result = None
    if idno:
        result = await do_extremes(idno=idno, from_year=fy, to_year=ty)

    _action = make_action(
        do_extremes,
        {"idno": "{{ idno }}", "from_year": "{{ from_year }}", "to_year": "{{ to_year }}"},
        error_msg="Extremes lookup failed. Check the idno.",
    )

    with PrefabApp(
        title="Extremes",
        state={
            "idno": idno,
            "from_year": from_year,
            "to_year": to_year,
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
        },
        css_class="p-4 max-w-2xl mx-auto",
    ) as app:

        H3("Extremes", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="min-w-24", gap=1):
                Small("From year")
                Input(placeholder="e.g. 2000", name="from_year", value="{{ from_year }}")
            with Column(css_class="min-w-24", gap=1):
                Small("To year")
                Input(placeholder="e.g. 2022", name="to_year", value="{{ to_year }}")
            Button("Find Extremes", on_click=_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Scanning observations…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-2"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Badge("{{ result.total_observations }} observations scanned")

                with Grid(css_class="grid-cols-1 md:grid-cols-2 gap-3"):
                    with GridItem():
                        with Card():
                            with CardContent(css_class="pt-4"):
                                Metric(
                                    label="Maximum",
                                    value="{{ result.maximum.value }}",
                                    description="{{ result.maximum.ref_area_label }} ({{ result.maximum.ref_area }}) — {{ result.maximum.period }}",
                                )
                    with GridItem():
                        with Card():
                            with CardContent(css_class="pt-4"):
                                Metric(
                                    label="Minimum",
                                    value="{{ result.minimum.value }}",
                                    description="{{ result.minimum.ref_area_label }} ({{ result.minimum.ref_area }}) — {{ result.minimum.period }}",
                                )

    return ui_result(app, app_name="Extremes", result=result,
                     params={"idno": idno, "from_year": from_year or None, "to_year": to_year or None})
