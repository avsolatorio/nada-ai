"""CompareApp — side-by-side time series across multiple ref areas."""

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
from prefab_ui.components.data_table import DataTable, DataTableColumn
from prefab_ui.rx import STATE

from nada_ai.nada.models import CompareResponse
from nada_ai.mcp_server.tools import _nada_compare
from nada_ai.mcp_server.apps._actions import make_action
from nada_ai.mcp_server.apps._ui_result import ui_result

compare_app = FastMCPApp("Compare")


@compare_app.tool()
async def do_compare(
    idno: str,
    ref_areas: str,
    from_year: int | None = None,
    to_year: int | None = None,
    dimensions: dict[str, str] | None = None,
) -> CompareResponse:
    """Fetch schema + data and build a pivoted comparison.

    ``ref_areas`` is a comma-separated string, parsed here — the click action
    binds it directly to the same live state key as the text input, so edits
    to the field are always reflected (a separately-tracked pre-parsed list
    would go stale the moment the user edits the input after initial load).
    """
    parsed = [r.strip() for r in ref_areas.split(",") if r.strip()]
    return await _nada_compare(idno=idno, ref_areas=parsed,
                               from_year=from_year, to_year=to_year, dimensions=dimensions)


@compare_app.ui(
    description=(
        "Open an interactive time-series comparison UI for a timeseries indicator. "
        "Shows a searchable, sortable table of values for multiple ref areas "
        "(countries, provinces, etc.) across all available periods."
    ),
    title="Compare Ref Areas",
)
async def compare_ref_areas(
    idno: str = "",
    ref_areas: str = "",
    from_year: str = "",
    to_year: str = "",
) -> PrefabApp:
    """Open the comparison UI, pre-loaded with data when idno and ref areas are supplied.

    Args:
        idno: Indicator idno to compare (e.g. SP.POP.TOTL). Pre-fills the form.
        ref_areas: Comma-separated ref area codes to compare (e.g. KEN,UGA,TZA). Pre-fills the form.
        from_year: Start year filter (e.g. 2000). Optional.
        to_year: End year filter (e.g. 2022). Optional.
    """
    fy = int(from_year) if from_year and from_year.isdigit() else None
    ty = int(to_year) if to_year and to_year.isdigit() else None
    result = None
    if idno and ref_areas.strip():
        result = await do_compare(idno=idno, ref_areas=ref_areas, from_year=fy, to_year=ty)

    _compare_action = make_action(
        do_compare,
        {"idno": "{{ idno }}", "ref_areas": "{{ ref_areas }}",
         "from_year": "{{ from_year }}", "to_year": "{{ to_year }}"},
        error_msg="Comparison failed. Check the idno and ref area codes.",
    )

    with PrefabApp(
        title="Compare Ref Areas",
        state={
            "idno": idno,
            "ref_areas": ref_areas,
            "from_year": from_year,
            "to_year": to_year,
            "loading": False,
            "result": result.model_dump() if result and not result.error else None,
            "error": result.error if result else None,
        },
        css_class="p-4 max-w-4xl mx-auto",
    ) as app:

        H3("Compare Ref Areas", css_class="mb-4 text-xl font-semibold")

        with Row(gap=2, css_class="mb-4 flex-wrap items-end"):
            with Column(css_class="min-w-40", gap=1):
                Small("Indicator idno")
                Input(placeholder="e.g. SP.POP.TOTL", name="idno", value="{{ idno }}")
            with Column(css_class="flex-1 min-w-48", gap=1):
                Small("Ref areas (comma-separated codes)")
                Input(placeholder="e.g. KEN,UGA,TZA", name="ref_areas", value="{{ ref_areas }}")
            with Column(css_class="min-w-24", gap=1):
                Small("From year")
                Input(placeholder="e.g. 2010", name="from_year", value="{{ from_year }}")
            with Column(css_class="min-w-24", gap=1):
                Small("To year")
                Input(placeholder="e.g. 2022", name="to_year", value="{{ to_year }}")
            Button("Compare", on_click=_compare_action, css_class="shrink-0")

        with If(STATE.loading):
            with Row(css_class="justify-center py-6"):
                Loader()
                Muted("Loading comparison…", css_class="ml-2")

        with If(STATE.error):
            Muted(STATE.error, css_class="text-destructive py-2")

        with If(STATE.result):
            with Column(gap=3):
                with Row(css_class="items-center gap-2 mb-1 flex-wrap"):
                    Small("{{ result.indicator_name }}", css_class="text-muted-foreground")
                    Text(
                        "{{ result.ref_areas | join(', ') }}",
                        css_class="text-sm text-muted-foreground",
                    )

                # Use unpivoted rows so the table works regardless of which ref_areas were queried
                DataTable(
                    columns=[
                        DataTableColumn(key="period", header="Period", sortable=True),
                        DataTableColumn(key="ref_area", header="Ref Area", sortable=True),
                        DataTableColumn(key="value", header="Value",
                                        sortable=True, format="number:2", align="right"),
                    ],
                    rows="{{ result.rows_unpivoted }}",
                    paginated=True,
                    page_size=20,
                    search=True,
                )

    return ui_result(app, app_name="Compare", result=result,
                     params={"idno": idno, "ref_areas": ref_areas or None,
                             "from_year": from_year or None, "to_year": to_year or None})
