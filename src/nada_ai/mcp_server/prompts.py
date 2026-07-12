"""Prompts for the NADA MCP Server.

MCP prompts are user-controlled (surfaced as slash commands / explicit
selections in clients, not auto-invoked by the model). Each one below seeds a
well-formed starting instruction for a common analytical workflow so the user
doesn't have to spell out the tool-chaining steps by hand.
"""

from .tool_config import get_mcp_tool_texts
from ._server_definition import mcp

_TOOL_TEXTS = get_mcp_tool_texts()
_PREFIX = _TOOL_TEXTS.prefix


@mcp.prompt(name=f"{_PREFIX}_explore_indicator")
def explore_indicator(idno: str) -> str:
    """Get a first-look overview of a timeseries indicator: schema, coverage, and trend.

    Args:
        idno: Indicator idno to explore (e.g. ``SP.POP.TOTL``).
    """
    return (
        f"Explore the indicator `{idno}` in the {_TOOL_TEXTS.catalog_name}. "
        f"Call {_PREFIX}_get_schema first to see the column names and any disaggregation "
        f"dimensions. Then call {_PREFIX}_coverage to see which ref areas and periods have "
        f"data, and {_PREFIX}_trend to see which ref areas are improving, declining, or "
        "stable over the available time range. Summarize the indicator's definition "
        "(from get_metadata if not already known), its data coverage, and the overall trend "
        "in a few sentences."
    )


@mcp.prompt(name=f"{_PREFIX}_compare_countries")
def compare_countries(idno: str, ref_areas: str, period: str) -> str:
    """Compare a set of ref areas on one indicator for a given period: values, rank, and benchmark.

    Args:
        idno: Indicator idno to compare (e.g. ``SP.POP.TOTL``).
        ref_areas: Comma-separated ref area codes to compare (e.g. ``KEN,UGA,TZA``).
        period: The time period to compare within (e.g. ``2022``).
    """
    areas = [a.strip() for a in ref_areas.split(",") if a.strip()]
    areas_repr = str(areas)
    return (
        f"Compare {areas_repr} on indicator `{idno}` for period `{period}` in the "
        f"{_TOOL_TEXTS.catalog_name}. Call {_PREFIX}_get_schema first to confirm column names "
        f"and dimensions, then call {_PREFIX}_benchmark with ref_areas={areas_repr} and "
        f"period='{period}' to get each area's percentile rank and deviation from the peer "
        f"group. Also call {_PREFIX}_compare with the same ref_areas to show the trend leading "
        "up to that period. Present the results as a short comparison with the benchmark "
        "figures and a one-line takeaway on how the group compares."
    )


@mcp.prompt(name=f"{_PREFIX}_find_anomalies")
def find_anomalies(idno: str, period: str | None = None, ref_area: str | None = None) -> str:
    """Detect and explain statistical anomalies for an indicator — cross-section or longitudinal.

    Args:
        idno: Indicator idno to check for anomalies (e.g. ``SP.POP.TOTL``).
        period: Time period for cross-section mode (ranks all ref areas by deviation from peers
            in that period). Provide exactly one of ``period`` or ``ref_area``.
        ref_area: Ref area code for longitudinal mode (ranks all periods for that ref area to
            find unusual years). Provide exactly one of ``period`` or ``ref_area``.
    """
    mode_hint = ""
    if period and not ref_area:
        mode_hint = f"period='{period}' (cross-section mode — ranks ref areas by deviation from peers in that period)"
    elif ref_area and not period:
        mode_hint = f"ref_area='{ref_area}' (longitudinal mode — ranks periods for that ref area)"
    else:
        mode_hint = (
            "either period='<year>' (cross-section) or ref_area='<code>' (longitudinal) — "
            "ask the user which one they want if unclear"
        )
    return (
        f"Find statistical anomalies in indicator `{idno}` in the {_TOOL_TEXTS.catalog_name}. "
        f"Call {_PREFIX}_get_schema first, then call {_PREFIX}_outliers with {mode_hint}. "
        "Use the default modified_zscore method unless the data looks skewed, in which case "
        "try method='iqr' instead. For each flagged anomaly, note the ref area/period, the "
        "observed value, and the Z-score, then briefly suggest a plausible explanation or "
        "flag it as worth investigating further."
    )
