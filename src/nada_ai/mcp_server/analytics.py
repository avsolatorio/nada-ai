"""Pure analytical functions over timeseries observation rows.

All functions are schema-driven: column names for geography, time period, and
observation value are resolved from ``IndicatorSchema`` rather than hardcoded.
This makes them work for any indicator regardless of its DSD structure —
country-level annual data, provincial monthly data, or anything with extra
disaggregation dimensions.

None of these functions perform I/O; they accept pre-fetched data and a schema
so they are straightforward to unit-test.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from nada_ai.nada.models import (
    CompareResponse,
    CompareRow,
    ExtremePoint,
    ExtremesResponse,
    GrowthResponse,
    GrowthRow,
    IndicatorSchema,
    RankResponse,
    RankRow,
    SummaryStats,
    SummarizeResponse,
    TimeseriesDataRow,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_obs(value: str | float | None) -> float | None:
    """Coerce an OBS_VALUE (may arrive as string) to float, or None."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    except (ValueError, TypeError):
        return None


def _row_value(row: TimeseriesDataRow, column: str) -> Any:
    """Read a named column from a row, falling back to model_extra."""
    v = getattr(row, column, _MISSING)
    if v is _MISSING:
        return (row.model_extra or {}).get(column)
    return v


_MISSING = object()


def _apply_dimension_filter(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    dimensions: dict[str, str],
) -> list[TimeseriesDataRow]:
    """Keep only rows where every requested dimension value matches."""
    if not dimensions:
        return rows
    result = []
    for row in rows:
        match = True
        for col, val in dimensions.items():
            row_val = _row_value(row, col)
            if row_val is None or str(row_val) != str(val):
                match = False
                break
        if match:
            result.append(row)
    return result


def _label_map(
    rows: list[TimeseriesDataRow],
    code_col: str,
    label_col: str | None,
) -> dict[str, str]:
    """Build {code: label} from rows when a companion label column exists."""
    if not label_col:
        return {}
    mapping: dict[str, str] = {}
    for row in rows:
        code = _row_value(row, code_col)
        label = _row_value(row, label_col)
        if code is not None and label is not None and str(code) not in mapping:
            mapping[str(code)] = str(label)
    return mapping


def _indicator_name(rows: list[TimeseriesDataRow]) -> str | None:
    """Extract the indicator name from the first row that has it."""
    for row in rows:
        # Standard WDI column; generalise via model_extra fallback
        name = getattr(row, "INDICATOR_NAME", None)
        if name:
            return str(name)
        extra_name = (row.model_extra or {}).get("INDICATOR_NAME")
        if extra_name:
            return str(extra_name)
    return None


def _label_column_for(schema: IndicatorSchema, code_col: str) -> str | None:
    """Find the companion label/name column for a code column (heuristic)."""
    base = code_col.replace("_CODE", "").replace("_ID", "")
    for c in schema.components:
        if c.column_type == "attribute" and c.name != code_col:
            if base in c.name and "NAME" in c.name:
                return c.name
    return None


def _require_columns(schema: IndicatorSchema) -> tuple[str, str, str]:
    """Return (geo_column, time_column, obs_column) or raise ValueError."""
    missing = []
    if not schema.geo_column:
        missing.append("geography")
    if not schema.time_column:
        missing.append("time_period")
    if not schema.obs_column:
        missing.append("observation_value")
    if missing:
        raise ValueError(f"Schema for '{schema.idno}' is missing required columns: {missing}")
    return schema.geo_column, schema.time_column, schema.obs_column  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    period: str,
    n: int = 10,
    ascending: bool = False,
    dimensions: dict[str, str] | None = None,
) -> RankResponse:
    """Return the top/bottom N ref areas by observation value for a given period.

    Args:
        rows: Pre-fetched observation rows (may span multiple periods).
        schema: DSD schema for the indicator.
        period: The time period to rank within (must match TIME_PERIOD values exactly).
        n: Number of ref areas to return.
        ascending: If True return lowest values first (bottom-N); default False (top-N).
        dimensions: Optional dimension filters applied before ranking.
    """
    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)

    period_rows = [r for r in filtered if str(_row_value(r, time_col) or "") == str(period)]
    label_col = _label_column_for(schema, geo_col)
    labels = _label_map(period_rows, geo_col, label_col)

    valued: list[tuple[str, float]] = []
    for row in period_rows:
        ref = _row_value(row, geo_col)
        val = _parse_obs(_row_value(row, obs_col))
        if ref is not None and val is not None:
            valued.append((str(ref), val))

    valued.sort(key=lambda x: x[1], reverse=not ascending)
    total = len(valued)
    top = valued[:n]

    rank_rows = [
        RankRow(
            rank=i + 1,
            ref_area=ref,
            ref_area_label=labels.get(ref),
            period=period,
            value=val,
        )
        for i, (ref, val) in enumerate(top)
    ]

    return RankResponse(
        idno=schema.idno,
        indicator_name=_indicator_name(rows),
        period=period,
        n=n,
        ascending=ascending,
        geo_column=geo_col,
        time_column=time_col,
        obs_column=obs_col,
        dimensions_applied=dims,
        rows=rank_rows,
        total_ref_areas=total,
    )


def get_extremes(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    dimensions: dict[str, str] | None = None,
) -> ExtremesResponse:
    """Find the global maximum and minimum observation across all periods and ref areas.

    Args:
        rows: Pre-fetched observation rows.
        schema: DSD schema for the indicator.
        dimensions: Optional dimension filters applied before analysis.
    """
    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)

    label_col = _label_column_for(schema, geo_col)
    labels = _label_map(filtered, geo_col, label_col)

    max_point: tuple[str, str, float] | None = None
    min_point: tuple[str, str, float] | None = None
    total = 0

    for row in filtered:
        ref = _row_value(row, geo_col)
        period = _row_value(row, time_col)
        val = _parse_obs(_row_value(row, obs_col))
        if ref is None or period is None or val is None:
            continue
        total += 1
        if max_point is None or val > max_point[2]:
            max_point = (str(ref), str(period), val)
        if min_point is None or val < min_point[2]:
            min_point = (str(ref), str(period), val)

    def _ep(t: tuple[str, str, float] | None) -> ExtremePoint | None:
        if t is None:
            return None
        return ExtremePoint(ref_area=t[0], ref_area_label=labels.get(t[0]), period=t[1], value=t[2])

    return ExtremesResponse(
        idno=schema.idno,
        indicator_name=_indicator_name(rows),
        geo_column=geo_col,
        time_column=time_col,
        obs_column=obs_col,
        dimensions_applied=dims,
        maximum=_ep(max_point),
        minimum=_ep(min_point),
        total_observations=total,
    )


def compare(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    ref_areas: list[str],
    dimensions: dict[str, str] | None = None,
) -> CompareResponse:
    """Build a pivoted time-series table for a set of ref areas.

    Each row in the result is one time period with a value per requested ref area.

    Args:
        rows: Pre-fetched observation rows (should already be filtered to the
              desired period range if needed).
        schema: DSD schema for the indicator.
        ref_areas: List of ref area codes to include.
        dimensions: Optional dimension filters applied before pivoting.
    """
    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)

    label_col = _label_column_for(schema, geo_col)
    labels = _label_map(filtered, geo_col, label_col)

    # Only keep rows for requested ref areas
    ref_set = set(ref_areas)
    relevant = [r for r in filtered if str(_row_value(r, geo_col) or "") in ref_set]

    # Pivot: {period: {ref_area: value}}
    pivot: dict[str, dict[str, float | None]] = {}
    for row in relevant:
        ref = str(_row_value(row, geo_col) or "")
        period = str(_row_value(row, time_col) or "")
        val = _parse_obs(_row_value(row, obs_col))
        pivot.setdefault(period, {})[ref] = val

    compare_rows = [
        CompareRow(period=period, values={ra: pivot[period].get(ra) for ra in ref_areas})
        for period in sorted(pivot)
    ]

    return CompareResponse(
        idno=schema.idno,
        indicator_name=_indicator_name(rows),
        ref_areas=ref_areas,
        ref_area_labels={ra: labels[ra] for ra in ref_areas if ra in labels},
        geo_column=geo_col,
        time_column=time_col,
        obs_column=obs_col,
        dimensions_applied=dims,
        rows=compare_rows,
    )


def summarize(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    period: str,
    dimensions: dict[str, str] | None = None,
) -> SummarizeResponse:
    """Compute descriptive statistics across all ref areas for one period.

    Args:
        rows: Pre-fetched observation rows.
        schema: DSD schema for the indicator.
        period: The time period to summarize.
        dimensions: Optional dimension filters applied before analysis.
    """
    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)

    period_rows = [r for r in filtered if str(_row_value(r, time_col) or "") == str(period)]

    valued: list[tuple[str, float]] = []
    for row in period_rows:
        ref = _row_value(row, geo_col)
        val = _parse_obs(_row_value(row, obs_col))
        if ref is not None and val is not None:
            valued.append((str(ref), val))

    if not valued:
        return SummarizeResponse(
            idno=schema.idno,
            indicator_name=_indicator_name(rows),
            period=period,
            geo_column=geo_col,
            obs_column=obs_col,
            dimensions_applied=dims,
            stats=SummaryStats(count=0),
            error=f"No data found for period '{period}' after applying filters.",
        )

    vals = [v for _, v in valued]
    ref_of_max = max(valued, key=lambda x: x[1])[0]
    ref_of_min = min(valued, key=lambda x: x[1])[0]

    stats = SummaryStats(
        count=len(vals),
        min_value=min(vals),
        max_value=max(vals),
        mean=statistics.mean(vals),
        median=statistics.median(vals),
        std=statistics.stdev(vals) if len(vals) > 1 else 0.0,
        min_ref_area=ref_of_min,
        max_ref_area=ref_of_max,
    )

    return SummarizeResponse(
        idno=schema.idno,
        indicator_name=_indicator_name(rows),
        period=period,
        geo_column=geo_col,
        obs_column=obs_col,
        dimensions_applied=dims,
        stats=stats,
    )


def growth(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    ref_areas: list[str] | None = None,
    base_period: str,
    end_period: str,
    dimensions: dict[str, str] | None = None,
) -> GrowthResponse:
    """Compute period-over-period change for each ref area.

    Args:
        rows: Pre-fetched observation rows spanning at least base and end periods.
        schema: DSD schema for the indicator.
        ref_areas: Ref area codes to include. If None, all ref areas are used.
        base_period: Starting period string (must match TIME_PERIOD values exactly).
        end_period: Ending period string.
        dimensions: Optional dimension filters applied before analysis.
    """
    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)

    label_col = _label_column_for(schema, geo_col)
    labels = _label_map(filtered, geo_col, label_col)

    # Index by (ref_area, period) → value
    index: dict[tuple[str, str], float] = {}
    for row in filtered:
        ref = _row_value(row, geo_col)
        period = _row_value(row, time_col)
        val = _parse_obs(_row_value(row, obs_col))
        if ref is None or period is None or val is None:
            continue
        key = (str(ref), str(period))
        index[key] = val

    # Collect ref areas that have data in either period
    all_refs = {k[0] for k in index if k[1] in {base_period, end_period}}
    if ref_areas:
        all_refs = all_refs & set(ref_areas)

    growth_rows: list[GrowthRow] = []
    for ref in sorted(all_refs):
        base_val = index.get((ref, base_period))
        end_val = index.get((ref, end_period))
        abs_change = (end_val - base_val) if (end_val is not None and base_val is not None) else None
        pct = (abs_change / abs(base_val) * 100) if (abs_change is not None and base_val and base_val != 0) else None
        growth_rows.append(GrowthRow(
            ref_area=ref,
            ref_area_label=labels.get(ref),
            base_value=base_val,
            end_value=end_val,
            absolute_change=abs_change,
            pct_change=pct,
        ))

    return GrowthResponse(
        idno=schema.idno,
        indicator_name=_indicator_name(rows),
        base_period=base_period,
        end_period=end_period,
        geo_column=geo_col,
        obs_column=obs_col,
        dimensions_applied=dims,
        rows=growth_rows,
    )
