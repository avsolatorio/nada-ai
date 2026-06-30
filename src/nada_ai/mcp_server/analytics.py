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

import logging
import math
import statistics
from typing import Any

import numpy as np
from scipy.stats import median_abs_deviation

_logger = logging.getLogger(__name__)

from nada_ai.nada.models import (
    AggregateResponse,
    AggregateRow,
    BenchmarkResponse,
    BenchmarkRow,
    CompareResponse,
    CompareRow,
    CorrelatePoint,
    CorrelateResponse,
    CoverageResponse,
    CoverageSummary,
    ExtremePoint,
    ExtremesResponse,
    GrowthResponse,
    GrowthRow,
    IndicatorSchema,
    JoinResponse,
    JoinRow,
    OutlierRow,
    OutliersResponse,
    RankResponse,
    RankRow,
    SummaryStats,
    SummarizeResponse,
    TimeseriesDataRow,
    TrendResponse,
    TrendRow,
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
        name = row.INDICATOR_NAME or (row.model_extra or {}).get("INDICATOR_NAME")
        if name:
            return str(name)
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

    # Deduplicate: for disaggregated indicators, multiple rows may share the same ref_area.
    # Use the first value seen (caller should pass dimensions= to pre-filter).
    valued: list[tuple[str, float]] = []
    seen_refs: set[str] = set()
    for row in period_rows:
        ref = _row_value(row, geo_col)
        val = _parse_obs(_row_value(row, obs_col))
        if ref is not None and val is not None:
            ref_str = str(ref)
            if ref_str not in seen_refs:
                seen_refs.add(ref_str)
                valued.append((ref_str, val))

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
    # For disaggregated indicators (no dimensions filter applied), the same ref_area
    # may appear multiple times in a period. Log a warning and skip duplicates.
    pivot: dict[str, dict[str, float | None]] = {}
    for row in relevant:
        ref = str(_row_value(row, geo_col) or "")
        period = str(_row_value(row, time_col) or "")
        val = _parse_obs(_row_value(row, obs_col))
        period_pivot = pivot.setdefault(period, {})
        if ref in period_pivot:
            _logger.debug(
                "compare: duplicate ref_area '%s' for period '%s' — "
                "pass dimensions= to filter disaggregated indicators", ref, period
            )
        else:
            period_pivot[ref] = val

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
        pct = (abs_change / abs(base_val) * 100) if (abs_change is not None and base_val) else None
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


# ---------------------------------------------------------------------------
# Shared math helpers
# ---------------------------------------------------------------------------


def _period_to_numeric(period: str) -> float | None:
    """Convert a period string to a float for regression. Returns None on failure."""
    s = period.strip()
    try:
        # Annual: "2022"
        if len(s) == 4 and s.isdigit():
            return float(s)
        # Quarterly: "2022-Q1" or "2022Q1"
        if len(s) >= 6 and ("Q" in s.upper()):
            parts = s.upper().replace("-", "").replace("Q", " ").split()
            if len(parts) == 2:
                return float(parts[0]) + (float(parts[1]) - 1) / 4
        # Monthly: "2022-01"
        if len(s) == 7 and s[4] == "-":
            return float(s[:4]) + (float(s[5:]) - 1) / 12
        return float(s)
    except (ValueError, IndexError):
        return None


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared) for simple OLS."""
    n = len(xs)
    if n < 2:
        return (0.0, ys[0] if ys else 0.0, 0.0)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_yy = sum((y - y_mean) ** 2 for y in ys)
    if ss_xx == 0:
        return (0.0, y_mean, 0.0)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_squared = (ss_xy ** 2 / (ss_xx * ss_yy)) if ss_yy > 0 else 0.0
    return (slope, intercept, r_squared)


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient; None when undefined."""
    n = len(xs)
    if n < 2:
        return None
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_yy = sum((y - y_mean) ** 2 for y in ys)
    denom = math.sqrt(ss_xx * ss_yy)
    return ss_xy / denom if denom > 0 else None


# ---------------------------------------------------------------------------
# Correlate
# ---------------------------------------------------------------------------


def correlate(
    rows1: list[TimeseriesDataRow],
    schema1: IndicatorSchema,
    rows2: list[TimeseriesDataRow],
    schema2: IndicatorSchema,
    *,
    period: str,
    dimensions1: dict[str, str] | None = None,
    dimensions2: dict[str, str] | None = None,
) -> CorrelateResponse:
    """Pearson correlation between two indicators across ref areas for a given period.

    Args:
        rows1: Observation rows for indicator 1.
        schema1: DSD schema for indicator 1.
        rows2: Observation rows for indicator 2.
        schema2: DSD schema for indicator 2.
        period: Time period to slice (must match TIME_PERIOD values exactly).
        dimensions1: Dimension filters applied to indicator 1 before correlation.
        dimensions2: Dimension filters applied to indicator 2 before correlation.
    """
    geo_col1, time_col1, obs_col1 = _require_columns(schema1)
    geo_col2, time_col2, obs_col2 = _require_columns(schema2)

    filtered1 = _apply_dimension_filter(rows1, schema1, dimensions1 or {})
    filtered2 = _apply_dimension_filter(rows2, schema2, dimensions2 or {})

    label_col = _label_column_for(schema1, geo_col1)
    labels = _label_map(filtered1, geo_col1, label_col)

    def _period_values(rows: list[TimeseriesDataRow], geo_col: str, time_col: str, obs_col: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for row in rows:
            if str(_row_value(row, time_col) or "") != str(period):
                continue
            ref = _row_value(row, geo_col)
            val = _parse_obs(_row_value(row, obs_col))
            if ref is not None and val is not None:
                result[str(ref)] = val
        return result

    vals1 = _period_values(filtered1, geo_col1, time_col1, obs_col1)
    vals2 = _period_values(filtered2, geo_col2, time_col2, obs_col2)

    common = sorted(vals1.keys() & vals2.keys())
    xs = [vals1[r] for r in common]
    ys = [vals2[r] for r in common]

    return CorrelateResponse(
        idno1=schema1.idno,
        idno2=schema2.idno,
        indicator_name1=_indicator_name(rows1),
        indicator_name2=_indicator_name(rows2),
        period=period,
        geo_column=geo_col1,
        n=len(common),
        pearson_r=_pearson_r(xs, ys),
        rows=[
            CorrelatePoint(
                ref_area=r,
                ref_area_label=labels.get(r),
                value1=vals1[r],
                value2=vals2[r],
            )
            for r in common
        ],
    )


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


def _modified_zscores(vals: list[float]) -> tuple[list[float], float, float]:
    """Return (modified_z_scores, median, MAD).

    Modified Z-score = 0.6745 * (x - median) / MAD.
    Robust to outliers because it uses median and MAD rather than mean and std.
    When MAD == 0 (all values identical or constant), falls back to plain Z-score.
    """
    arr = np.array(vals, dtype=float)
    med = float(np.median(arr))
    mad = float(median_abs_deviation(arr, scale=1.0))
    if mad > 0:
        zs = [0.6745 * (v - med) / mad for v in vals]
    else:
        # Fall back to mean/std when MAD is 0
        std = float(np.std(arr))
        zs = [(v - med) / std if std > 0 else 0.0 for v in vals]
    return zs, med, mad


def _iqr_scores(vals: list[float]) -> tuple[list[float], float, float, float, float]:
    """Return (scores, Q1, Q3, lower_fence, upper_fence).

    Score is signed distance outside the fence normalised by IQR (0 if inside).
    IQR fences: [Q1 - 1.5*IQR, Q3 + 1.5*IQR].  Threshold of 1.0 flags anything
    beyond 1 IQR-width outside the fence (i.e. beyond 3.0 IQR from Q1/Q3).
    """
    arr = np.array(vals, dtype=float)
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        scores = [
            (v - hi) / iqr if v > hi else (lo - v) / iqr if v < lo else 0.0
            for v in vals
        ]
    else:
        lo, hi = q1, q3
        scores = [0.0] * len(vals)
    return scores, q1, q3, lo, hi


def _lowess_trend_residuals(
    xs: list[float], ys: list[float], frac: float = 0.4
) -> list[float]:
    """Fit a LOWESS trend and return residuals (observed - fitted)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    arr_x = np.array(xs, dtype=float)
    arr_y = np.array(ys, dtype=float)
    # lowess returns (x_sorted, y_fitted) sorted by x
    fitted_sorted = lowess(arr_y, arr_x, frac=frac, return_sorted=True)
    # Build a lookup from x -> fitted_y
    fit_map = {float(fx): float(fy) for fx, fy in fitted_sorted}
    return [y - fit_map.get(x, y) for x, y in zip(xs, ys)]


def detect_outliers(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    period: str | None = None,
    ref_area: str | None = None,
    method: str = "modified_zscore",
    threshold: float | None = None,
    dimensions: dict[str, str] | None = None,
) -> OutliersResponse:
    """Detect outliers in two modes using three methods.

    **Modes** — supply exactly one of ``period`` / ``ref_area``:

    - *cross_section*: compares all ref areas for a fixed period.
    - *longitudinal*: compares all time periods for a fixed ref area.

    **Methods**:

    - ``"modified_zscore"`` *(default)*: MAD-based modified Z-score.
      More robust than plain Z-score because outliers don't inflate the spread.
      Default threshold 3.5.
    - ``"iqr"``: Tukey fences (Q1 − 1.5·IQR, Q3 + 1.5·IQR).
      Score is how many IQR-widths outside the fence a point sits (0 = inside).
      Default threshold 0 (any point outside the fence is flagged).
    - ``"trend_residual"`` *(longitudinal only)*: fits a LOWESS trend to the
      time series and applies modified Z-score on the residuals. Correctly
      distinguishes "unusual given the trend" from "high overall level".
      Falls back to ``modified_zscore`` for cross-section mode.
      Default threshold 3.5.

    Args:
        rows: Pre-fetched observation rows.
        schema: DSD schema for the indicator.
        period: Time period to analyse (cross-section mode).
        ref_area: Ref area code to analyse across time (longitudinal mode).
        method: Detection method — ``"modified_zscore"``, ``"iqr"``, or ``"trend_residual"``.
        threshold: Flagging threshold (method-dependent default if None).
        dimensions: Optional dimension filters.
    """
    if (period is None) == (ref_area is None):
        raise ValueError("Exactly one of 'period' or 'ref_area' must be provided.")

    _method = method if method in ("modified_zscore", "iqr", "trend_residual") else "modified_zscore"
    # Default thresholds per method
    if threshold is None:
        _threshold = 0.0 if _method == "iqr" else 3.5
    else:
        _threshold = threshold

    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)

    # ------------------------------------------------------------------
    # Cross-section
    # ------------------------------------------------------------------
    if period is not None:
        if _method == "trend_residual":
            _method = "modified_zscore"  # no trend in cross-section

        slice_rows = [r for r in filtered if str(_row_value(r, time_col) or "") == str(period)]
        label_col = _label_column_for(schema, geo_col)
        labels = _label_map(slice_rows, geo_col, label_col)

        valued: list[tuple[str, float]] = []
        seen: set[str] = set()
        for row in slice_rows:
            ref = _row_value(row, geo_col)
            val = _parse_obs(_row_value(row, obs_col))
            if ref is not None and val is not None:
                ref_str = str(ref)
                if ref_str not in seen:
                    seen.add(ref_str)
                    valued.append((ref_str, val))

        if not valued:
            return OutliersResponse(
                idno=schema.idno, indicator_name=_indicator_name(rows),
                mode="cross_section", method=_method, period=period,
                geo_column=geo_col, obs_column=obs_col,
                threshold=_threshold, dimensions_applied=dims,
                error=f"No data found for period '{period}'.",
            )

        vals = [v for _, v in valued]

        if _method == "iqr":
            scores, q1, q3, lo, hi = _iqr_scores(vals)
            peer_mean, peer_std = float(np.mean(vals)), float(np.std(vals))
            outlier_rows: list[OutlierRow] = []
            for (ref, val), score in sorted(
                zip(valued, scores), key=lambda x: x[1], reverse=True
            ):
                outlier_rows.append(OutlierRow(
                    ref_area=ref, ref_area_label=labels.get(ref),
                    value=val, z_score=round(score, 4),
                    is_outlier=score > _threshold,
                ))
        else:  # modified_zscore
            zscores, med, mad = _modified_zscores(vals)
            peer_mean, peer_std = med, mad
            outlier_rows = []
            for (ref, val), z in sorted(
                zip(valued, zscores), key=lambda x: abs(x[1]), reverse=True
            ):
                outlier_rows.append(OutlierRow(
                    ref_area=ref, ref_area_label=labels.get(ref),
                    value=val, z_score=round(z, 4),
                    is_outlier=abs(z) >= _threshold,
                ))

        return OutliersResponse(
            idno=schema.idno, indicator_name=_indicator_name(rows),
            mode="cross_section", method=_method, period=period,
            geo_column=geo_col, obs_column=obs_col,
            threshold=_threshold, peer_mean=peer_mean, peer_std=peer_std,
            n_outliers=sum(1 for r in outlier_rows if r.is_outlier),
            dimensions_applied=dims, rows=outlier_rows,
        )

    # ------------------------------------------------------------------
    # Longitudinal
    # ------------------------------------------------------------------
    ref_area_str = str(ref_area)
    slice_rows = [
        r for r in filtered
        if str(_row_value(r, geo_col) or "") == ref_area_str
    ]

    valued_t: list[tuple[str, float]] = []
    seen_t: set[str] = set()
    for row in slice_rows:
        t = _row_value(row, time_col)
        val = _parse_obs(_row_value(row, obs_col))
        if t is not None and val is not None:
            t_str = str(t)
            if t_str not in seen_t:
                seen_t.add(t_str)
                valued_t.append((t_str, val))

    if not valued_t:
        return OutliersResponse(
            idno=schema.idno, indicator_name=_indicator_name(rows),
            mode="longitudinal", method=_method, ref_area=ref_area_str,
            geo_column=geo_col, obs_column=obs_col,
            threshold=_threshold, dimensions_applied=dims,
            error=f"No data found for ref area '{ref_area_str}'.",
        )

    valued_t.sort(key=lambda x: x[0])
    ts = [t for t, _ in valued_t]
    vals_t = [v for _, v in valued_t]

    if _method == "trend_residual":
        # Convert period labels to numeric x-axis for LOWESS
        xs = [_period_to_numeric(t) or float(i) for i, t in enumerate(ts)]
        # LOWESS needs at least a few points; fall back gracefully
        frac = max(0.3, min(0.6, 5.0 / len(vals_t))) if len(vals_t) >= 4 else 1.0
        try:
            residuals = _lowess_trend_residuals(xs, vals_t, frac=frac)
        except Exception:
            residuals = [v - statistics.mean(vals_t) for v in vals_t]
        # Fit map: period -> fitted value
        fit_vals = [v - r for v, r in zip(vals_t, residuals)]
        zscores, med, mad = _modified_zscores(residuals)
        peer_mean, peer_std = med, mad
        outlier_rows_t: list[OutlierRow] = []
        for (t_str, val), resid, fit, z in sorted(
            zip(valued_t, residuals, fit_vals, zscores),
            key=lambda x: abs(x[3]), reverse=True,
        ):
            outlier_rows_t.append(OutlierRow(
                period=t_str, value=val,
                trend_value=round(fit, 6),
                residual=round(resid, 6),
                z_score=round(z, 4),
                is_outlier=abs(z) >= _threshold,
            ))

    elif _method == "iqr":
        scores, q1, q3, lo, hi = _iqr_scores(vals_t)
        peer_mean, peer_std = float(np.mean(vals_t)), float(np.std(vals_t))
        outlier_rows_t = []
        for (t_str, val), score in sorted(
            zip(valued_t, scores), key=lambda x: x[1], reverse=True
        ):
            outlier_rows_t.append(OutlierRow(
                period=t_str, value=val,
                z_score=round(score, 4), is_outlier=score > _threshold,
            ))

    else:  # modified_zscore
        zscores, med, mad = _modified_zscores(vals_t)
        peer_mean, peer_std = med, mad
        outlier_rows_t = []
        for (t_str, val), z in sorted(
            zip(valued_t, zscores), key=lambda x: abs(x[1]), reverse=True
        ):
            outlier_rows_t.append(OutlierRow(
                period=t_str, value=val,
                z_score=round(z, 4), is_outlier=abs(z) >= _threshold,
            ))

    return OutliersResponse(
        idno=schema.idno, indicator_name=_indicator_name(rows),
        mode="longitudinal", method=_method, ref_area=ref_area_str,
        geo_column=geo_col, obs_column=obs_col,
        threshold=_threshold, peer_mean=peer_mean, peer_std=peer_std,
        n_outliers=sum(1 for r in outlier_rows_t if r.is_outlier),
        dimensions_applied=dims, rows=outlier_rows_t,
    )


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def trend(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    ref_areas: list[str] | None = None,
    dimensions: dict[str, str] | None = None,
) -> TrendResponse:
    """Fit a linear trend (OLS) per ref area over all available periods.

    Args:
        rows: Pre-fetched observation rows spanning the desired time window.
        schema: DSD schema for the indicator.
        ref_areas: Ref areas to include (default: all with sufficient data).
        dimensions: Optional dimension filters.
    """
    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)

    label_col = _label_column_for(schema, geo_col)
    labels = _label_map(filtered, geo_col, label_col)

    # Index: {ref_area: [(numeric_period, value), ...]}
    series: dict[str, list[tuple[float, float, str]]] = {}
    for row in filtered:
        ref = _row_value(row, geo_col)
        period_str = str(_row_value(row, time_col) or "")
        val = _parse_obs(_row_value(row, obs_col))
        t = _period_to_numeric(period_str)
        if ref is None or val is None or t is None:
            continue
        series.setdefault(str(ref), []).append((t, val, period_str))

    if ref_areas:
        series = {k: v for k, v in series.items() if k in set(ref_areas)}

    trend_rows: list[TrendRow] = []
    for ref in sorted(series):
        pts = sorted(series[ref], key=lambda x: x[0])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        period_strs = [p[2] for p in pts]
        slope, intercept, r2 = _linear_regression(xs, ys)
        if abs(slope) < 1e-10:
            direction = "stable"
        elif slope > 0:
            direction = "improving"
        else:
            direction = "declining"
        trend_rows.append(TrendRow(
            ref_area=ref,
            ref_area_label=labels.get(ref),
            slope=round(slope, 6),
            intercept=round(intercept, 6),
            r_squared=round(r2, 4),
            n_periods=len(pts),
            first_period=period_strs[0],
            last_period=period_strs[-1],
            direction=direction,
        ))

    trend_rows.sort(key=lambda r: abs(r.slope or 0), reverse=True)

    return TrendResponse(
        idno=schema.idno,
        indicator_name=_indicator_name(rows),
        geo_column=geo_col,
        obs_column=obs_col,
        dimensions_applied=dims,
        rows=trend_rows,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    ref_areas: list[str],
    period: str,
    dimensions: dict[str, str] | None = None,
) -> BenchmarkResponse:
    """Benchmark ref areas against all peers for a given period.

    Args:
        rows: Pre-fetched observation rows (should include peer ref areas).
        schema: DSD schema for the indicator.
        ref_areas: The ref areas to benchmark.
        period: The period to benchmark in.
        dimensions: Optional dimension filters.
    """
    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)
    period_rows = [r for r in filtered if str(_row_value(r, time_col) or "") == str(period)]

    label_col = _label_column_for(schema, geo_col)
    labels = _label_map(period_rows, geo_col, label_col)

    # Collect all peers (deduplicated)
    peer_vals: dict[str, float] = {}
    for row in period_rows:
        ref = _row_value(row, geo_col)
        val = _parse_obs(_row_value(row, obs_col))
        if ref is not None and val is not None:
            ref_str = str(ref)
            if ref_str not in peer_vals:
                peer_vals[ref_str] = val

    if not peer_vals:
        return BenchmarkResponse(
            idno=schema.idno,
            indicator_name=_indicator_name(rows),
            period=period,
            geo_column=geo_col,
            obs_column=obs_col,
            dimensions_applied=dims,
            error=f"No peer data found for period '{period}'.",
        )

    all_vals = list(peer_vals.values())
    sorted_vals = sorted(all_vals)
    peer_mean = statistics.mean(all_vals)
    peer_median = statistics.median(all_vals)
    peer_std = statistics.stdev(all_vals) if len(all_vals) > 1 else 0.0

    def _percentile_rank(val: float, sorted_peers: list[float]) -> float:
        below = sum(1 for v in sorted_peers if v < val)
        equal = sum(1 for v in sorted_peers if v == val)
        return round((below + 0.5 * equal) / len(sorted_peers) * 100, 1)

    bench_rows: list[BenchmarkRow] = []
    for ref in ref_areas:
        val = peer_vals.get(ref)
        if val is None:
            bench_rows.append(BenchmarkRow(ref_area=ref, ref_area_label=labels.get(ref)))
            continue
        z = (val - peer_mean) / peer_std if peer_std > 0 else 0.0
        bench_rows.append(BenchmarkRow(
            ref_area=ref,
            ref_area_label=labels.get(ref),
            value=val,
            percentile_rank=_percentile_rank(val, sorted_vals),
            z_score=round(z, 4),
            vs_mean=round(val - peer_mean, 4),
            vs_median=round(val - peer_median, 4),
        ))

    return BenchmarkResponse(
        idno=schema.idno,
        indicator_name=_indicator_name(rows),
        period=period,
        geo_column=geo_col,
        obs_column=obs_col,
        peer_count=len(peer_vals),
        peer_mean=peer_mean,
        peer_median=peer_median,
        peer_std=peer_std,
        dimensions_applied=dims,
        rows=bench_rows,
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def coverage(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    dimensions: dict[str, str] | None = None,
) -> CoverageResponse:
    """Summarise data availability per ref area.

    Args:
        rows: Pre-fetched observation rows.
        schema: DSD schema for the indicator.
        dimensions: Optional dimension filters.
    """
    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)

    label_col = _label_column_for(schema, geo_col)
    labels = _label_map(filtered, geo_col, label_col)

    # {ref_area: set of periods with data}
    data_map: dict[str, set[str]] = {}
    for row in filtered:
        ref = _row_value(row, geo_col)
        period = _row_value(row, time_col)
        val = _parse_obs(_row_value(row, obs_col))
        if ref is None or period is None or val is None:
            continue
        data_map.setdefault(str(ref), set()).add(str(period))

    all_periods = {p for periods in data_map.values() for p in periods}
    total_periods = len(all_periods)

    cov_rows: list[CoverageSummary] = []
    for ref in sorted(data_map):
        periods = sorted(data_map[ref])
        cov_rows.append(CoverageSummary(
            ref_area=ref,
            ref_area_label=labels.get(ref),
            n_periods=len(periods),
            first_period=periods[0] if periods else None,
            last_period=periods[-1] if periods else None,
            coverage_pct=round(len(periods) / total_periods * 100, 1) if total_periods > 0 else None,
        ))

    cov_rows.sort(key=lambda r: r.n_periods, reverse=True)

    return CoverageResponse(
        idno=schema.idno,
        indicator_name=_indicator_name(rows),
        geo_column=geo_col,
        time_column=time_col,
        total_periods=total_periods,
        total_ref_areas=len(data_map),
        dimensions_applied=dims,
        rows=cov_rows,
    )


# ---------------------------------------------------------------------------
# Join (cross-indicator)
# ---------------------------------------------------------------------------


def join_indicators(
    rows1: list[TimeseriesDataRow],
    schema1: IndicatorSchema,
    rows2: list[TimeseriesDataRow],
    schema2: IndicatorSchema,
    *,
    dimensions1: dict[str, str] | None = None,
    dimensions2: dict[str, str] | None = None,
) -> JoinResponse:
    """Align two indicators by (ref_area, period) and return merged rows.

    Args:
        rows1: Observation rows for indicator 1.
        schema1: DSD schema for indicator 1.
        rows2: Observation rows for indicator 2.
        schema2: DSD schema for indicator 2.
        dimensions1: Dimension filters for indicator 1.
        dimensions2: Dimension filters for indicator 2.
    """
    geo_col1, time_col1, obs_col1 = _require_columns(schema1)
    geo_col2, time_col2, obs_col2 = _require_columns(schema2)

    filtered1 = _apply_dimension_filter(rows1, schema1, dimensions1 or {})
    filtered2 = _apply_dimension_filter(rows2, schema2, dimensions2 or {})

    label_col = _label_column_for(schema1, geo_col1)
    labels = _label_map(filtered1, geo_col1, label_col)

    def _index(rows: list[TimeseriesDataRow], geo_col: str, time_col: str, obs_col: str) -> dict[tuple[str, str], float]:
        idx: dict[tuple[str, str], float] = {}
        for row in rows:
            ref = _row_value(row, geo_col)
            period = _row_value(row, time_col)
            val = _parse_obs(_row_value(row, obs_col))
            if ref is not None and period is not None and val is not None:
                key = (str(ref), str(period))
                if key not in idx:
                    idx[key] = val
        return idx

    idx1 = _index(filtered1, geo_col1, time_col1, obs_col1)
    idx2 = _index(filtered2, geo_col2, time_col2, obs_col2)

    all_keys = sorted(idx1.keys() | idx2.keys())
    join_rows = [
        JoinRow(
            ref_area=ref,
            ref_area_label=labels.get(ref),
            period=period,
            value1=idx1.get((ref, period)),
            value2=idx2.get((ref, period)),
        )
        for ref, period in all_keys
    ]

    return JoinResponse(
        idno1=schema1.idno,
        idno2=schema2.idno,
        indicator_name1=_indicator_name(rows1),
        indicator_name2=_indicator_name(rows2),
        geo_column=geo_col1,
        n_matched=sum(1 for r in join_rows if r.value1 is not None and r.value2 is not None),
        dimensions_applied1=dimensions1 or {},
        dimensions_applied2=dimensions2 or {},
        rows=join_rows,
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def aggregate(
    rows: list[TimeseriesDataRow],
    schema: IndicatorSchema,
    *,
    ref_areas: list[str] | None = None,
    dimensions: dict[str, str] | None = None,
) -> AggregateResponse:
    """Compute group-level statistics per period across a set of ref areas.

    Args:
        rows: Pre-fetched observation rows.
        schema: DSD schema for the indicator.
        ref_areas: Ref areas to aggregate (default: all).
        dimensions: Optional dimension filters.
    """
    geo_col, time_col, obs_col = _require_columns(schema)
    dims = dimensions or {}
    filtered = _apply_dimension_filter(rows, schema, dims)

    if ref_areas:
        ref_set = set(ref_areas)
        filtered = [r for r in filtered if str(_row_value(r, geo_col) or "") in ref_set]

    # {period: [values]}
    period_vals: dict[str, list[float]] = {}
    for row in filtered:
        period = _row_value(row, time_col)
        val = _parse_obs(_row_value(row, obs_col))
        if period is not None and val is not None:
            period_vals.setdefault(str(period), []).append(val)

    agg_rows: list[AggregateRow] = []
    for period in sorted(period_vals):
        vals = period_vals[period]
        agg_rows.append(AggregateRow(
            period=period,
            n_ref_areas=len(vals),
            mean=round(statistics.mean(vals), 4),
            median=round(statistics.median(vals), 4),
            total=round(sum(vals), 4),
            min_value=min(vals),
            max_value=max(vals),
            std=round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        ))

    used_refs: list[str] = sorted({
        str(_row_value(r, geo_col) or "")
        for r in filtered
        if _row_value(r, geo_col) is not None
    })

    return AggregateResponse(
        idno=schema.idno,
        indicator_name=_indicator_name(rows),
        ref_areas=ref_areas or used_refs,
        geo_column=geo_col,
        obs_column=obs_col,
        dimensions_applied=dims,
        rows=agg_rows,
    )
