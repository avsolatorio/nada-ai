"""Shared schema-and-data fetch helper for single-indicator analytics tools."""

from __future__ import annotations

from nada_ai.nada import api as nada_api
from nada_ai.nada.models import IndicatorSchema


async def fetch_schema_and_data(
    idno: str,
    *,
    from_year: int | None = None,
    to_year: int | None = None,
    country_codes: list[str] | None = None,
    geo_column: str | None = None,
    dimensions: dict[str, str] | None = None,
) -> tuple[IndicatorSchema | None, list | None, str | None]:
    """Fetch schema then timeseries rows for a single indicator.

    Returns ``(schema, rows, error)``.
    - Schema failure: ``schema`` is ``None``, ``rows`` is ``None``.
    - Data failure:  ``schema`` is populated, ``rows`` is ``None``.
    - Success:       ``error`` is ``None``.
    """
    schema_resp = await nada_api.get_indicator_schema(idno)
    if schema_resp.error or not schema_resp.schema_:
        return None, None, schema_resp.error or "Schema unavailable"
    schema = schema_resp.schema_
    resolved_geo = geo_column or schema.geo_column or "COUNTRY_CODE"
    data = await nada_api.get_all_timeseries_data(
        idno,
        from_year=from_year,
        to_year=to_year,
        country_codes=country_codes,
        geo_column=resolved_geo,
        dimensions=dimensions,
    )
    if data.error:
        return schema, None, data.error
    return schema, data.data, None
