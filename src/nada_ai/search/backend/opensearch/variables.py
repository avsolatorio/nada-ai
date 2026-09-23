"""The variable document: one per variable in the variable index, built from NADA's metadata-extract data.

Mirrors ``studies.py``'s shape, but the source is flat (``core_fields`` + ``filters``, no nested metadata/facets):
see ``build_variable_document``/``build_variables_by_survey``/``build_variable_batch`` in NADA's
``Catalog_search_metadata_extract.php``.
"""

from __future__ import annotations

from typing import Any

from nada_ai.search.backend.opensearch.mapping import VARIABLE_TEXT_FIELDS

_KEYWORD_CORE_FIELDS = ("fid", "vid", "idno", "title", "nation", "dataset_type")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def variable_to_source(uid: int, core_fields: dict[str, Any], filters: dict[str, Any]) -> dict[str, Any]:
    """Variable document from one variable's extract ``core_fields`` and ``filters``.

    ``sid`` is NADA's own ``surveys.id`` (the study index's ``_id``), so a variable hit resolves to its study without
    another lookup. Fields NADA leaves empty are omitted.
    """
    sid = _as_int(core_fields.get("sid"))
    if sid is None:
        raise ValueError("core_fields has no sid")
    source: dict[str, Any] = {"uid": int(uid), "sid": sid}
    for field in VARIABLE_TEXT_FIELDS:
        text = str(core_fields.get(field) or "").strip()
        if text:
            source[field] = text
    for field in _KEYWORD_CORE_FIELDS:
        value = str(core_fields.get(field) or "").strip()
        if value:
            source[field] = value
    source["published"] = _as_int(filters.get("published")) or 0
    for field in ("year_start", "year_end"):
        number = _as_int(filters.get(field))
        if number is not None:
            source[field] = number
    countries = filters.get("countries")
    if isinstance(countries, list) and countries:
        source["countries"] = [n for c in countries if (n := _as_int(c)) is not None]
    return source


def variable_bulk_action(index: str, uid: int, core_fields: dict[str, Any], filters: dict[str, Any]) -> dict[str, Any]:
    """``bulk`` action that (re)writes the variable document; ``_id`` is the ``uid``, so a re-index replaces it."""
    return {
        "_op_type": "index",
        "_index": index,
        "_id": str(int(uid)),
        "_source": variable_to_source(uid, core_fields, filters),
    }
