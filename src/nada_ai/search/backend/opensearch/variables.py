"""The variable document: one per variable in the variable index, built from NADA's metadata-extract data.

The variable only, plus its study's title (to sort by). Everything else about the study (published, type, countries,
years, collections, repository, data access) is read from the study index at search time (see ``variables_search``),
so a change to the study never has to touch its variables. Source: ``build_variable_document``/
``build_variables_by_survey``/``build_variable_batch`` in NADA's ``Catalog_search_metadata_extract.php``.
"""

from __future__ import annotations

from typing import Any

from nada_ai.search.backend.opensearch.mapping import VARIABLE_TEXT_FIELDS
from nada_ai.search.cursor import sort_key


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def variable_to_source(uid: int, core_fields: dict[str, Any]) -> dict[str, Any]:
    """Variable document from one variable's extract ``core_fields``.

    ``sid`` is ``core_fields.catalog_id``, NADA's own ``surveys.id`` (the study index's ``_id``), so a variable hit
    resolves to its study without another lookup. Fields NADA leaves empty are omitted.
    """
    sid = _as_int(core_fields.get("catalog_id"))
    if sid is None:
        raise ValueError("core_fields has no catalog_id")
    source: dict[str, Any] = {"uid": int(uid), "sid": sid}
    for field in VARIABLE_TEXT_FIELDS:
        text = str(core_fields.get(field) or "").strip()
        if text:
            source[field] = text
    title = str(core_fields.get("title") or "").strip()
    if title:
        source["title"] = sort_key(title)  # the study title is only sorted on, never returned
    return source


def variable_bulk_action(index: str, uid: int, core_fields: dict[str, Any]) -> dict[str, Any]:
    """``bulk`` action that (re)writes the variable document; ``_id`` is the ``uid``, so a re-index replaces it."""
    return {
        "_op_type": "index",
        "_index": index,
        "_id": str(int(uid)),
        "_source": variable_to_source(uid, core_fields),
    }
