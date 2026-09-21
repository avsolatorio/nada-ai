"""The study document: one per study in the study index, built from NADA's metadata-extract data."""

from __future__ import annotations

from typing import Any

from nada_ai.search.backend.opensearch.mapping import FILTER_FACETS_KEY, STUDY_TEXT_FIELDS
from nada_ai.search.dynamic_filters import normalize_external_filters, normalized_to_facets_map

_INT_FIELDS = ("year_start", "year_end", "created", "changed", "total_views", "total_downloads", "varcount")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def study_to_source(sid: int, core_fields: dict[str, Any], filters: dict[str, Any]) -> dict[str, Any]:
    """Study document from a study's extract ``core_fields`` and ``filters``.

    ``idno`` is NADA's own ``surveys.idno`` (not the record's schema idno, which can differ). ``filter_facets``
    holds every filter key NADA emits, one flat field per key. Fields NADA leaves empty are omitted.
    """
    idno = str(core_fields.get("idno") or "").strip()
    if not idno:
        raise ValueError("core_fields has no idno")
    source: dict[str, Any] = {"sid": int(sid), "idno": idno}
    for field in STUDY_TEXT_FIELDS:
        text = str(core_fields.get(field) or "").strip()
        if text:
            source[field] = text
    if "title" in source:
        source["title_sort"] = source["title"]
    if "nation" in source:
        source["nation_sort"] = source["nation"]
    for field in _INT_FIELDS:
        number = _as_int(core_fields.get(field))
        if number is not None:
            source[field] = number
    source[FILTER_FACETS_KEY] = normalized_to_facets_map(normalize_external_filters(filters))
    return source


def study_bulk_action(index: str, sid: int, core_fields: dict[str, Any], filters: dict[str, Any]) -> dict[str, Any]:
    """``bulk`` action that (re)writes the study document; ``_id`` is the ``sid``, so a re-index replaces it."""
    return {
        "_op_type": "index",
        "_index": index,
        "_id": str(int(sid)),
        "_source": study_to_source(sid, core_fields, filters),
    }


def sids_for_idnos(client: Any, studies_index: str, idnos: list[str]) -> dict[str, int]:
    """``{idno: sid}`` for the given NADA idnos, from the study index (which stores NADA's own idno).

    Chunk documents carry the record's schema idno, which can differ, so anything that must reach every chunk of
    a study by NADA idno goes through the ``sid`` found here. Idnos with no study document are absent.
    """
    if not idnos:
        return {}
    found = client.search(
        index=studies_index,
        body={"size": len(idnos), "_source": ["sid", "idno"], "query": {"terms": {"idno": idnos}}},
        ignore_unavailable=True,
    )
    return {
        hit["_source"]["idno"]: int(hit["_source"]["sid"]) for hit in found.get("hits", {}).get("hits", [])
    }
