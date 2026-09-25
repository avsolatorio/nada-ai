"""The citation document: one per citation in the citation index, built from NADA's metadata-extract data.

Same shape as ``variables.py``: the source is ``core_fields`` + ``metadata`` + ``filters`` of NADA's
``build_citation_document`` (``Catalog_search_metadata_extract.php``).
"""

from __future__ import annotations

import json
from typing import Any

from nada_ai.search.backend.opensearch.mapping import CITATION_TEXT_FIELDS


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_authors(raw: Any) -> str:
    """Author names as searchable text.

    NADA stores a citation's authors as a JSON list of ``{lname, fname, initial}``; that is flattened to
    ``"Fname Lname; Fname Lname"`` so a search for a name matches, instead of matching JSON keys. A value that is not
    that JSON list (plain text) is kept as it is.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        people = json.loads(raw)
    except ValueError:
        return raw.strip()
    if not isinstance(people, list):
        return raw.strip()
    names = []
    for person in people:
        if isinstance(person, dict):
            name = " ".join(str(person.get(k) or "").strip() for k in ("fname", "initial", "lname")).strip()
            name = " ".join(name.split())
            if name:
                names.append(name)
    return "; ".join(names)


def citation_to_source(
    core_fields: dict[str, Any], metadata: dict[str, Any], filters: dict[str, Any]
) -> dict[str, Any]:
    """Citation document from one citation's extract ``core_fields``, ``metadata`` and ``filters``.

    Fields NADA leaves empty are omitted.
    """
    citation_id = _as_int(core_fields.get("citation_id"))
    if citation_id is None:
        raise ValueError("core_fields has no citation_id")
    source: dict[str, Any] = {"citation_id": citation_id}
    uuid = str(core_fields.get("citation_uuid") or "").strip()
    if uuid:
        source["uuid"] = uuid
    for field in CITATION_TEXT_FIELDS:
        raw = metadata.get(field) if field in metadata else core_fields.get(field)
        text = format_authors(raw) if field == "authors" else str(raw or "").strip()
        if text:
            source[field] = text
    if source.get("title"):
        source["title_sort"] = source["title"]
    doi = str(core_fields.get("doi") or "").strip()
    if doi:
        source["doi"] = doi
    ctype = str(filters.get("ctype") or "").strip()
    if ctype:
        source["ctype"] = ctype
    year = _as_int(filters.get("pub_date"))
    if year:
        source["pub_year"] = year
    source["published"] = _as_int(filters.get("published")) or 0
    return source


def citation_bulk_action(index: str, citation: dict[str, Any]) -> dict[str, Any]:
    """``bulk`` action that (re)writes the citation document; ``_id`` is the citation id, so a re-index replaces it."""
    source = citation_to_source(citation["core_fields"], citation.get("metadata") or {}, citation.get("filters") or {})
    return {"_op_type": "index", "_index": index, "_id": str(source["citation_id"]), "_source": source}
