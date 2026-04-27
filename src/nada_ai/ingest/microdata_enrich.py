"""
Optional enrichment for microdata discoverability (variable labels, groups).

Walks common DDI-like shapes when present; safe no-op when fields are absent.
"""

from __future__ import annotations

from typing import Any


def extract_microdata_enrichment_from_raw(metadata: dict) -> str:
    """Collect searchable text from variable/group structures in raw catalog JSON."""
    chunks: list[str] = []

    def add_strings(obj: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(obj, str) and len(obj) > 1:
            s = obj.strip()
            if s and s not in chunks:
                chunks.append(s)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in ("labl", "label", "name", "var_lab", "title") and isinstance(v, str):
                    add_strings(v, depth + 1)
                elif "variable" in lk or "group" in lk or lk in ("data_dictionary", "file_description"):
                    add_strings(v, depth + 1)
                elif isinstance(v, (list, dict)):
                    add_strings(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:500]:
                add_strings(item, depth + 1)

    # Prefer known top-level regions
    for key in ("data_dictionary", "study_description", "study_desc", "file_description"):
        if key in metadata:
            add_strings(metadata[key], 0)

    if not chunks:
        add_strings(metadata, 0)

    # De-dupe while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return "\n".join(ordered[:2000])


def append_microdata_discoverability_text(metadata_type: str | None, meta: dict[str, Any], raw_metadata: dict | None) -> str:
    """Return extra text to append to page_content for microdata."""
    if metadata_type != "microdata" or not raw_metadata:
        return ""
    extra = extract_microdata_enrichment_from_raw(raw_metadata)
    # Also honor explicit list fields if ever attached to filter payload
    for key in ("variable_labels", "variable_groups"):
        val = meta.get(key)
        if isinstance(val, list):
            extra += "\n" + "\n".join(str(x) for x in val if x)
        elif isinstance(val, str) and val.strip():
            extra += "\n" + val.strip()
    return extra.strip()
