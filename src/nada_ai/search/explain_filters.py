"""Explain helpers: how request filters relate to a sample document (no LLM)."""

from __future__ import annotations

from typing import Any


def _as_set(v: Any) -> set[Any]:
    if v is None:
        return set()
    if isinstance(v, list):
        return set(v)
    return {v}


def _terms_overlap(doc_vals: Any, filter_vals: list[Any]) -> bool:
    if not filter_vals:
        return True
    d = _as_set(doc_vals)
    return bool(d & set(filter_vals))


def compute_filter_match(sample: dict[str, Any], filters: dict[str, Any] | None) -> dict[str, Any]:
    """Return per-field match booleans for filters against ``sample`` (subset of _source / payload).

    Mirrors the intent of ``build_filters`` / Qdrant ``Filter`` (conjunctive). ``idno`` in
    ``filters`` is evaluated against ``sample['idno']`` when present.
    """
    if not filters:
        return {"all_matched": True, "per_field": {}, "notes": "no filters"}

    per: dict[str, Any] = {}
    ok = True

    if "type" in filters:
        m = sample.get("type") == filters["type"]
        per["type"] = {"expected": filters["type"], "actual": sample.get("type"), "matched": m}
        ok &= m

    if "idno" in filters:
        m = sample.get("idno") == filters["idno"]
        per["idno"] = {"expected": filters["idno"], "actual": sample.get("idno"), "matched": m}
        ok &= m

    if "idnos" in filters:
        idnos = list(filters["idnos"])
        m = sample.get("idno") in idnos
        per["idnos"] = {"expected": idnos, "actual": sample.get("idno"), "matched": m}
        ok &= m

    if "geographies" in filters:
        fg = list(filters["geographies"])
        m = _terms_overlap(sample.get("geographies"), fg)
        per["geographies"] = {"expected": fg, "actual": sample.get("geographies"), "matched": m}
        ok &= m

    if "source" in filters:
        s = filters["source"]
        if isinstance(s, list):
            m = _terms_overlap(sample.get("source"), list(s))
            per["source"] = {"expected": s, "actual": sample.get("source"), "matched": m}
        else:
            m = sample.get("source") == s
            per["source"] = {"expected": s, "actual": sample.get("source"), "matched": m}
        ok &= m

    if "periodicity" in filters:
        m = sample.get("periodicity") == filters["periodicity"]
        per["periodicity"] = {"expected": filters["periodicity"], "actual": sample.get("periodicity"), "matched": m}
        ok &= m

    if "document_type" in filters:
        m = sample.get("document_type") == filters["document_type"]
        per["document_type"] = {
            "expected": filters["document_type"],
            "actual": sample.get("document_type"),
            "matched": m,
        }
        ok &= m

    if "authors" in filters:
        fa = list(filters["authors"])
        m = _terms_overlap(sample.get("authors"), fa)
        per["authors"] = {"expected": fa, "actual": sample.get("authors"), "matched": m}
        ok &= m

    ys = filters.get("year_start")
    ye = filters.get("year_end")
    if ys is not None or ye is not None:
        doc_y = sample.get("year_start")
        if doc_y is None:
            m = False
        else:
            try:
                y = int(doc_y)
                m = True
                if ys is not None:
                    m &= y >= int(ys)
                if ye is not None:
                    m &= y <= int(ye)
            except (TypeError, ValueError):
                m = False
        per["year_range"] = {"year_start_filter": ys, "year_end_filter": ye, "doc_year_start": doc_y, "matched": m}
        ok &= m

    return {"all_matched": ok, "per_field": per}
