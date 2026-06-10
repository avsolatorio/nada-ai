"""Translate OpenSearch-style filter dicts into Qdrant ``Filter`` conditions."""

from __future__ import annotations

from typing import Any

from qdrant_client.http import models as qm

from nada_ai.search.canonical import stored_filter_field_name
from nada_ai.search.dynamic_filters import dynamic_filters_to_qdrant_conditions, split_filters


def filters_to_qdrant_filter(filters: dict[str, Any] | None) -> qm.Filter | None:
    """Build a conjunctive Qdrant filter from the same keys as ``build_filters`` in OpenSearch queries."""
    must = _filter_must_conditions(filters)
    if not must:
        return None
    return qm.Filter(must=must)


def _filter_must_conditions(filters: dict[str, Any] | None) -> list[qm.Condition]:
    if not filters:
        return []
    fixed, dynamic = split_filters(filters)
    clauses: list[qm.Condition] = []
    filters = fixed
    if t := filters.get("type"):
        clauses.append(qm.FieldCondition(key=stored_filter_field_name("type"), match=qm.MatchValue(value=t)))
    if idno := filters.get("idno"):
        clauses.append(qm.FieldCondition(key=stored_filter_field_name("idno"), match=qm.MatchValue(value=idno)))
    if idnos := filters.get("idnos"):
        clauses.append(
            qm.FieldCondition(key=stored_filter_field_name("idno"), match=qm.MatchAny(any=list(idnos)))
        )
    if g := filters.get("geographies"):
        clauses.append(
            qm.FieldCondition(key=stored_filter_field_name("geographies"), match=qm.MatchAny(any=list(g)))
        )
    if s := filters.get("source"):
        if isinstance(s, list):
            clauses.append(
                qm.FieldCondition(key=stored_filter_field_name("source"), match=qm.MatchAny(any=list(s)))
            )
        else:
            clauses.append(qm.FieldCondition(key=stored_filter_field_name("source"), match=qm.MatchValue(value=s)))
    if p := filters.get("periodicity"):
        clauses.append(
            qm.FieldCondition(key=stored_filter_field_name("periodicity"), match=qm.MatchValue(value=p))
        )
    if dt := filters.get("document_type"):
        clauses.append(
            qm.FieldCondition(key=stored_filter_field_name("document_type"), match=qm.MatchValue(value=dt))
        )
    if authors := filters.get("authors"):
        clauses.append(
            qm.FieldCondition(key=stored_filter_field_name("authors"), match=qm.MatchAny(any=list(authors)))
        )
    if filters.get("year_start") is not None or filters.get("year_end") is not None:
        rng: dict[str, int] = {}
        if filters.get("year_start") is not None:
            rng["gte"] = int(filters["year_start"])
        if filters.get("year_end") is not None:
            rng["lte"] = int(filters["year_end"])
        clauses.append(
            qm.FieldCondition(key=stored_filter_field_name("year_start"), range=qm.Range(**rng))
        )
    clauses.extend(dynamic_filters_to_qdrant_conditions(dynamic))
    return clauses
