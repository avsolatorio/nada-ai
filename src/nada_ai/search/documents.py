"""Canonical stored-document shape for search backends (OpenSearch ``_source``, Qdrant payload).

Root holds ``page_content`` and optional ``embedding``; facet and filter fields live under
``metadata`` so lexical and vector fields stay separate from catalog attributes (aligned with
OpenSearch mapping and Qdrant payload indexes).
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document as LangchainDocument

from nada_ai.ingest.microdata_enrich import append_microdata_discoverability_text


def langdoc_to_source(
    doc: LangchainDocument,
    embedding: list[float] | None,
    raw_metadata: dict | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON document: ``page_content``, ``metadata`` facets, optional ``embedding``.

    Pass ``embedding=None`` when an OpenSearch ingest pipeline (e.g. ``text_embedding``)
    fills ``embedding`` server-side. Qdrant upserts pass the dense vector separately and
    typically omit ``embedding`` from the payload.
    """
    meta = dict(doc.metadata)
    qfield = meta.pop("qfield", None)
    mtype = meta.get("type")
    extra_text = append_microdata_discoverability_text(mtype, meta, raw_metadata)
    page_content = doc.page_content
    if extra_text:
        page_content = f"{page_content}\n\n{extra_text}".strip()

    metadata_body: dict[str, Any] = dict(meta)
    if qfield is not None:
        metadata_body["qfield"] = qfield

    pruned_meta = _prune_none(metadata_body)
    source: dict[str, Any] = {
        "page_content": page_content,
        "metadata": pruned_meta if pruned_meta else {},
    }
    if embedding is not None:
        source["embedding"] = embedding
    return source


def _prune_none(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            nested = _prune_none(v)
            if nested:
                out[k] = nested
        else:
            out[k] = v
    return out
