from __future__ import annotations

from typing import Any

from langchain_core.documents import Document as LangchainDocument

from nada_ai.ingest.microdata_enrich import append_microdata_discoverability_text


def langdoc_to_source(
    doc: LangchainDocument,
    embedding: list[float] | None,
    raw_metadata: dict | None = None,
) -> dict[str, Any]:
    """Build OpenSearch _source from a LangChain document.

    Pass ``embedding=None`` when an ingest pipeline (e.g. ``text_embedding``) fills ``embedding`` in OpenSearch.
    """
    meta = dict(doc.metadata)
    qfield = meta.pop("qfield", None)
    mtype = meta.get("type")
    extra_text = append_microdata_discoverability_text(mtype, meta, raw_metadata)
    page_content = doc.page_content
    if extra_text:
        page_content = f"{page_content}\n\n{extra_text}".strip()

    source: dict[str, Any] = {
        "page_content": page_content,
        "qfield": qfield,
        **meta,
    }
    if embedding is not None:
        source["embedding"] = embedding
    return _prune_none(source)


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
