from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from ai4data.discovery.catalog import get_langdoc_uuid
from ai4data.discovery.metadata.handler import MetadataLoader
from tqdm.auto import tqdm

from nada_ai.ingest.quality import QualityReport
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD, index_body
from nada_ai.search.documents import langdoc_to_source
from nada_ai.search.dynamic_filters import normalize_external_filters, normalized_to_facets_map
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def _assert_dense_dim_matches(client, name: str, embedding_dim: int, model_id: str) -> None:
    """Guard against silently writing wrong-dimension vectors into an existing index.

    Mirrors the check in ingest.qdrant_writer._assert_dense_dim_matches and
    the reporting logic in GET /admin/embeddings/drift — same failure mode,
    same fix: without this, a changed NADA_EMBEDDING_MODEL_ID fails per-doc
    deep inside the bulk write instead of failing fast here.
    """
    mapping = client.indices.get_mapping(index=name)
    stored_dim: int | None = None
    for body in mapping.values():
        props = (body.get("mappings") or {}).get("properties") or {}
        emb = props.get(EMBEDDING_FIELD) or {}
        if "dimension" in emb:
            stored_dim = emb["dimension"]
            break
    if stored_dim is not None and stored_dim != embedding_dim:
        raise ValueError(
            f"Index {name!r} was created with dense vector dimension {stored_dim}, "
            f"but the configured embedding model {model_id!r} produces {embedding_dim}-dim "
            "vectors. Check GET /admin/embeddings/drift, then either revert the embedding "
            "model or recreate the index (index_from_catalog/index with recreate_index=True "
            "— plan for a full reindex)."
        )


def ensure_index(client, settings: Settings, embedding_dim: int) -> None:
    name = settings.index_name
    if client.indices.exists(index=name):
        _assert_dense_dim_matches(client, name, embedding_dim, settings.embedding_model_id)
        return
    # `body` carries settings + mappings; if opensearch-py deprecates this shape, see UPGRADING.md and split kwargs.
    client.indices.create(index=name, body=index_body(embedding_dim))


def _fetch_filter_payload(
    settings: Settings, idno: str, raw: dict[str, Any]
) -> tuple[list[dict[str, Any]] | None, dict[str, list[str]] | None]:
    """Best-effort fetch + normalize + auto-register NADA's filters for one idno.

    Returns ``(filter_fields, filter_facets)`` — both ``None`` when no
    filters data is available (never raises; a filters hiccup must not break
    content ingest, matching the QualityReport "purely observational" ethos
    used elsewhere in this pipeline). ``filter_facets`` is only populated for
    the Qdrant backend, which is the only one that stores/uses it.
    """
    if not settings.sync_filters_during_ingest:
        return None, None
    # Deferred import: nada_ai.filters.sync -> nada_ai.ingest.qdrant_writer ->
    # nada_ai.ingest.pipeline would otherwise be a circular import at module load time.
    from nada_ai.filters.sync import auto_register_new_facet_keys, fetch_filters_for_idno

    try:
        raw_filters = fetch_filters_for_idno(settings, idno, raw_metadata=raw)
    except Exception as e:  # noqa: BLE001 - best-effort, must never break ingest
        logger.warning("Filters fetch failed for idno=%s, indexing without facets: %s", idno, e)
        return None, None
    if raw_filters is None:
        return None, None
    normalized = normalize_external_filters(raw_filters)
    auto_register_new_facet_keys(settings, [entry["key"] for entry in normalized])
    facets = normalized_to_facets_map(normalized) if settings.search_backend == "qdrant" else None
    return normalized, facets


def iter_langdoc_records(
    settings: Settings,
    embedding: EmbeddingService | None,
    pairs: Iterable[tuple[str, str]],
    force: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 1000,
    quality_report: QualityReport | None = None,
) -> Iterator[tuple[str, list[float] | None, dict[str, Any]]]:
    """Yield ``(document_id, embedding_or_none_if_ml_backend, source_payload)`` for each langdoc row.

    ``quality_report``, if given, observes every ``source`` payload as it's
    built (see ``ingest/quality.py``) — purely additive, never skips or
    rejects a document.

    Also fetches and bakes in NADA's dynamic ``filter_fields``/``filter_facets``
    for each idno (see ``_fetch_filter_payload``), so bulk-indexed documents
    are facetable immediately rather than needing a separate filters-sync
    pass afterward. Disable with ``settings.sync_filters_during_ingest = False``.
    """
    use_ml = settings.embedding_backend == "opensearch_ml"
    buffer: list[tuple[Any, Any | None, list[dict[str, Any]] | None, dict[str, list[str]] | None]] = []

    def flush() -> Iterator[tuple[str, list[float] | None, dict[str, Any]]]:
        nonlocal buffer
        if not buffer:
            return
        items = buffer
        buffer = []
        if use_ml:
            ml_iter = enumerate(items)
            if show_progress_bar:
                ml_iter = tqdm(ml_iter, total=len(items), unit="doc", desc="Pack records", leave=False)
            for _, (doc, raw_meta, filter_fields, filter_facets) in ml_iter:
                doc_id = get_langdoc_uuid(doc)
                source = langdoc_to_source(
                    doc, None, raw_metadata=raw_meta, filter_fields=filter_fields, filter_facets=filter_facets
                )
                if quality_report is not None:
                    quality_report.observe(source)
                yield doc_id, None, source
            return
        if embedding is None:
            raise RuntimeError("embedding service required for local embedding backend")
        texts = [d.page_content for d, _, _, _ in items]
        vectors = embedding.encode_corpus(texts, show_progress_bar=show_progress_bar)
        pack_iter = enumerate(items)
        if show_progress_bar:
            pack_iter = tqdm(pack_iter, total=len(items), unit="doc", desc="Pack records", leave=False)
        for i, (doc, raw_meta, filter_fields, filter_facets) in pack_iter:
            vec = vectors[i].tolist()
            doc_id = get_langdoc_uuid(doc)
            source = langdoc_to_source(
                doc, vec, raw_metadata=raw_meta, filter_fields=filter_fields, filter_facets=filter_facets
            )
            if quality_report is not None:
                quality_report.observe(source)
            yield doc_id, vec, source

    pairs_iter: Iterable[tuple[str, str]] = pairs
    if show_progress_bar:
        total_rows = len(pairs) if isinstance(pairs, list) else None
        pairs_iter = tqdm(pairs, total=total_rows, unit="row", desc="Load metadata")

    for idno, metadata_type in pairs_iter:
        try:
            loader = MetadataLoader(idno=idno, metadata_type=metadata_type, force=force, include_resources=True)
            raw = loader.metadata
            docs = loader.get_metadata_handler().get_langdocs()
        except Exception as e:
            logger.warning("Skip %s %s: %s", metadata_type, idno, e)
            continue
        if not docs:
            continue
        non_empty = [d for d in docs if d.page_content and str(d.page_content).strip()]
        if not non_empty:
            continue
        raw_meta = raw if metadata_type == "microdata" else None
        filter_fields, filter_facets = _fetch_filter_payload(settings, idno, raw)
        for doc in non_empty:
            buffer.append((doc, raw_meta, filter_fields, filter_facets))
            if len(buffer) >= buffer_size:
                yield from flush()

    yield from flush()


def iter_bulk_actions(
    settings: Settings,
    embedding: EmbeddingService | None,
    pairs: Iterable[tuple[str, str]],
    force: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 1000,
    quality_report: QualityReport | None = None,
) -> Iterator[dict[str, Any]]:
    """pairs: (idno, metadata_type).

    Langdocs are accumulated across records. **Local** backend: ``encode_corpus`` runs when the buffer reaches
    ``buffer_size`` texts. **OpenSearch ML** backend: no local encoding; pipeline embeds ``page_content`` on ingest.
    """
    use_ml = settings.embedding_backend == "opensearch_ml"
    for doc_id, vec, source in iter_langdoc_records(
        settings,
        embedding,
        pairs,
        force=force,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
        quality_report=quality_report,
    ):
        if use_ml:
            yield {
                "_op_type": "index",
                "_index": settings.index_name,
                "_id": doc_id,
                "pipeline": settings.opensearch_ml_ingest_pipeline_name,
                "_source": source,
            }
        else:
            yield {
                "_op_type": "index",
                "_index": settings.index_name,
                "_id": doc_id,
                "_source": source,
            }


def run_bulk_index(
    settings: Settings,
    pairs: list[tuple[str, str]],
    force: bool = False,
    recreate_index: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 1000,
    embedding: EmbeddingService | None = None,
    quality_report: QualityReport | None = None,
) -> tuple[int, list | None]:
    from nada_ai.ingest.factory import create_ingest_writer

    writer = create_ingest_writer(settings)
    return writer.run_bulk(
        pairs,
        force=force,
        recreate_target=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
        quality_report=quality_report,
        embedding=embedding,
    )


def index_ids(
    settings: Settings | None = None,
    idnos: list[str] | None = None,
    metadata_type: str = "indicator",
    force: bool = False,
    recreate_index: bool = False,
    show_progress_bar: bool = True,
) -> None:
    settings = settings or Settings()
    if not idnos:
        logger.warning("No idnos provided")
        return
    pairs = [(i, metadata_type) for i in idnos]
    n, err = run_bulk_index(
        settings, pairs, force=force, recreate_index=recreate_index, show_progress_bar=show_progress_bar
    )
    err_part = f"{len(err)} error(s)" if err else "no errors"
    print(f"Indexed {n} documents; {err_part}")
