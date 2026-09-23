from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import ai4data.discovery.catalog.extract as catalog_extract
from ai4data.discovery.catalog import get_langdoc_uuid
from ai4data.discovery.metadata.handler import MetadataLoader
from tqdm.auto import tqdm

from nada_ai.filters.metadata_extract import MetadataExtractNotConfigured
from nada_ai.ingest.progress import CancelToken, IngestProgressTracker
from nada_ai.ingest.quality import QualityReport
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.mapping import (
    EMBEDDING_FIELD,
    index_body,
    new_index_generation,
    studies_index_body,
)
from nada_ai.search.documents import langdoc_to_source
from nada_ai.search.dynamic_filters import normalize_external_filters, normalized_to_facets_map
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def _assert_dense_dim_matches(client, name: str, embedding_dim: int | None, model_id: str) -> None:
    """Guard against silently writing wrong-dimension vectors into an existing index.

    Mirrors the check in ingest.qdrant_writer._assert_dense_dim_matches and
    the reporting logic in GET /admin/embeddings/drift — same failure mode,
    same fix: without this, a changed NADA_EMBEDDING_MODEL_ID fails per-doc
    deep inside the bulk write instead of failing fast here.

    ``embedding_dim=None`` (``embedding_backend=none``) skips the check entirely: there is no vector field to
    drift, by design (see ``mapping.index_body``).
    """
    if embedding_dim is None:
        return
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


def ensure_index(client, settings: Settings, embedding_dim: int | None) -> None:
    """Create the chunk index if missing (stamped with a generation and the embedding model), else check it.

    ``embedding_dim=None`` (``embedding_backend=none``) stamps no ``embedding_model``/``embedding_dim`` at all,
    rather than a model id that was never actually loaded or used — ``GET /info`` would otherwise report a model
    for a deployment that never ran one.
    """
    name = settings.index_name
    if client.indices.exists(index=name):
        _assert_dense_dim_matches(client, name, embedding_dim, settings.embedding_model_id)
        return
    body = index_body(embedding_dim)
    meta: dict[str, Any] = {"generation": new_index_generation()}
    if embedding_dim is not None:
        meta["embedding_model"] = settings.embedding_model_id
        meta["embedding_dim"] = embedding_dim
    body["mappings"]["_meta"] = meta
    # `body` carries settings + mappings; if opensearch-py deprecates this shape, see UPGRADING.md and split kwargs.
    client.indices.create(index=name, body=body)


def ensure_studies_index(client, settings: Settings) -> None:
    """Create the study index if missing (stamped with a generation)."""
    name = settings.studies_index
    if client.indices.exists(index=name):
        return
    body = studies_index_body()
    body["mappings"]["_meta"] = {"generation": new_index_generation()}
    client.indices.create(index=name, body=body)


class StudyExtractError(ValueError):
    """A study's metadata-extract data lacks something indexing requires."""


def require_extract_mode() -> None:
    """Fail fast (before anything is touched) unless NADA's metadata-extract API is configured.

    The extract API is the only source of each study's internal id (``sid``); the plain catalog API does not
    provide it, so indexing cannot work without it.
    """
    if not catalog_extract.is_extract_mode():
        raise MetadataExtractNotConfigured(
            "Indexing requires NADA's metadata-extract API: set AI4DATA_METADATA_CATALOG_EXTRACT_PATH. "
            "It is the only source of each study's internal id (sid)."
        )


@dataclass(frozen=True)
class StudyExtract:
    """What NADA's metadata-extract API returned for one study (attached by ai4data as ``_extract_*``)."""

    sid: int
    core_fields: dict[str, Any]
    filters: dict[str, Any]


def _study_extract(raw: dict[str, Any] | None) -> StudyExtract:
    """The study's ``sid`` (NADA internal id, ``surveys.id``), ``core_fields`` and ``filters`` from the extract data.

    Raises :class:`StudyExtractError` when a required value is missing; such a study is not indexed.
    """
    raw = raw if isinstance(raw, dict) else {}
    core = raw.get("_extract_core_fields")
    try:
        sid = int(core["survey_uid"])
    except (KeyError, TypeError, ValueError):
        sid = 0
    if sid <= 0:
        raise StudyExtractError("metadata-extract data has no valid core_fields.survey_uid (the NADA internal id)")
    if not str(core.get("idno") or "").strip():
        raise StudyExtractError("metadata-extract data has no core_fields.idno")
    filters = raw.get("_extract_filters")
    if not isinstance(filters, dict):
        raise StudyExtractError("metadata-extract data has no filters")
    return StudyExtract(sid=sid, core_fields=core, filters=filters)


def _filter_payload(
    settings: Settings, raw_filters: dict[str, Any]
) -> tuple[list[dict[str, Any]] | None, dict[str, list[str]]]:
    """Normalize and auto-register NADA's raw filters for one study.

    Returns ``(filter_fields, filter_facets)``. ``filter_facets`` is the flat map every chunk document stores.
    The nested ``filter_fields`` rows are only stored by the Qdrant backend (its payload / admin response shape).
    """
    # Deferred import: nada_ai.filters.sync -> nada_ai.ingest.qdrant_writer -> nada_ai.ingest.pipeline
    # would otherwise be a circular import at module load time.
    from nada_ai.filters.sync import auto_register_new_facet_keys

    normalized = normalize_external_filters(raw_filters)
    auto_register_new_facet_keys(settings, [entry["key"] for entry in normalized])
    return (normalized if settings.search_backend == "qdrant" else None), normalized_to_facets_map(normalized)


def iter_langdoc_records(
    settings: Settings,
    embedding: EmbeddingService | None,
    pairs: Iterable[tuple[str, str]],
    force: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 200,
    quality_report: QualityReport | None = None,
    progress: IngestProgressTracker | None = None,
    cancel_token: CancelToken | None = None,
    load_errors: list[dict[str, Any]] | None = None,
    empty_docs: list[dict[str, Any]] | None = None,
    studies: list[StudyExtract] | None = None,
) -> Iterator[tuple[str, list[float] | None, dict[str, Any]]]:
    """Yield ``(document_id, embedding_or_none_if_ml_backend, source_payload)`` for each langdoc row.

    ``quality_report``, if given, observes every ``source`` payload as it's
    built (see ``ingest/quality.py``) — purely additive, never skips or
    rejects a document.

    ``progress``, if given, is stepped once per ``(idno, metadata_type)`` row
    (after that row's documents are loaded/skipped, regardless of outcome) —
    this is also what persists the resume checkpoint (see ``ingest/progress.py``).

    ``cancel_token``, if given, is checked once per row; when set, the loop
    stops yielding immediately (whatever is already buffered still gets
    flushed by the caller) instead of running the remaining rows to
    completion — see ``CancelToken`` for why this matters more than it looks.

    ``load_errors``, if given, collects one entry per row whose
    ``MetadataLoader`` call itself raised — previously these were silently
    logged and skipped, so a job's ``indexed`` count could be lower than
    ``requested``/``rows`` with no record anywhere of which idno or why.

    ``empty_docs``, if given, collects one entry per row that loaded without
    error but produced zero indexable content (no langdocs at all, or every
    langdoc's ``page_content`` was empty/whitespace) — distinct from
    ``load_errors`` (the loader itself failed) and from ``quality`` (observes
    documents that *were* built). Without this, these rows counted as
    "processed" with no trace anywhere of why they contributed nothing to
    ``indexed``.

    Also bakes in NADA's flat ``filter_facets`` (and, for Qdrant, ``filter_fields``) for each idno,
    taken from the metadata-extract data (see ``_study_extract``), so bulk-indexed documents are
    filterable immediately rather than needing a separate filters-sync pass afterward.

    ``studies``, if given, collects one :class:`StudyExtract` per study that loaded, whether or not it
    produced any chunk documents (the OpenSearch writer turns them into study-index documents).

    Every document carries ``metadata.sid``, the NADA internal study id. A study whose
    extract data lacks it is not indexed: it is reported in ``load_errors`` (``stage``
    ``"extract"``) like any other study that cannot be loaded.
    """
    use_ml = settings.embedding_backend == "opensearch_ml"
    # Deliberately no embeddings at all (see mapping.index_body): same shape as the opensearch_ml path below
    # (yield vec=None, no client-side encoding) but for a different reason — there is no pipeline to embed it
    # server-side either, the chunk index simply has no embedding field to fill.
    no_embedding = settings.embedding_backend == "none"
    buffer: list[tuple[Any, Any | None, list[dict[str, Any]] | None, dict[str, list[str]], StudyExtract]] = []

    def flush() -> Iterator[tuple[str, list[float] | None, dict[str, Any]]]:
        nonlocal buffer
        if not buffer:
            return
        items = buffer
        buffer = []
        if use_ml or no_embedding:
            ml_iter = enumerate(items)
            if show_progress_bar:
                ml_iter = tqdm(ml_iter, total=len(items), unit="doc", desc="Pack records", leave=False)
            for _, (doc, raw_meta, filter_fields, filter_facets, study) in ml_iter:
                doc_id = get_langdoc_uuid(doc)
                source = langdoc_to_source(
                    doc,
                    None,
                    raw_metadata=raw_meta,
                    filter_fields=filter_fields,
                    filter_facets=filter_facets,
                    sid=study.sid,
                    created=study.core_fields.get("created"),
                )
                if quality_report is not None:
                    quality_report.observe(source)
                yield doc_id, None, source
            return
        if embedding is None:
            raise RuntimeError("embedding service required for local embedding backend")
        texts = [item[0].page_content for item in items]
        vectors = embedding.encode_corpus(texts, show_progress_bar=show_progress_bar)
        pack_iter = enumerate(items)
        if show_progress_bar:
            pack_iter = tqdm(pack_iter, total=len(items), unit="doc", desc="Pack records", leave=False)
        for i, (doc, raw_meta, filter_fields, filter_facets, study) in pack_iter:
            vec = vectors[i].tolist()
            doc_id = get_langdoc_uuid(doc)
            source = langdoc_to_source(
                doc,
                vec,
                raw_metadata=raw_meta,
                filter_fields=filter_fields,
                filter_facets=filter_facets,
                sid=study.sid,
                created=study.core_fields.get("created"),
            )
            if quality_report is not None:
                quality_report.observe(source)
            yield doc_id, vec, source

    pairs_iter: Iterable[tuple[str, str]] = pairs
    if show_progress_bar:
        total_rows = len(pairs) if isinstance(pairs, list) else None
        pairs_iter = tqdm(pairs, total=total_rows, unit="row", desc="Load metadata")

    for idno, metadata_type in pairs_iter:
        if cancel_token is not None and cancel_token.is_set():
            logger.info("Ingest cancelled before idno=%s %s; stopping early", metadata_type, idno)
            break
        try:
            loader = MetadataLoader(idno=idno, metadata_type=metadata_type, force=force, include_resources=True)
            raw = loader.metadata
            docs = loader.get_metadata_handler().get_langdocs()
        except Exception as e:
            logger.warning("Skip %s %s: %s", metadata_type, idno, e)
            if load_errors is not None:
                load_errors.append({"idno": idno, "metadata_type": metadata_type, "stage": "load", "error": str(e)})
            if progress is not None:
                progress.mark(idno, ok=False, error=str(e))
            continue
        try:
            study = _study_extract(raw)
        except StudyExtractError as e:
            logger.warning("Skip %s %s: %s", metadata_type, idno, e)
            if load_errors is not None:
                load_errors.append({"idno": idno, "metadata_type": metadata_type, "stage": "extract", "error": str(e)})
            if progress is not None:
                progress.mark(idno, ok=False, error=str(e))
            continue
        if studies is not None:
            studies.append(study)
        if not docs:
            if empty_docs is not None:
                empty_docs.append({"idno": idno, "metadata_type": metadata_type, "reason": "no_langdocs"})
            if progress is not None:
                progress.mark(idno, ok=True)
            continue
        non_empty = [d for d in docs if d.page_content and str(d.page_content).strip()]
        if not non_empty:
            if empty_docs is not None:
                empty_docs.append({"idno": idno, "metadata_type": metadata_type, "reason": "empty_page_content"})
            if progress is not None:
                progress.mark(idno, ok=True)
            continue
        raw_meta = raw if metadata_type == "microdata" else None
        filter_fields, filter_facets = _filter_payload(settings, study.filters)
        for doc in non_empty:
            buffer.append((doc, raw_meta, filter_fields, filter_facets, study))
            if len(buffer) >= buffer_size:
                yield from flush()
        if progress is not None:
            progress.mark(idno, ok=True)

    yield from flush()


def iter_bulk_actions(
    settings: Settings,
    embedding: EmbeddingService | None,
    pairs: Iterable[tuple[str, str]],
    force: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 200,
    quality_report: QualityReport | None = None,
    progress: IngestProgressTracker | None = None,
    cancel_token: CancelToken | None = None,
    load_errors: list[dict[str, Any]] | None = None,
    empty_docs: list[dict[str, Any]] | None = None,
    studies: list[StudyExtract] | None = None,
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
        progress=progress,
        cancel_token=cancel_token,
        load_errors=load_errors,
        empty_docs=empty_docs,
        studies=studies,
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
    buffer_size: int = 200,
    embedding: EmbeddingService | None = None,
    quality_report: QualityReport | None = None,
    progress: IngestProgressTracker | None = None,
    cancel_token: CancelToken | None = None,
    load_errors: list[dict[str, Any]] | None = None,
    empty_docs: list[dict[str, Any]] | None = None,
) -> tuple[int, list | None]:
    from nada_ai.ingest.factory import create_ingest_writer

    # Before the writer can recreate or otherwise touch the target index.
    require_extract_mode()
    writer = create_ingest_writer(settings)
    return writer.run_bulk(
        pairs,
        force=force,
        recreate_target=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
        quality_report=quality_report,
        embedding=embedding,
        progress=progress,
        cancel_token=cancel_token,
        load_errors=load_errors,
        empty_docs=empty_docs,
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
