from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import ai4data.discovery.catalog.extract as catalog_extract
from ai4data.discovery.catalog import get_langdoc_uuid
from ai4data.discovery.metadata.handler import MetadataLoader
from tqdm.auto import tqdm

from nada_ai.filters.metadata_extract import MetadataExtractNotConfigured
from nada_ai.ingest.progress import CancelToken, IngestProgressTracker
from nada_ai.ingest.quality import QualityReport
from nada_ai.ingest.study_sources import StudySource
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.generation import forget_index_generation
from nada_ai.search.backend.opensearch.mapping import (
    EMBEDDING_FIELD,
    STUDY_ALL_TEXT_FIELD,
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


def new_studies_index_body() -> dict[str, Any]:
    """Settings + mappings for a new study index, stamped with a fresh generation."""
    body = studies_index_body()
    body["mappings"]["_meta"] = {"generation": new_index_generation()}
    return body


def ensure_studies_index(client, settings: Settings) -> None:
    """Create the study index if missing (stamped with a generation)."""
    from nada_ai.search.backend.opensearch.index_template import put_index_template_of

    name = settings.studies_index
    if client.indices.exists(index=name):
        return
    put_index_template_of(client, settings, name)
    client.indices.create(index=name, body=new_studies_index_body())
    forget_index_generation()  # NADA must be sent the new generation from the next request on


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


class _RecordLoader(MetadataLoader):
    """ai4data's loader handed the study's record: it neither fetches nor caches it by idno (see
    ``ingest.study_sources``). The loader stamps the key it is given as the record's ``idno``: NADA's own idno."""

    def __init__(self, source: StudySource) -> None:
        self._source = source
        super().__init__(idno=source.idno, metadata_type=source.metadata_type, include_resources=True)

    def load_metadata(self) -> dict:
        return dict(self._source.metadata)


@dataclass(frozen=True)
class StudyExtract:
    """What NADA's metadata-extract API returned for one study (attached by ai4data as ``_extract_*``)."""

    sid: int
    core_fields: dict[str, Any]
    filters: dict[str, Any]
    #: How many chunk documents the pipeline hands to the writer for this study (set once they are known).
    chunks: int = 0


def _study_extract(raw: dict[str, Any] | None, *, require_all_text: bool = False) -> StudyExtract:
    """The study's ``sid`` (NADA internal id, ``surveys.id``), ``core_fields`` and ``filters`` from the extract data.

    Raises :class:`StudyExtractError` when a required value is missing; such a study is not indexed.
    ``require_all_text``: the study document is written too, and needs ``all_text``. NADA sends it with every full
    record, so a record without it was cached before NADA did: it is not indexed incomplete, it is refused.
    """
    raw = raw if isinstance(raw, dict) else {}
    core = raw.get("_extract_core_fields")
    try:
        sid = int(core["catalog_id"])
    except (KeyError, TypeError, ValueError):
        sid = 0
    if sid <= 0:
        raise StudyExtractError("metadata-extract data has no valid core_fields.catalog_id (the NADA internal id)")
    if not str(core.get("idno") or "").strip():
        raise StudyExtractError("metadata-extract data has no core_fields.idno")
    filters = raw.get("_extract_filters")
    if not isinstance(filters, dict):
        raise StudyExtractError("metadata-extract data has no filters")
    if require_all_text and STUDY_ALL_TEXT_FIELD not in core:
        raise StudyExtractError(
            "metadata-extract data has no core_fields.all_text: the record was cached before NADA sent it; delete its "
            "file in the metadata cache (metadata_cache) and index it again"
        )
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
    sources: Iterable[StudySource],
    show_progress_bar: bool = True,
    buffer_size: int = 200,
    quality_report: QualityReport | None = None,
    progress: IngestProgressTracker | None = None,
    cancel_token: CancelToken | None = None,
    load_errors: list[dict[str, Any]] | None = None,
    empty_docs: list[dict[str, Any]] | None = None,
    studies: list[StudyExtract] | None = None,
    stored_vectors: Callable[[list[str]], dict[str, list[float]]] | None = None,
    study_documents: bool = False,
) -> Iterator[tuple[str, list[float] | None, dict[str, Any]]]:
    """Yield ``(document_id, embedding_or_none_if_ml_backend, source_payload)`` for each langdoc row.

    ``quality_report``, if given, observes every ``source`` payload as it's
    built (see ``ingest/quality.py``) — purely additive, never skips or
    rejects a document.

    ``sources``: the studies, each with its record (see ``ingest.study_sources``); nothing is fetched here.

    ``progress``, if given, is stepped once per study (by ``sid``) — this is also what persists the
    resume checkpoint (see ``ingest/progress.py``). A study that fails to load, or has nothing to write, is marked here.
    A row with documents is only announced (``progress.expect``): the writer marks it by confirming each document's
    write (``progress.confirm``), so a row is never checkpointed as done before its documents are in the backend.
    ``study_documents``: the writer also writes one document per study (OpenSearch's study index), so that one is
    expected too -- a study with no chunks then waits for its study document instead of being marked here.

    ``cancel_token``, if given, is checked once per row; when set, the loop
    stops yielding immediately (whatever is already buffered still gets
    flushed by the caller) instead of running the remaining rows to
    completion — see ``CancelToken`` for why this matters more than it looks.

    ``load_errors``, if given, collects one entry (``sid``, ``idno``, ``metadata_type``, ``stage``, ``error``) per
    study whose metadata handler raised, or whose extract data lacks what indexing needs — so a job's ``indexed``
    count lower than ``requested`` always says which study and why.

    ``empty_docs``, if given, collects one entry per row that loaded without
    error but produced zero indexable content (no langdocs at all, or every
    langdoc's ``page_content`` was empty/whitespace) — distinct from
    ``load_errors`` (the loader itself failed) and from ``quality`` (observes
    documents that *were* built). Without this, these rows counted as
    "processed" with no trace anywhere of why they contributed nothing to
    ``indexed``.

    Also bakes in NADA's flat ``filter_facets`` (and, for Qdrant, ``filter_fields``) for each study,
    taken from the metadata-extract data (see ``_study_extract``), so bulk-indexed documents are
    filterable immediately rather than needing a separate filters-sync pass afterward.

    ``studies``, if given, collects one :class:`StudyExtract` per study that loaded, whether or not it
    produced any chunk documents (the OpenSearch writer turns them into study-index documents).

    ``stored_vectors``, if given (local embedding backend only), maps chunk ids to the vectors already stored for
    them. A chunk id is a hash of the chunk's type, idno, field and text, so a stored vector belongs to exactly
    this text: those chunks are not embedded again, only the new or changed ones. The caller decides when reuse is
    valid (same embedding model and input cap, not a forced re-embed). Each vector is stored with the input cap it
    was made with (``embedding_input_cap``).

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
        doc_ids = [get_langdoc_uuid(item[0]) for item in items]
        reused = stored_vectors(doc_ids) if stored_vectors is not None else {}
        to_embed = [i for i, doc_id in enumerate(doc_ids) if doc_id not in reused]
        fresh: dict[int, list[float]] = {}
        if to_embed:
            encoded = embedding.encode_corpus(
                [items[i][0].page_content for i in to_embed], show_progress_bar=show_progress_bar
            )
            fresh = {i: encoded[n].tolist() for n, i in enumerate(to_embed)}
        if reused:
            logger.info("Reused %d stored vector(s); embedded %d chunk(s)", len(items) - len(to_embed), len(to_embed))
        pack_iter = enumerate(items)
        if show_progress_bar:
            pack_iter = tqdm(pack_iter, total=len(items), unit="doc", desc="Pack records", leave=False)
        for i, (doc, raw_meta, filter_fields, filter_facets, study) in pack_iter:
            doc_id = doc_ids[i]
            vec = reused[doc_id] if doc_id in reused else fresh[i]
            source = langdoc_to_source(
                doc,
                vec,
                raw_metadata=raw_meta,
                filter_fields=filter_fields,
                filter_facets=filter_facets,
                sid=study.sid,
                created=study.core_fields.get("created"),
                embedding_input_cap=embedding.input_cap,
            )
            if quality_report is not None:
                quality_report.observe(source)
            yield doc_id, vec, source

    sources_iter: Iterable[StudySource] = sources
    if show_progress_bar:
        total_rows = len(sources) if isinstance(sources, list) else None
        sources_iter = tqdm(sources, total=total_rows, unit="study", desc="Load metadata")

    def skip(source: StudySource, stage: str, error: Exception) -> None:
        logger.warning("Skip %s study %s (%s): %s", source.metadata_type, source.sid, source.idno, error)
        if load_errors is not None:
            load_errors.append(
                {
                    "sid": source.sid,
                    "idno": source.idno,
                    "metadata_type": source.metadata_type,
                    "stage": stage,
                    "error": str(error),
                }
            )
        if progress is not None:
            progress.mark(source.sid, ok=False, error=str(error))

    for source in sources_iter:
        if cancel_token is not None and cancel_token.is_set():
            logger.info("Ingest cancelled before study %s; stopping early", source.sid)
            break
        metadata_type, idno = source.metadata_type, source.idno
        try:
            loader = _RecordLoader(source)
            raw = loader.metadata
            docs = loader.get_metadata_handler().get_langdocs()
        except Exception as e:
            skip(source, "load", e)
            continue
        try:
            study = _study_extract(raw, require_all_text=study_documents)
        except StudyExtractError as e:
            skip(source, "extract", e)
            continue
        non_empty = [d for d in docs if d.page_content and str(d.page_content).strip()]
        study = replace(study, chunks=len(non_empty))
        if studies is not None:
            studies.append(study)
        # Announced before any of its documents can be flushed and confirmed.
        expected = len(non_empty) + (1 if study_documents else 0)
        if progress is not None:
            if expected:
                progress.expect(study.sid, expected)
            else:
                progress.mark(study.sid, ok=True)  # nothing will be written for it
        if not non_empty:
            if empty_docs is not None:
                reason = "empty_page_content" if docs else "no_langdocs"
                empty_docs.append({"sid": study.sid, "idno": idno, "metadata_type": metadata_type, "reason": reason})
            continue
        raw_meta = raw if metadata_type == "microdata" else None
        filter_fields, filter_facets = _filter_payload(settings, study.filters)
        for doc in non_empty:
            buffer.append((doc, raw_meta, filter_fields, filter_facets, study))
            if len(buffer) >= buffer_size:
                yield from flush()

    yield from flush()


def iter_bulk_actions(
    settings: Settings,
    embedding: EmbeddingService | None,
    sources: Iterable[StudySource],
    show_progress_bar: bool = True,
    buffer_size: int = 200,
    quality_report: QualityReport | None = None,
    progress: IngestProgressTracker | None = None,
    cancel_token: CancelToken | None = None,
    load_errors: list[dict[str, Any]] | None = None,
    empty_docs: list[dict[str, Any]] | None = None,
    studies: list[StudyExtract] | None = None,
    stored_vectors: Callable[[list[str]], dict[str, list[float]]] | None = None,
    study_documents: bool = False,
) -> Iterator[dict[str, Any]]:
    """OpenSearch bulk actions of the chunk documents of ``sources`` (see :func:`iter_langdoc_records`).

    Langdocs are accumulated across records. **Local** backend: ``encode_corpus`` runs when the buffer reaches
    ``buffer_size`` texts. **OpenSearch ML** backend: no local encoding; pipeline embeds ``page_content`` on ingest.
    """
    use_ml = settings.embedding_backend == "opensearch_ml"
    for doc_id, vec, source in iter_langdoc_records(
        settings,
        embedding,
        sources,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
        quality_report=quality_report,
        progress=progress,
        cancel_token=cancel_token,
        load_errors=load_errors,
        empty_docs=empty_docs,
        studies=studies,
        stored_vectors=stored_vectors,
        study_documents=study_documents,
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
    sources: Iterable[StudySource],
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
    """Index ``sources`` (see ``ingest.study_sources``). ``force`` embeds every chunk again (no stored vector reused)."""
    from nada_ai.ingest.factory import create_ingest_writer

    # Before the writer can recreate or otherwise touch the target index.
    require_extract_mode()
    writer = create_ingest_writer(settings)
    return writer.run_bulk(
        sources,
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
