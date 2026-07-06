from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from ai4data.discovery.catalog import get_langdoc_uuid
from ai4data.discovery.metadata.handler import MetadataLoader
from tqdm.auto import tqdm

from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.mapping import index_body
from nada_ai.search.documents import langdoc_to_source
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def ensure_index(client, settings: Settings, embedding_dim: int) -> None:
    name = settings.index_name
    if client.indices.exists(index=name):
        return
    # `body` carries settings + mappings; if opensearch-py deprecates this shape, see UPGRADING.md and split kwargs.
    client.indices.create(index=name, body=index_body(embedding_dim))


def iter_langdoc_records(
    settings: Settings,
    embedding: EmbeddingService | None,
    pairs: Iterable[tuple[str, str]],
    force: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 1000,
) -> Iterator[tuple[str, list[float] | None, dict[str, Any]]]:
    """Yield ``(document_id, embedding_or_none_if_ml_backend, source_payload)`` for each langdoc row."""
    use_ml = settings.embedding_backend == "opensearch_ml"
    buffer: list[tuple[Any, Any | None]] = []

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
            for _, (doc, raw_meta) in ml_iter:
                doc_id = get_langdoc_uuid(doc)
                source = langdoc_to_source(doc, None, raw_metadata=raw_meta)
                yield doc_id, None, source
            return
        if embedding is None:
            raise RuntimeError("embedding service required for local embedding backend")
        texts = [d.page_content for d, _ in items]
        vectors = embedding.encode_corpus(texts, show_progress_bar=show_progress_bar)
        pack_iter = enumerate(items)
        if show_progress_bar:
            pack_iter = tqdm(pack_iter, total=len(items), unit="doc", desc="Pack records", leave=False)
        for i, (doc, raw_meta) in pack_iter:
            vec = vectors[i].tolist()
            doc_id = get_langdoc_uuid(doc)
            source = langdoc_to_source(doc, vec, raw_metadata=raw_meta)
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
        for doc in non_empty:
            buffer.append((doc, raw_meta))
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
) -> Iterator[dict[str, Any]]:
    """pairs: (idno, metadata_type).

    Langdocs are accumulated across records. **Local** backend: ``encode_corpus`` runs when the buffer reaches
    ``buffer_size`` texts. **OpenSearch ML** backend: no local encoding; pipeline embeds ``page_content`` on ingest.
    """
    use_ml = settings.embedding_backend == "opensearch_ml"
    for doc_id, vec, source in iter_langdoc_records(
        settings, embedding, pairs, force=force, show_progress_bar=show_progress_bar, buffer_size=buffer_size
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
) -> tuple[int, list | None]:
    from nada_ai.ingest.factory import create_ingest_writer

    writer = create_ingest_writer(settings)
    return writer.run_bulk(
        pairs,
        force=force,
        recreate_target=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
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
