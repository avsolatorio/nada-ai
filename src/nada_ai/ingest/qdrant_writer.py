"""Qdrant bulk ingest (sync): collection + payload indexes + dense upserts."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.models import PointStruct

from nada_ai.filters.indexes import ensure_qdrant_filter_field_indexes
from nada_ai.ingest.pipeline import iter_langdoc_records
from nada_ai.ingest.ports import IngestWriterPort
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD, TEXT_FIELD
from nada_ai.search.backend.qdrant.sparse_lexical import embed_documents_sparse
from nada_ai.search.canonical import stored_filter_field_name
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

_KEYWORD_INDEX_FIELDS = frozenset(
    {
        "type",
        "idno",
        "idno_uuid",
        "qfield",
        "periodicity",
        "source",
        "document_type",
        "authors",
        "geographies",
    }
)
_INTEGER_INDEX_FIELDS = frozenset({"year_start", "year_end", "years"})


def _client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )


def _payload_for_point(source: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in source.items() if k != EMBEDDING_FIELD}


def _ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    def _try(name: str, schema: Any) -> None:
        try:
            client.create_payload_index(collection_name=collection, field_name=name, field_schema=schema, wait=True)
        except Exception as e:
            logger.debug("Payload index %s: %s", name, e)

    _try(
        TEXT_FIELD,
        qm.TextIndexParams(type="text", tokenizer=qm.TokenizerType.WORD, min_token_len=2, max_token_len=40, lowercase=True),
    )
    for f in sorted(_KEYWORD_INDEX_FIELDS):
        _try(stored_filter_field_name(f), qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD))
    for f in sorted(_INTEGER_INDEX_FIELDS):
        _try(
            stored_filter_field_name(f),
            qm.IntegerIndexParams(type=qm.IntegerIndexType.INTEGER, lookup=True, range=True),
        )
    ensure_qdrant_filter_field_indexes(client, collection)


def _assert_sparse_config_if_needed(client: QdrantClient, collection: str, sparse_name: str) -> None:
    info = client.get_collection(collection_name=collection)
    sparse_cfg = info.config.params.sparse_vectors or {}
    if sparse_name not in sparse_cfg:
        raise ValueError(
            f"Collection {collection!r} has no sparse vector {sparse_name!r}. "
            "Recreate the collection (ingest with recreate_target=True) after enabling NADA_QDRANT_SPARSE_LEXICAL."
        )


class QdrantIngestWriter(IngestWriterPort):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def ensure_target(self, embedding_dim: int, *, recreate: bool = False) -> None:
        if self._settings.embedding_backend != "local":
            raise ValueError("Qdrant ingest requires embedding_backend=local (stored dense vectors).")
        coll = self._settings.qdrant_collection
        client = _client(self._settings)
        sparse_on = self._settings.qdrant_sparse_lexical
        sparse_name = self._settings.qdrant_sparse_vector_name
        try:
            if recreate and client.collection_exists(collection_name=coll):
                client.delete_collection(collection_name=coll)
            if not client.collection_exists(collection_name=coll):
                create_kwargs: dict[str, Any] = {
                    "collection_name": coll,
                    "vectors_config": qm.VectorParams(size=embedding_dim, distance=qm.Distance.COSINE),
                }
                if sparse_on:
                    create_kwargs["sparse_vectors_config"] = {sparse_name: qm.SparseVectorParams()}
                client.create_collection(**create_kwargs)
            elif sparse_on:
                _assert_sparse_config_if_needed(client, coll, sparse_name)
            _ensure_payload_indexes(client, coll)
        finally:
            client.close()

    def run_bulk(
        self,
        pairs: list[tuple[str, str]],
        *,
        force: bool = False,
        recreate_target: bool = False,
        show_progress_bar: bool = True,
        buffer_size: int = 1000,
        embedding: EmbeddingService | None = None,
    ) -> tuple[int, list[Any] | None]:
        if self._settings.embedding_backend != "local":
            raise ValueError("Qdrant ingest requires embedding_backend=local.")
        embedding = embedding or EmbeddingService(self._settings)
        dim = embedding.embedding_dimension()
        self.ensure_target(dim, recreate=recreate_target)

        coll = self._settings.qdrant_collection
        client = _client(self._settings)
        batch_buf: list[tuple[str, list[float], dict[str, Any]]] = []
        success = 0
        errors: list[Any] = []
        batch_size = 128
        sparse_on = self._settings.qdrant_sparse_lexical
        sparse_name = self._settings.qdrant_sparse_vector_name
        model_id = self._settings.qdrant_sparse_model_id

        def flush_buf() -> None:
            nonlocal success
            if not batch_buf:
                return
            try:
                if sparse_on:
                    texts = [str(s.get("page_content") or "") for _, _, s in batch_buf]
                    sparse_vecs = embed_documents_sparse(texts, model_id=model_id)
                    points = [
                        PointStruct(
                            id=str(did),
                            vector={"": v, sparse_name: sv},
                            payload=_payload_for_point(src),
                        )
                        for (did, v, src), sv in zip(batch_buf, sparse_vecs, strict=True)
                    ]
                else:
                    points = [
                        PointStruct(id=str(did), vector=v, payload=_payload_for_point(src)) for did, v, src in batch_buf
                    ]
                client.upsert(collection_name=coll, points=points, wait=True)
                success += len(points)
            except Exception as e:
                errors.append({"error": str(e), "batch_size": len(batch_buf)})
            finally:
                batch_buf.clear()

        try:
            for doc_id, vec, source in iter_langdoc_records(
                self._settings,
                embedding,
                pairs,
                force=force,
                show_progress_bar=show_progress_bar,
                buffer_size=buffer_size,
            ):
                if not vec:
                    errors.append({"id": doc_id, "error": "missing vector (opensearch_ml is not supported on Qdrant)"})
                    continue
                try:
                    batch_buf.append((doc_id, vec, source))
                except Exception as e:
                    errors.append({"id": doc_id, "error": str(e)})
                    continue
                if len(batch_buf) >= batch_size:
                    flush_buf()
            flush_buf()
        except Exception as e:
            errors.append({"error": str(e)})
        finally:
            client.close()
        return success, errors or None
