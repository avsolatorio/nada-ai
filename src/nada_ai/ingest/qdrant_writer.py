"""Qdrant bulk ingest (sync): collection + payload indexes + dense upserts."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.models import PointStruct

from nada_ai.ingest.pipeline import iter_langdoc_records
from nada_ai.ingest.ports import IngestWriterPort
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD, TEXT_FIELD
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


class QdrantIngestWriter(IngestWriterPort):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def ensure_target(self, embedding_dim: int, *, recreate: bool = False) -> None:
        if self._settings.embedding_backend != "local":
            raise ValueError("Qdrant ingest requires embedding_backend=local (stored dense vectors).")
        coll = self._settings.qdrant_collection
        client = _client(self._settings)
        try:
            if recreate and client.collection_exists(collection_name=coll):
                client.delete_collection(collection_name=coll)
            if not client.collection_exists(collection_name=coll):
                client.create_collection(
                    collection_name=coll,
                    vectors_config=qm.VectorParams(size=embedding_dim, distance=qm.Distance.COSINE),
                )
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
    ) -> tuple[int, list[Any] | None]:
        if self._settings.embedding_backend != "local":
            raise ValueError("Qdrant ingest requires embedding_backend=local.")
        embedding = EmbeddingService(self._settings)
        dim = embedding.embedding_dimension()
        self.ensure_target(dim, recreate=recreate_target)

        coll = self._settings.qdrant_collection
        client = _client(self._settings)
        batch: list[PointStruct] = []
        success = 0
        errors: list[Any] = []
        batch_size = 128
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
                    batch.append(PointStruct(id=str(doc_id), vector=vec, payload=_payload_for_point(source)))
                except Exception as e:
                    errors.append({"id": doc_id, "error": str(e)})
                    continue
                if len(batch) >= batch_size:
                    client.upsert(collection_name=coll, points=batch, wait=True)
                    success += len(batch)
                    batch.clear()
            if batch:
                client.upsert(collection_name=coll, points=batch, wait=True)
                success += len(batch)
        except Exception as e:
            errors.append({"error": str(e)})
        finally:
            client.close()
        return success, errors or None
