"""OpenSearch bulk ingest (sync)."""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy.helpers import bulk

from nada_ai.ingest.pipeline import ensure_index, iter_bulk_actions
from nada_ai.ingest.ports import IngestWriterPort
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.index_template import (
    put_cluster_auto_create_index,
    put_composable_index_template,
)
from nada_ai.search.backend.opensearch.ml.setup import ensure_text_embedding_ingest_pipeline
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def _close_quiet(client: Any) -> None:
    try:
        client.transport.close()
    except Exception:
        pass


class OpenSearchIngestWriter(IngestWriterPort):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def ensure_target(self, embedding_dim: int, *, recreate: bool = False) -> None:
        client = build_client(self._settings)
        try:
            if recreate and client.indices.exists(index=self._settings.index_name):
                client.indices.delete(index=self._settings.index_name)
            if self._settings.embedding_backend == "opensearch_ml":
                ensure_text_embedding_ingest_pipeline(client, self._settings)
            if self._settings.opensearch_put_composable_index_template:
                put_composable_index_template(client, self._settings, embedding_dim)
            if self._settings.opensearch_cluster_auto_create_index:
                put_cluster_auto_create_index(client, self._settings.opensearch_cluster_auto_create_index)
            ensure_index(client, self._settings, embedding_dim)
        finally:
            _close_quiet(client)

    def run_bulk(
        self,
        pairs: list[tuple[str, str]],
        *,
        force: bool = False,
        recreate_target: bool = False,
        show_progress_bar: bool = True,
        buffer_size: int = 1000,
    ) -> tuple[int, list[Any] | None]:
        if self._settings.embedding_backend == "opensearch_ml":
            embedding: EmbeddingService | None = None
            dim = int(self._settings.opensearch_ml_embedding_dimension or 0)
        else:
            embedding = EmbeddingService(self._settings)
            dim = embedding.embedding_dimension()

        client = build_client(self._settings)
        try:
            if recreate_target and client.indices.exists(index=self._settings.index_name):
                client.indices.delete(index=self._settings.index_name)
            if self._settings.embedding_backend == "opensearch_ml":
                ensure_text_embedding_ingest_pipeline(client, self._settings)
            if self._settings.opensearch_put_composable_index_template:
                put_composable_index_template(client, self._settings, dim)
            if self._settings.opensearch_cluster_auto_create_index:
                put_cluster_auto_create_index(client, self._settings.opensearch_cluster_auto_create_index)
            ensure_index(client, self._settings, dim)

            actions = iter_bulk_actions(
                self._settings,
                embedding,
                pairs,
                force=force,
                show_progress_bar=show_progress_bar,
                buffer_size=buffer_size,
            )
            success, errors = bulk(client, actions, raise_on_error=False, refresh="wait_for")
            err_list: list[Any] | None = None
            if isinstance(errors, list) and errors:
                err_list = errors
                logger.error("Bulk indexing errors: %s", errors[:5])
            return int(success), err_list
        finally:
            _close_quiet(client)
