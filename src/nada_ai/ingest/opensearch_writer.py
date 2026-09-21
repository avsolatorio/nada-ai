"""OpenSearch bulk ingest (sync): the chunk index (text chunks + embeddings) and the study index (one per study)."""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy.helpers import bulk

from nada_ai.ingest.pipeline import StudyExtract, ensure_index, ensure_studies_index, iter_bulk_actions
from nada_ai.ingest.ports import IngestWriterPort
from nada_ai.ingest.progress import CancelToken, IngestProgressTracker
from nada_ai.ingest.quality import QualityReport
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.index_template import (
    put_cluster_auto_create_index,
    put_composable_index_template,
)
from nada_ai.search.backend.opensearch.ml.setup import ensure_text_embedding_ingest_pipeline
from nada_ai.search.backend.opensearch.studies import study_bulk_action
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def _close_quiet(client: Any) -> None:
    try:
        client.transport.close()
    except Exception:
        pass


#: Studies pruned per ``delete_by_query`` (each contributes one clause and its list of current chunk ids).
_PRUNE_BATCH = 50


def _recording(actions: Any, ids_by_sid: dict[int, set[str]]) -> Any:
    """Pass chunk actions through, remembering which document ids were written for each study."""
    for action in actions:
        ids_by_sid.setdefault(int(action["_source"]["metadata"]["sid"]), set()).add(action["_id"])
        yield action


class OpenSearchIngestWriter(IngestWriterPort):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _prepare(self, client: Any, embedding_dim: int, *, recreate: bool) -> None:
        """Drop (when recreating) and create both indexes, with their templates, before anything is written."""
        settings = self._settings
        if recreate:
            for name in (settings.index_name, settings.studies_index):
                if client.indices.exists(index=name):
                    client.indices.delete(index=name)
        if settings.embedding_backend == "opensearch_ml":
            ensure_text_embedding_ingest_pipeline(client, settings)
        if settings.opensearch_put_composable_index_template:
            put_composable_index_template(client, settings, embedding_dim)
        if settings.opensearch_cluster_auto_create_index:
            put_cluster_auto_create_index(client, settings.opensearch_cluster_auto_create_index)
        ensure_index(client, settings, embedding_dim)
        ensure_studies_index(client, settings)

    def _prune_stale_chunks(self, client: Any, current_ids: dict[int, set[str]]) -> int:
        """Delete each study's chunks that are not in ``current_ids`` (its chunk ids from this run).

        A study that produced no chunks this run has an empty set, so all of its old chunks go. Chunks that failed to
        write are unaffected: an unchanged chunk keeps its id, so its old copy is never in the deleted set.
        """
        pruned = 0
        studies = sorted(current_ids)
        for start in range(0, len(studies), _PRUNE_BATCH):
            should = [
                {
                    "bool": {
                        "filter": [{"term": {"metadata.sid": sid}}],
                        "must_not": [{"ids": {"values": sorted(current_ids[sid])}}],
                    }
                }
                for sid in studies[start : start + _PRUNE_BATCH]
            ]
            body = {"query": {"bool": {"should": should, "minimum_should_match": 1}}}
            resp = client.delete_by_query(index=self._settings.index_name, body=body, refresh=True)
            pruned += int(resp.get("deleted") or 0)
        if pruned:
            logger.info("Pruned %d stale chunk document(s) of re-indexed studies", pruned)
        return pruned

    def ensure_target(self, embedding_dim: int, *, recreate: bool = False) -> None:
        client = build_client(self._settings)
        try:
            self._prepare(client, embedding_dim, recreate=recreate)
        finally:
            _close_quiet(client)

    def run_bulk(
        self,
        pairs: list[tuple[str, str]],
        *,
        force: bool = False,
        recreate_target: bool = False,
        show_progress_bar: bool = True,
        buffer_size: int = 200,
        embedding: EmbeddingService | None = None,
        quality_report: QualityReport | None = None,
        progress: IngestProgressTracker | None = None,
        cancel_token: CancelToken | None = None,
        load_errors: list[dict[str, Any]] | None = None,
        empty_docs: list[dict[str, Any]] | None = None,
    ) -> tuple[int, list[Any] | None]:
        if self._settings.embedding_backend == "opensearch_ml":
            _embedding: EmbeddingService | None = None
            dim = int(self._settings.opensearch_ml_embedding_dimension or 0)
        else:
            _embedding = embedding or EmbeddingService(self._settings)
            dim = _embedding.embedding_dimension()

        client = build_client(self._settings)
        try:
            self._prepare(client, dim, recreate=recreate_target)

            studies: list[StudyExtract] = []
            ids_by_sid: dict[int, set[str]] = {}
            actions = iter_bulk_actions(
                self._settings,
                _embedding,
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
            )
            success, errors = bulk(client, _recording(actions, ids_by_sid), raise_on_error=False, refresh="wait_for")
            err_list: list[Any] = list(errors) if isinstance(errors, list) else []

            # One study document per loaded study, written after its chunks. ``_id`` is the sid, so a re-index
            # replaces the document instead of adding one.
            study_actions = [
                study_bulk_action(self._settings.studies_index, s.sid, s.core_fields, s.filters) for s in studies
            ]
            _, study_errors = bulk(client, study_actions, raise_on_error=False, refresh="wait_for")
            if isinstance(study_errors, list):
                err_list.extend(study_errors)

            # Chunk ids are content hashes, so a study whose text changed leaves its old chunks behind; remove them.
            self._prune_stale_chunks(client, {s.sid: ids_by_sid.get(s.sid, set()) for s in studies})

            if err_list:
                logger.error("Bulk indexing errors: %s", err_list[:5])
            return int(success), err_list or None
        finally:
            _close_quiet(client)
