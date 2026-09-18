"""Ingest writer port (OpenSearch, Qdrant, …)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nada_ai.ingest.progress import CancelToken, IngestProgressTracker
    from nada_ai.ingest.quality import QualityReport
    from nada_ai.search.backend.opensearch.embeddings import EmbeddingService


@runtime_checkable
class IngestWriterPort(Protocol):
    """Sync bulk ingest into the configured search store."""

    def ensure_target(self, embedding_dim: int, *, recreate: bool = False) -> None:
        """Create index/collection if missing; optionally drop and recreate."""

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
        """Index all langdocs for ``pairs``; returns ``(success_count, errors_or_none)``.

        ``embedding`` — pre-built :class:`EmbeddingService` to reuse rather than
        loading the model again.  Pass ``None`` (default) and the writer will
        instantiate its own; this is the path taken by the CLI and tests.

        ``quality_report`` — optional :class:`QualityReport` that observes each
        source document as it's built; purely additive, never rejects a document.

        ``progress`` — optional :class:`~nada_ai.ingest.progress.IngestProgressTracker`,
        stepped once per ``(idno, metadata_type)`` pair.

        ``cancel_token`` — optional :class:`~nada_ai.ingest.progress.CancelToken`;
        checked once per pair so a cancelled job stops promptly instead of
        running the remaining pairs to completion.

        ``load_errors`` — optional list collecting one entry per pair whose
        metadata load itself failed (distinct from ``errors_or_none``, which is
        write-time failures against the search backend).

        ``empty_docs`` — optional list collecting one entry per pair that loaded
        without error but produced zero indexable content.
        """
