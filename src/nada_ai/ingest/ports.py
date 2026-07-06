"""Ingest writer port (OpenSearch, Qdrant, …)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
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
        buffer_size: int = 1000,
        embedding: "EmbeddingService | None" = None,
    ) -> tuple[int, list[Any] | None]:
        """Index all langdocs for ``pairs``; returns ``(success_count, errors_or_none)``.

        ``embedding`` — pre-built :class:`EmbeddingService` to reuse rather than
        loading the model again.  Pass ``None`` (default) and the writer will
        instantiate its own; this is the path taken by the CLI and tests.
        """
