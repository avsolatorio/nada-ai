"""Ingest writer port (OpenSearch, Qdrant, …)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
    ) -> tuple[int, list[Any] | None]:
        """Index all langdocs for ``pairs``; returns ``(success_count, errors_or_none)``."""
