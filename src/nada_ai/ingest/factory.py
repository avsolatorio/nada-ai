"""Factory for :class:`nada_ai.ingest.ports.IngestWriterPort` implementations."""

from __future__ import annotations

from nada_ai.ingest.ports import IngestWriterPort
from nada_ai.settings import Settings


def create_ingest_writer(settings: Settings) -> IngestWriterPort:
    if settings.search_backend == "qdrant":
        from nada_ai.ingest.qdrant_writer import QdrantIngestWriter

        return QdrantIngestWriter(settings)
    from nada_ai.ingest.opensearch_writer import OpenSearchIngestWriter

    return OpenSearchIngestWriter(settings)
