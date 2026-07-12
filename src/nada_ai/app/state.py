"""Shared FastAPI app state and small helpers.

Kept in its own module so multiple routers (``main``, ``admin``) can import
``state`` and dependencies without circular imports.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from nada_ai.settings import Settings

if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from nada_ai.app.jobs import JobRegistry
    from nada_ai.app.metrics import MetricsRegistry
    from nada_ai.app.rate_limit import RateLimiter
    from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
    from nada_ai.search.ports import SearchBackendPort


class AppState:
    settings: Settings
    #: OpenSearch client; ``None`` when ``search_backend=qdrant`` (admin index routes unavailable).
    client: AsyncOpenSearch | None
    search: SearchBackendPort
    embedding: EmbeddingService | None
    embedding_init_lock: asyncio.Lock
    embedding_init_error: str | None
    jobs: JobRegistry
    #: Serialises concurrent load→mutate→save cycles for the facets config file.
    facets_config_lock: asyncio.Lock
    #: Bounds the number of jobs that may run embedding compute simultaneously.
    #: Sized from ``settings.max_concurrent_ingest_jobs`` at startup.
    ingest_semaphore: asyncio.Semaphore
    #: Serialises load→mutate→save cycles for the API keys store (``app/keys_store.py``).
    api_keys_lock: asyncio.Lock
    #: Serialises appends to the admin audit log (``app/audit.py``).
    audit_lock: asyncio.Lock
    #: Per-process rate limiter for public search endpoints. ``limit_per_minute<=0`` disables.
    search_rate_limiter: RateLimiter
    #: Per-process HTTP request counters/latency histogram, exposed at GET /admin/metrics.
    metrics: MetricsRegistry


state = AppState()


def get_state() -> AppState:
    return state


async def ensure_embedding_initialized(s: AppState) -> None:
    """Lazily initialize the local embedding backend once, on demand."""
    if s.settings.embedding_backend != "local":
        return
    if s.embedding is not None:
        return

    from nada_ai.search.backend.opensearch.embeddings import EmbeddingService

    async with s.embedding_init_lock:
        if s.embedding is not None:
            return
        s.embedding_init_error = None
        try:
            s.embedding = await asyncio.to_thread(EmbeddingService, s.settings)
        except Exception as e:
            s.embedding_init_error = str(e)
            raise
