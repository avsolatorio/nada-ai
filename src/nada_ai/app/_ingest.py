"""Shared ingest dispatch helper for all API entry points.

``guarded_ingest`` is the single place where:

1. The app's shared :class:`~nada_ai.search.backend.opensearch.embeddings.EmbeddingService`
   is warmed up (no-op after the first call).
2. The ingest semaphore is acquired, bounding concurrent embedding-compute jobs
   to ``settings.max_concurrent_ingest_jobs``.
3. The sync ingest function is dispatched to a thread pool via
   ``asyncio.to_thread``.

Usage in any ingest factory::

    async def factory() -> dict[str, Any]:
        return await guarded_ingest(s, index_ids_op, settings, [idno], metadata_type, force)

The call signature mirrors a direct ``asyncio.to_thread(fn, *args, **kwargs)``
call, except that ``embedding=s.embedding`` is automatically injected as a
keyword argument so every job shares the already-loaded model.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from nada_ai.app.state import AppState, ensure_embedding_initialized


def content_sync_job_key(metadata_type: str, idno: str) -> str:
    """Canonical job-registry key for "write current NADA content for this idno".

    Every entry point that writes content for one (metadata_type, idno) pair —
    the admin index/reindex routes, the catalog webhook, and the search-index
    queue scheduler — MUST submit under this same key so JobRegistry's
    single-flight dedup actually protects against them racing each other for
    the same idno. Before this existed, "index" and "reindex" used different
    key prefixes for the same underlying operation and could run concurrently
    for the same idno with no coordination at all.
    """
    return f"content:{metadata_type}:{idno}"


async def guarded_ingest(
    s: AppState,
    fn: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Warm the shared embedding model, acquire the ingest semaphore, then
    call ``fn(*args, embedding=s.embedding, **kwargs)`` on a thread.

    Jobs that arrive when all slots are taken wait here (still alive as asyncio
    tasks, visible as ``running`` in the job registry) until a slot opens.
    """
    if s.settings.embedding_backend == "local":
        await ensure_embedding_initialized(s)

    async with s.ingest_semaphore:
        return await asyncio.to_thread(fn, *args, embedding=s.embedding, **kwargs)
