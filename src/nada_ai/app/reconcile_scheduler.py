"""In-process periodic scheduler for NADA's search-index change queue.

``ingest.search_index_sync.reconcile_once`` (used by the standalone
``reconcile_search_index`` CLI command) is a one-shot, bare-Settings function
with no access to the FastAPI app's ``JobRegistry`` — so a CLI-driven
reconciliation run has zero coordination with webhook- or admin-triggered
reindexes for the same idno; nothing stops them from racing each other.

This module runs the same reconciliation logic as an in-process periodic
loop *inside* the FastAPI app instead, submitting each queue item through the
same ``JobRegistry`` and the same ``content_sync_job_key`` that
``app.webhooks`` and ``app.catalog_admin`` use — so a queue-driven reindex and
a webhook/admin-triggered one for the same idno now properly single-flight
against each other instead of running unattended and unaware of each other.

Enable with ``NADA_RECONCILE_SEARCH_INDEX_ENABLED=true``. See
``Settings.reconcile_search_index_*`` for the interval/batch-size knobs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nada_ai.app._ingest import content_sync_job_key, guarded_ingest
from nada_ai.app.state import AppState
from nada_ai.ingest.search_index_sync import (
    SearchIndexQueueItem,
    apply_and_ack_queue_item,
    get_status,
    list_queue,
    lookup_metadata_type,
)
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


async def _submit_one(s: AppState, item: SearchIndexQueueItem) -> None:
    settings = s.settings
    idno = item.object_key

    if item.is_delete:
        key = f"delete:{idno}"
        kind = "search_index_reconcile_delete"

        async def factory() -> dict[str, Any]:
            return await guarded_ingest(s, apply_and_ack_queue_item, settings, item)

    else:
        # Resolve metadata_type up front (a small extra API call) so the job
        # key matches content_sync_job_key exactly — this is what lets the
        # scheduler's job single-flight against a webhook/admin job for the
        # same idno. Passing it through also saves apply_and_ack_queue_item
        # from re-resolving it once the job actually runs.
        try:
            metadata_type = await asyncio.to_thread(lookup_metadata_type, settings, idno)
        except Exception as e:  # noqa: BLE001 - best-effort; fall through to failed ack below
            logger.warning("search-index scheduler: metadata_type lookup failed for idno=%s: %s", idno, e)
            metadata_type = None

        if metadata_type is None:
            # No content_sync_job_key can be built without a type; key on the
            # idno alone so at least repeated polls of the same unmappable
            # item single-flight against themselves, if not against other paths.
            key = f"content:unknown:{idno}"
        else:
            key = content_sync_job_key(metadata_type, idno)
        kind = "search_index_reconcile"

        async def factory() -> dict[str, Any]:
            return await guarded_ingest(
                s, apply_and_ack_queue_item, settings, item, metadata_type=metadata_type
            )

    await s.jobs.submit(kind=kind, key=key, factory=factory, params={"idno": idno, "queue_item_id": item.id})


async def poll_once(s: AppState) -> dict[str, Any]:
    """Poll one page of the pending queue and submit each item as its own job.

    Unlike ``reconcile_once``, this does not wait for the submitted jobs to
    finish or ack — it only submits them (fire-and-forget, tracked via
    ``GET /jobs``). That's safe even if this poll and the next one overlap in
    wall-clock time: resubmitting the same (metadata_type, idno) key while a
    job for it is still running just returns the already-running job
    (JobRegistry single-flight), not a duplicate.
    """
    settings = s.settings
    try:
        status = await asyncio.to_thread(get_status, settings)
        if not status.tracking_enabled:
            logger.warning(
                "search-index reconciliation is enabled here, but NADA reports "
                "tracking_enabled=false (search_provider=%r) — the queue will "
                "stay empty until this deployment is configured as NADA's "
                "search provider. Nothing to do this poll.",
                status.search_provider,
            )
    except Exception as e:  # noqa: BLE001 - status check is informational only
        logger.warning("search-index scheduler: status check failed: %s", e)

    items = await asyncio.to_thread(
        list_queue, settings, status="pending", object_type="survey", limit=settings.reconcile_search_index_batch_limit
    )
    for item in items:
        await _submit_one(s, item)
    return {"polled": len(items)}


async def reconcile_loop(s: AppState) -> None:
    """Poll forever at ``settings.reconcile_search_index_interval_seconds``.

    Run as a background ``asyncio.Task`` from the app lifespan; cancel it on
    shutdown (see ``app.main.lifespan``). Errors from one poll are logged and
    swallowed so a transient NADA API hiccup doesn't kill the loop — the next
    poll just tries again.
    """
    settings: Settings = s.settings
    interval = settings.reconcile_search_index_interval_seconds
    logger.info("search-index reconciliation scheduler started (interval=%ss)", interval)
    try:
        while True:
            try:
                await poll_once(s)
            except Exception as e:  # noqa: BLE001 - keep the loop alive across transient failures
                logger.warning("search-index scheduler poll failed: %s", e)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("search-index reconciliation scheduler stopped")
        raise
