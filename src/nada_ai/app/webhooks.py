"""Webhook endpoints for catalog lifecycle events.

POST /webhooks/catalog receives NADA catalog events and dispatches non-blocking
background jobs to keep the search index in sync automatically.

Auth: see ``nada_ai.app.auth`` — role ``write`` required (either the legacy
``NADA_ADMIN_API_KEY`` or a stored key). Issue the calling system a dedicated
key via ``POST /admin/keys`` with ``role=write`` rather than sharing the
admin key with an external webhook sender.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nada_ai.app._ingest import content_sync_job_key, guarded_ingest
from nada_ai.app.admin import _submit_or_409
from nada_ai.app.auth import Principal, require_role
from nada_ai.app.keys_store import Role
from nada_ai.app.state import AppState, get_state
from nada_ai.filters.service import sync_filter_for_sid_op
from nada_ai.ingest.service import delete_by_sids_op, index_sids_op

logger = logging.getLogger(__name__)

webhooks_router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class CatalogWebhookPayload(BaseModel):
    event: Literal["created", "updated", "deleted"]
    sid: int = Field(..., ge=1, description="The study's NADA internal id (surveys.id)")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional filter_fields to sync onto the newly indexed points.",
    )


@webhooks_router.post("/catalog")
async def webhook_catalog(
    body: CatalogWebhookPayload,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.write)),
) -> JSONResponse:
    """Handle a NADA catalog lifecycle event.

    - ``created`` / ``updated``: deletes any existing points and re-indexes the
      study (its type comes from its NADA record). If ``filters`` is supplied, syncs them onto the new points.
    - ``deleted``: removes all indexed documents of the study.

    All operations run as non-blocking background jobs; poll ``/jobs/{job_id}``
    for completion.
    """
    sid = body.sid
    event = body.event
    settings = s.settings

    if event == "deleted":

        async def factory() -> dict[str, Any]:
            return await asyncio.to_thread(delete_by_sids_op, settings, [sid])

        return await _submit_or_409(
            s,
            kind="delete_by_sid",
            key=f"delete:{sid}",
            factory=factory,
            params={"sid": sid, "event": event},
            principal=principal,
        )

    # created | updated — delete stale points then re-index
    filters = body.filters

    async def factory() -> dict[str, Any]:
        delete_result = await asyncio.to_thread(delete_by_sids_op, settings, [sid])
        index_result = await guarded_ingest(s, index_sids_op, settings, [sid], True)
        result: dict[str, Any] = {"event": event, "delete": delete_result, "index": index_result}
        if filters:
            filter_result = await asyncio.to_thread(sync_filter_for_sid_op, settings, sid, filters)
            result["filters"] = filter_result
        return result

    return await _submit_or_409(
        s,
        kind="webhook_catalog",
        key=content_sync_job_key(sid),
        factory=factory,
        params={"sid": sid, "event": event, "has_filters": bool(filters)},
        principal=principal,
    )
