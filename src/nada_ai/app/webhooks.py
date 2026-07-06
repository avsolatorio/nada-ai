"""Webhook endpoints for catalog lifecycle events.

POST /webhooks/catalog receives NADA catalog events and dispatches non-blocking
background jobs to keep the search index in sync automatically.

Auth: same X-NADA-Admin-Key as admin routes (set NADA_ADMIN_API_KEY env var).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nada_ai.app.admin import admin_auth, _submit_or_409
from nada_ai.app.state import AppState, get_state
from nada_ai.filters.service import sync_filter_for_idno_op
from nada_ai.ingest.service import delete_by_idno_op, index_ids_op

logger = logging.getLogger(__name__)

webhooks_router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class CatalogWebhookPayload(BaseModel):
    event: Literal["created", "updated", "deleted"]
    idno: str = Field(..., min_length=1)
    metadata_type: str = Field(default="indicator")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional filter_fields to sync onto the newly indexed points.",
    )


@webhooks_router.post("/catalog", dependencies=[Depends(admin_auth)])
async def webhook_catalog(body: CatalogWebhookPayload, s: AppState = Depends(get_state)) -> JSONResponse:
    """Handle a NADA catalog lifecycle event.

    - ``created`` / ``updated``: deletes any existing points and re-indexes the
      idno. If ``filters`` is supplied, syncs them onto the new points.
    - ``deleted``: removes all indexed documents for the idno.

    All operations run as non-blocking background jobs; poll ``/jobs/{job_id}``
    for completion.
    """
    idno = body.idno.strip()
    event = body.event
    metadata_type = body.metadata_type
    settings = s.settings

    if event == "deleted":
        async def factory() -> dict[str, Any]:
            return await asyncio.to_thread(delete_by_idno_op, settings, idno)

        return await _submit_or_409(
            s,
            kind="delete_by_idno",
            key=f"delete:{idno}",
            factory=factory,
            params={"idno": idno, "event": event},
        )

    # created | updated — delete stale points then re-index
    filters = body.filters

    async def factory() -> dict[str, Any]:
        delete_result = await asyncio.to_thread(delete_by_idno_op, settings, idno)
        index_result = await asyncio.to_thread(index_ids_op, settings, [idno], metadata_type, True)
        result: dict[str, Any] = {"event": event, "delete": delete_result, "index": index_result}
        if filters:
            filter_result = await asyncio.to_thread(sync_filter_for_idno_op, settings, idno, filters)
            result["filters"] = filter_result
        return result

    return await _submit_or_409(
        s,
        kind="webhook_catalog",
        key=f"reindex:{metadata_type}:{idno}",
        factory=factory,
        params={"idno": idno, "event": event, "metadata_type": metadata_type, "has_filters": bool(filters)},
    )
