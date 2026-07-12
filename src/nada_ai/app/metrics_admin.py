"""Read-only endpoint exposing in-process HTTP metrics. Auth: role ``read``."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from nada_ai.app.auth import require_role
from nada_ai.app.keys_store import Role
from nada_ai.app.state import AppState, get_state

metrics_router = APIRouter(tags=["metrics"])


@metrics_router.get(
    "/admin/metrics",
    dependencies=[Depends(require_role(Role.read))],
    summary="Prometheus-format HTTP request metrics",
    include_in_schema=True,
)
async def metrics(s: AppState = Depends(get_state)) -> PlainTextResponse:
    return PlainTextResponse(s.metrics.render_prometheus(), media_type="text/plain; version=0.0.4")


__all__ = ["metrics_router"]
