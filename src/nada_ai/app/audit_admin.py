"""Read-only endpoint for querying the admin audit trail. Auth: role ``admin``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from nada_ai.app.audit import read_audit_log
from nada_ai.app.auth import require_role
from nada_ai.app.keys_store import Role
from nada_ai.app.state import AppState, get_state

audit_router = APIRouter(prefix="/admin/audit", tags=["audit"])


@audit_router.get("", dependencies=[Depends(require_role(Role.admin))], summary="Query the admin audit trail")
async def audit_list(
    limit: int = Query(default=100, ge=1, le=1000),
    action: str | None = Query(default=None, description="Filter by exact action name, e.g. 'index.delete'."),
    s: AppState = Depends(get_state),
) -> dict[str, Any]:
    entries = read_audit_log(s.settings, limit=limit, action=action)
    return {"entries": entries, "count": len(entries)}


__all__ = ["audit_router"]
