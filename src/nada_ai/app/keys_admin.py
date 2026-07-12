"""Admin endpoints for managing per-caller API keys (role-scoped, revocable).

Auth: role ``admin`` required — either the legacy ``NADA_ADMIN_API_KEY`` or a
stored key with ``role=admin``.

The raw key value is returned exactly once, at creation time. Only its
SHA-256 hash and a short display prefix (``key_prefix``) are ever persisted
or returned afterward — losing the raw value means creating a new key and
revoking the old one.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from nada_ai.app.audit import audit_log
from nada_ai.app.auth import Principal, require_role
from nada_ai.app.keys_store import Role, create_key, list_keys, revoke_key
from nada_ai.app.state import AppState, get_state

keys_router = APIRouter(prefix="/admin/keys", tags=["keys"])


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Human-readable label, e.g. 'ingest-cron' or 'jane@org'.")
    role: Role = Field(default=Role.read, description="read | write | admin")


@keys_router.post("", summary="Issue a new API key")
async def keys_create(
    body: CreateKeyRequest,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.admin)),
) -> dict[str, Any]:
    """Issue a new API key. The raw key is returned only in this response —
    it is never stored or shown again."""
    record, raw_key = await create_key(body.name, body.role, s.settings, s.api_keys_lock)
    await audit_log(
        s, principal, action="key.create", target=record.id, detail=f"name={body.name} role={body.role.value}"
    )
    return {**record.to_public_dict(), "key": raw_key}


@keys_router.get("", summary="List API keys (metadata only, never the raw value)")
async def keys_list(
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.admin)),
) -> dict[str, Any]:
    records = await list_keys(s.settings)
    return {"keys": [r.to_public_dict() for r in records]}


@keys_router.delete("/{key_id}", summary="Revoke an API key")
async def keys_revoke(
    key_id: str,
    s: AppState = Depends(get_state),
    principal: Principal = Depends(require_role(Role.admin)),
) -> dict[str, Any]:
    record = await revoke_key(key_id, s.settings, s.api_keys_lock)
    if record is None:
        raise HTTPException(status_code=404, detail=f"key {key_id} not found")
    await audit_log(s, principal, action="key.revoke", target=key_id)
    return record.to_public_dict()


__all__ = ["keys_router"]
