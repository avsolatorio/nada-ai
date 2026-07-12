"""Principal resolution and role-based access control for admin endpoints.

Two credential sources feed a single :class:`Principal`:

1. The legacy ``NADA_ADMIN_API_KEY`` environment variable — a single
   super-admin secret, kept for backward compatibility with existing
   deployments. Always resolves to role ``admin`` and satisfies every
   :func:`require_role` check.
2. The per-caller key store (:mod:`nada_ai.app.keys_store`) — individually
   issued, revocable, role-scoped API keys created via ``POST /admin/keys``.

If neither an env key nor any stored key exists, the server is in an
unconfigured (local-dev) state: requests are let through anonymously as
role ``admin`` with a one-time warning — the same fail-open behaviour this
module replaces. As soon as either credential source is configured, auth
is strictly enforced for every request.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Depends, Header, HTTPException

from nada_ai.app.keys_store import ROLE_RANK, Role, has_any_active_keys, verify_key
from nada_ai.app.state import AppState, get_state

logger = logging.getLogger(__name__)

ADMIN_API_KEY_ENV = "NADA_ADMIN_API_KEY"

_UNCONFIGURED_WARNED = False


class Principal:
    __slots__ = ("id", "name", "role", "source")

    def __init__(self, id: str, name: str, role: Role, source: str) -> None:
        self.id = id
        self.name = name
        self.role = role
        self.source = source

    def __repr__(self) -> str:
        return f"Principal(id={self.id!r}, name={self.name!r}, role={self.role!r}, source={self.source!r})"


def _warn_unconfigured() -> None:
    global _UNCONFIGURED_WARNED
    if not _UNCONFIGURED_WARNED:
        logger.warning(
            "SECURITY: no admin credentials configured (%s unset and no API "
            "keys issued) — all admin/webhook endpoints are unauthenticated. "
            "Set %s or create a key via POST /admin/keys before exposing the "
            "server on any network.",
            ADMIN_API_KEY_ENV,
            ADMIN_API_KEY_ENV,
        )
        _UNCONFIGURED_WARNED = True


async def resolve_principal(x_admin_key: str | None, s: AppState) -> Principal | None:
    """Resolve a presented key header to a :class:`Principal`, or ``None`` if invalid."""
    legacy = os.getenv(ADMIN_API_KEY_ENV)
    if legacy and hmac.compare_digest(x_admin_key or "", legacy):
        return Principal(id="env", name="legacy env admin key", role=Role.admin, source="env")

    record = await verify_key(x_admin_key, s.settings)
    if record is not None:
        return Principal(id=record.id, name=record.name, role=record.role, source="key")

    return None


def require_role(min_role: Role):
    """FastAPI dependency factory: require a principal whose role >= ``min_role``.

    Returns the resolved :class:`Principal` so route handlers can attribute
    audit-log entries to the caller.
    """

    async def dependency(
        x_admin_key: str | None = Header(default=None, alias="X-NADA-Admin-Key"),
        s: AppState = Depends(get_state),
    ) -> Principal:
        legacy = os.getenv(ADMIN_API_KEY_ENV)
        if not legacy and not await has_any_active_keys(s.settings):
            _warn_unconfigured()
            return Principal(id="anonymous", name="unauthenticated (unconfigured)", role=Role.admin, source="none")

        principal = await resolve_principal(x_admin_key, s)
        if principal is None:
            raise HTTPException(status_code=401, detail="invalid or missing X-NADA-Admin-Key")
        if ROLE_RANK[principal.role] < ROLE_RANK[min_role]:
            raise HTTPException(status_code=403, detail=f"this action requires role >= {min_role.value}")
        return principal

    return dependency


__all__ = ["ADMIN_API_KEY_ENV", "Principal", "Role", "require_role", "resolve_principal"]
