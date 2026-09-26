"""Principal resolution and role-based access control for admin endpoints.

Two credential sources feed a single :class:`Principal`:

1. The legacy ``NADA_ADMIN_API_KEY`` environment variable — a single
   super-admin secret, kept for backward compatibility with existing
   deployments. Always resolves to role ``admin`` and satisfies every
   :func:`require_role` check.
2. The per-caller key store (:mod:`nada_ai.app.keys_store`) — individually
   issued, revocable, role-scoped API keys created via ``POST /admin/keys``.

Auth is always enforced unless ``NADA_ADMIN_AUTH_DISABLED=true`` turns it off
explicitly (local development): then every caller is ``admin``. With auth on
and no credential configured at all, protected routes answer 503 — never
anonymous access. A key store that cannot be read also answers 503 (see
:class:`~nada_ai.app.keys_store.KeyStoreError`), except for the env key, which
never touches the store.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from nada_ai.app.keys_store import ROLE_RANK, KeyStoreError, Role, has_any_active_keys, verify_key
from nada_ai.app.state import AppState, get_state
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

ADMIN_API_KEY_ENV = "NADA_ADMIN_API_KEY"
ADMIN_AUTH_DISABLED_ENV = "NADA_ADMIN_AUTH_DISABLED"

NO_CREDENTIALS_MESSAGE = (
    f"no admin credentials configured: set {ADMIN_API_KEY_ENV}, "
    f"or {ADMIN_AUTH_DISABLED_ENV}=true for local development only"
)


class Principal:
    __slots__ = ("id", "name", "role", "source")

    def __init__(self, id: str, name: str, role: Role, source: str) -> None:
        self.id = id
        self.name = name
        self.role = role
        self.source = source

    def __repr__(self) -> str:
        return f"Principal(id={self.id!r}, name={self.name!r}, role={self.role!r}, source={self.source!r})"


KEY_STORE_UNREADABLE_MESSAGE = "the API key store is unreadable; see the server log"


async def key_store_error_handler(_request: Request, exc: KeyStoreError) -> JSONResponse:
    """503 for a key store that cannot be read outside ``require_role`` (the key routes, ``/search``'s debug check)."""
    logger.error("API key store: %s", exc)
    return JSONResponse(status_code=503, content={"detail": KEY_STORE_UNREADABLE_MESSAGE})


async def check_auth_config(settings: Settings) -> None:
    """Startup check. Raises when auth is disabled while credentials are configured too — someone who adds a key for
    production and leaves the flag set would believe they are protected. Logs an error when auth is disabled, or on
    and nothing is configured. A key store that cannot be read raises :class:`KeyStoreError`."""
    has_stored_keys = await has_any_active_keys(settings)  # always read, so a broken store fails startup
    has_credentials = bool(os.getenv(ADMIN_API_KEY_ENV)) or has_stored_keys
    if settings.admin_auth_disabled:
        if has_credentials:
            raise RuntimeError(
                f"{ADMIN_AUTH_DISABLED_ENV}=true but admin credentials are configured "
                f"({ADMIN_API_KEY_ENV} or an active stored key): remove one of them"
            )
        logger.error(
            "SECURITY: %s=true — admin authentication is off, every caller of a protected route is admin. "
            "For local development only.",
            ADMIN_AUTH_DISABLED_ENV,
        )
    elif not has_credentials:
        logger.error("SECURITY: %s; protected routes answer 503 until then", NO_CREDENTIALS_MESSAGE)


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
        if s.settings.admin_auth_disabled:
            return Principal(id="anonymous", name="unauthenticated (auth disabled)", role=Role.admin, source="none")

        try:
            principal = await resolve_principal(x_admin_key, s)
            if principal is None and not os.getenv(ADMIN_API_KEY_ENV) and not await has_any_active_keys(s.settings):
                raise HTTPException(status_code=503, detail=NO_CREDENTIALS_MESSAGE)
        except KeyStoreError as e:
            logger.error("admin auth: %s", e)
            raise HTTPException(status_code=503, detail=KEY_STORE_UNREADABLE_MESSAGE) from e
        if principal is None:
            raise HTTPException(status_code=401, detail="invalid or missing X-NADA-Admin-Key")
        if ROLE_RANK[principal.role] < ROLE_RANK[min_role]:
            raise HTTPException(status_code=403, detail=f"this action requires role >= {min_role.value}")
        return principal

    return dependency


__all__ = [
    "ADMIN_API_KEY_ENV",
    "ADMIN_AUTH_DISABLED_ENV",
    "Principal",
    "Role",
    "check_auth_config",
    "key_store_error_handler",
    "require_role",
    "resolve_principal",
]
