"""Error envelope and access guard shared by the standard study-search routes (``GET /info``, ``POST /studies/search``).

The contract (``docs/studies-search-contract.md``, section 7) gives these routes one error shape,
``{"error": {"code", "message", "details"}}``, including for authentication and rate limiting, so they cannot use
the framework default that the other routes return.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from nada_ai.app.auth import Principal, require_role
from nada_ai.app.keys_store import Role
from nada_ai.app.rate_limit import client_key
from nada_ai.app.state import AppState, get_state
from nada_ai.app.studies_schemas import ERROR_HTTP_STATUS, ErrorCode, ErrorDetail, ErrorResponse


class StudiesApiError(Exception):
    """An error with a contract code; rendered as the contract error envelope with the code's HTTP status."""

    def __init__(self, code: ErrorCode, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


async def studies_error_handler(_request: Request, exc: StudiesApiError) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details))
    return JSONResponse(status_code=ERROR_HTTP_STATUS[exc.code], content=body.model_dump(mode="json"))


def studies_guard(min_role: Role = Role.read):
    """Dependency for the study-search routes: rate limit, then authenticate, raising contract errors.

    Same rate limiter and role model as ``POST /search`` and the admin routes (key in ``X-NADA-Admin-Key``).
    """
    authenticate = require_role(min_role)

    async def dependency(
        request: Request,
        x_admin_key: str | None = Header(default=None, alias="X-NADA-Admin-Key"),
        s: AppState = Depends(get_state),
    ) -> Principal:
        limiter = getattr(s, "search_rate_limiter", None)
        if limiter is not None and not await limiter.check(client_key(request)):
            raise StudiesApiError(ErrorCode.rate_limited, "rate limit exceeded, slow down")
        try:
            return await authenticate(x_admin_key=x_admin_key, s=s)
        except HTTPException as e:
            # 503: auth is on and cannot be checked (no credentials configured, or the key store is unreadable).
            code = {401: ErrorCode.unauthorized, 503: ErrorCode.backend_unavailable}.get(
                e.status_code, ErrorCode.forbidden
            )
            raise StudiesApiError(code, str(e.detail)) from e

    return dependency
