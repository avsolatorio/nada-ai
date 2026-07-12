"""Request-scoped context shared between the HTTP middleware and logging.

A single ``ContextVar`` carries the current request ID so log records emitted
anywhere during request handling — including nested ``asyncio.to_thread``
calls that propagate context — can be correlated back to one HTTP request.
"""

from __future__ import annotations

import contextvars

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def get_request_id() -> str:
    return _request_id_var.get()


def set_request_id(value: str) -> contextvars.Token:
    return _request_id_var.set(value)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id_var.reset(token)
