"""OpenTelemetry parent spans for each MCP tool invocation (nests httpx child spans)."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, cast

from opentelemetry import trace

from .security_validator import validate_tool_call

_tracer = trace.get_tracer("nada.mcp.tools")
_logger = logging.getLogger(__name__)


def instrument_mcp_tool(fn: Callable[..., Any], *, tool_name: str) -> Callable[..., Any]:
    """Wrap a tool function: validate security, then run under an OTel span."""
    if inspect.iscoroutinefunction(fn):
        fn_async = cast("Callable[..., Any]", fn)

        @functools.wraps(fn_async)
        async def _async_impl(*args: Any, **kwargs: Any) -> Any:
            is_valid, error_msg = validate_tool_call(tool_name, kwargs)
            if not is_valid:
                _logger.warning("Security validation failed for %s: %s", tool_name, error_msg)
                raise ValueError(f"Security validation failed: {error_msg}")
            with _tracer.start_as_current_span(
                f"mcp.tool.{tool_name}",
                attributes={"mcp.tool.name": tool_name},
            ):
                return await fn_async(*args, **kwargs)

        return _async_impl

    fn_sync = cast("Callable[..., Any]", fn)

    @functools.wraps(fn_sync)
    def _sync_impl(*args: Any, **kwargs: Any) -> Any:
        is_valid, error_msg = validate_tool_call(tool_name, kwargs)
        if not is_valid:
            _logger.warning("Security validation failed for %s: %s", tool_name, error_msg)
            raise ValueError(f"Security validation failed: {error_msg}")
        with _tracer.start_as_current_span(
            f"mcp.tool.{tool_name}",
            attributes={"mcp.tool.name": tool_name},
        ):
            return fn_sync(*args, **kwargs)

    return _sync_impl
