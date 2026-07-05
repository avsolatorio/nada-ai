"""Shared UI action list builder for analytics apps."""

from __future__ import annotations

from prefab_ui.actions import SetState
from prefab_ui.actions.mcp import CallTool


def make_action(
    tool_fn,
    arguments: dict,
    *,
    result_key: str = "result",
    error_msg: str = "Operation failed.",
) -> list:
    """Build the standard loading/success/error action list for an analytics button."""
    return [
        SetState("loading", True),
        SetState(result_key, None),
        SetState("error", None),
        CallTool(
            tool_fn,
            arguments=arguments,
            on_success=[SetState(result_key, "{{ $result }}"), SetState("loading", False)],
            on_error=[SetState("error", error_msg), SetState("loading", False)],
        ),
    ]
