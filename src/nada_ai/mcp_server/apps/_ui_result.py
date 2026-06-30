"""Shared helper: build ToolResult with LLM-readable data + prefab JSON for UI rendering."""
from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent
from prefab_ui.app import PrefabApp

try:
    from fastmcp.tools.base import ToolResult, _prefab_to_json
    _HAS_PREFAB_JSON = True
except ImportError:  # pragma: no cover
    _HAS_PREFAB_JSON = False


def ui_result(
    app: PrefabApp,
    *,
    app_name: str,
    result: Any | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Return ToolResult (UI + LLM data) when result is available, else bare PrefabApp.

    When result is None the function returns PrefabApp unchanged so the framework
    renders an empty form.  When result is present it returns a ToolResult where:
    - content   = JSON the LLM can read to answer follow-up questions about the data
    - structured_content = prefab JSON for the UI renderer (with correct hash routing)
    """
    has_error = hasattr(result, "error") and result.error
    if result is None or has_error or not _HAS_PREFAB_JSON:
        return app

    payload: dict[str, Any] = {}
    if params:
        payload["params"] = {k: v for k, v in params.items() if v not in (None, "", [])}
    payload["result"] = result.model_dump() if hasattr(result, "model_dump") else result

    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, indent=2, default=str))],
        structured_content=_prefab_to_json(app, fastmcp_app_name=app_name),
    )
