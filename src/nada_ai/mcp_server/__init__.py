from . import (
    prompts,
    resources,
    tools,
)
from ._server_definition import mcp
from .search_app import search_app

mcp.add_provider(search_app)

__all__ = [
    "mcp",
    "search_app",
    "tools",
    "resources",
    "prompts",
]
