from . import (
    prompts,
    resources,
    tools,
)
from ._server_definition import mcp
from .search_app import search_app
from .apps.schema_app import schema_app
from .apps.rank_app import rank_app
from .apps.compare_app import compare_app
from .apps.summarize_app import summarize_app
from .apps.growth_app import growth_app

for _app in (search_app, schema_app, rank_app, compare_app, summarize_app, growth_app):
    mcp.add_provider(_app)

__all__ = [
    "mcp",
    "search_app",
    "schema_app",
    "rank_app",
    "compare_app",
    "summarize_app",
    "growth_app",
    "tools",
    "resources",
    "prompts",
]
