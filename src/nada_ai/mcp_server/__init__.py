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
from .apps.correlate_app import correlate_app
from .apps.outliers_app import outliers_app
from .apps.trend_app import trend_app
from .apps.benchmark_app import benchmark_app
from .apps.coverage_app import coverage_app
from .apps.join_app import join_app
from .apps.aggregate_app import aggregate_app

for _app in (
    search_app, schema_app, rank_app, compare_app, summarize_app, growth_app,
    correlate_app, outliers_app, trend_app, benchmark_app, coverage_app, join_app, aggregate_app,
):
    mcp.add_provider(_app)

__all__ = [
    "mcp",
    "search_app",
    "schema_app",
    "rank_app",
    "compare_app",
    "summarize_app",
    "growth_app",
    "correlate_app",
    "outliers_app",
    "trend_app",
    "benchmark_app",
    "coverage_app",
    "join_app",
    "aggregate_app",
    "tools",
    "resources",
    "prompts",
]
