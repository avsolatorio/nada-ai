"""Optional live ingest: catalog API + OpenSearch (heavy; off by default).

Run (from ``nada-ai/`` with compose up and a writable discovery cache)::

    export NADA_INTEGRATION_OPENSEARCH=1
    export AI4DATA_DISCOVERY_DATA_PATH="$(pwd)/data/nada-discovery"
    uv run pytest tests/integration/test_index_from_catalog_live.py -m integration -v

Requires network to the Data Compass search API used by ``ai4data.discovery.catalog``.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("NADA_INTEGRATION_OPENSEARCH", "").lower() not in ("1", "true", "yes"),
    reason="Set NADA_INTEGRATION_OPENSEARCH=1, OpenSearch reachable via NADA_OPENSEARCH_URL, and network for catalog",
)
def test_index_from_catalog_limit_two(tmp_discovery_data_path):
    """Happy path: fetch ≤2 catalog rows and bulk-index (same stack as CLI/API)."""
    from nada_ai.ingest.service import index_from_catalog_op
    from nada_ai.settings import Settings

    settings = Settings()
    res = index_from_catalog_op(
        settings,
        catalog_type="timeseries",
        ps=100,
        limit=2,
        force=False,
        recreate_index=False,
        show_progress_bar=False,
        buffer_size=100,
    )
    assert "indexed" in res
    assert "errors" in res
    assert res.get("rows", 0) <= 2
    assert isinstance(res["errors"], list)
