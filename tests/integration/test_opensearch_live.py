"""
Mark with @pytest.mark.integration; run when a real OpenSearch is available, e.g.:

  docker compose -f docker-compose.opensearch.yml up -d
  NADA_OPENSEARCH_URL=http://localhost:9200 uv run pytest tests/integration -m integration
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("NADA_INTEGRATION_OPENSEARCH", "").lower() not in ("1", "true", "yes"),
    reason="Set NADA_INTEGRATION_OPENSEARCH=1 and start OpenSearch to run",
)
def test_cluster_health_opensearch_reachable():
    from nada_ai.search.backend.opensearch.client import build_client
    from nada_ai.settings import Settings

    c = build_client(Settings())
    try:
        h = c.cluster.health()
        assert h.get("status") in ("green", "yellow", "red")
    finally:
        c.transport.close()
