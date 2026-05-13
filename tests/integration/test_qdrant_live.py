"""
Mark with @pytest.mark.integration; run when a real Qdrant is available, e.g.:

  docker compose -f docker-compose.qdrant.yml up -d
  NADA_INTEGRATION_QDRANT=1 NADA_SEARCH_BACKEND=qdrant uv run pytest tests/integration -m integration
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("NADA_INTEGRATION_QDRANT", "").lower() not in ("1", "true", "yes"),
    reason="Set NADA_INTEGRATION_QDRANT=1 and start Qdrant to run",
)
def test_qdrant_collections_reachable():
    from qdrant_client import QdrantClient

    from nada_ai.settings import Settings

    s = Settings(search_backend="qdrant")
    c = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, prefer_grpc=s.qdrant_prefer_grpc)
    try:
        cols = c.get_collections()
        assert hasattr(cols, "collections")
    finally:
        c.close()
