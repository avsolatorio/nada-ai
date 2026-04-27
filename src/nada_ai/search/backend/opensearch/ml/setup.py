from __future__ import annotations

import logging
from typing import Any

from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD, TEXT_FIELD
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def text_embedding_pipeline_body(settings: Settings) -> dict[str, Any]:
    """Body for ``PUT _ingest/pipeline/{name}`` with a ``text_embedding`` processor."""
    mid = settings.opensearch_ml_model_id
    if not mid:
        raise ValueError("opensearch_ml_model_id is required for text_embedding pipeline")
    return {
        "description": "NADA: embed page_content into knn_vector field via ML Commons",
        "processors": [
            {
                "text_embedding": {
                    "model_id": mid,
                    "field_map": {
                        TEXT_FIELD: EMBEDDING_FIELD,
                    },
                }
            }
        ],
    }


def ingest_pipeline_definition(settings: Settings) -> tuple[str, dict[str, Any]]:
    """Return pipeline name and PUT body."""
    return settings.opensearch_ml_ingest_pipeline_name, text_embedding_pipeline_body(settings)


def ensure_text_embedding_ingest_pipeline(client: Any, settings: Settings) -> None:
    """Create or replace the ingest pipeline if not skipped."""
    if settings.opensearch_ml_skip_ingest_pipeline_setup:
        logger.info("Skipping ingest pipeline setup (opensearch_ml_skip_ingest_pipeline_setup=true)")
        return
    name, body = ingest_pipeline_definition(settings)
    client.ingest.put_pipeline(id=name, body=body)
    logger.info("Ingest pipeline ready: %s", name)
