from __future__ import annotations

from typing import Any

EMBEDDING_FIELD = "embedding"
TEXT_FIELD = "page_content"
METADATA_OBJECT_KEY = "metadata"


def metadata_field(logical: str) -> str:
    """Stored path for a facet / filter field (nested under :data:`METADATA_OBJECT_KEY`)."""
    return f"{METADATA_OBJECT_KEY}.{logical}"


def index_body(embedding_dimension: int) -> dict[str, Any]:
    """OpenSearch index settings + mappings for k-NN + facet filters.

    Tuned for **OpenSearch 3.6+** (LTS): FAISS HNSW + ``cosinesimil`` (semantic embeddings). With FAISS +
    ``cosinesimil``, OpenSearch may L2-normalize vectors at index time so stored values can differ from ingest.
    See vector search settings for optional ``index.knn.*`` / quantization tuning on 3.6.
    """
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                TEXT_FIELD: {"type": "text"},
                METADATA_OBJECT_KEY: {
                    "type": "object",
                    "dynamic": True,
                    "properties": {
                        "qfield": {"type": "keyword"},
                        "type": {"type": "keyword"},
                        "idno": {"type": "keyword"},
                        "idno_uuid": {"type": "keyword"},
                        "year_start": {"type": "integer"},
                        "year_end": {"type": "integer"},
                        "years": {"type": "integer"},
                        "geographies": {"type": "keyword"},
                        "periodicity": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "document_type": {"type": "keyword"},
                        "date_published": {"type": "date", "ignore_malformed": True},
                        "date_created": {"type": "date", "ignore_malformed": True},
                        "authors": {"type": "keyword"},
                        "doc_meta": {"type": "object", "enabled": True},
                    },
                },
                EMBEDDING_FIELD: {
                    "type": "knn_vector",
                    "dimension": embedding_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "faiss",
                        "parameters": {
                            "m": 16,
                            "ef_construction": 100,
                        },
                    },
                },
            }
        },
    }
