from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

EMBEDDING_FIELD = "embedding"
TEXT_FIELD = "page_content"
METADATA_OBJECT_KEY = "metadata"
FILTER_FACETS_KEY = "filter_facets"

# Filter keys as NADA's metadata-extract API names them (see ``filters`` in a study extract). Everything else is
# handled by the dynamic templates below: ``fq_<facet>`` (user-defined facets) are integer term ids, any other key
# is a keyword.
_KEYWORD_FILTER_KEYS = ("dataset_type", "form_model", "repositoryid", "repositories", "tags")
_INTEGER_FILTER_KEYS = (
    "doctype",
    "published",
    "formid",
    "year_start",
    "year_end",
    "years",
    "countries",
    "regions",
    "data_class_id",
)


def metadata_field(logical: str) -> str:
    """Stored path for a facet / filter field (nested under :data:`METADATA_OBJECT_KEY`)."""
    return f"{METADATA_OBJECT_KEY}.{logical}"


def new_index_generation() -> str:
    """Identifier stamped on an index when it is created; it changes whenever the index is rebuilt."""
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def filter_facets_mapping(prefix: str = "") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Flat filter mapping shared by every index: ``(filter_facets object mapping, dynamic templates)``.

    One field per filter key under ``<prefix>filter_facets`` (no nested documents, plain ``term``/``terms``
    queries). ``prefix`` is the path of the object that holds ``filter_facets``: ``"metadata."`` on chunk
    documents, ``""`` on study documents.
    """
    properties: dict[str, Any] = {key: {"type": "keyword"} for key in _KEYWORD_FILTER_KEYS}
    properties.update({key: {"type": "integer"} for key in _INTEGER_FILTER_KEYS})
    templates = [
        {
            "filter_facets_user_facets": {
                "path_match": f"{prefix}{FILTER_FACETS_KEY}.fq_*",
                "mapping": {"type": "integer", "ignore_malformed": True},
            }
        },
        {
            "filter_facets_other_keys": {
                "path_match": f"{prefix}{FILTER_FACETS_KEY}.*",
                "mapping": {"type": "keyword", "ignore_above": 256},
            }
        },
    ]
    return {"type": "object", "dynamic": True, "properties": properties}, templates


def index_body(embedding_dimension: int) -> dict[str, Any]:
    """OpenSearch index settings + mappings of the **chunk** index (text chunks + embeddings + flat filters).

    Tuned for **OpenSearch 3.6+** (LTS): FAISS HNSW + ``cosinesimil`` (semantic embeddings). With FAISS +
    ``cosinesimil``, OpenSearch may L2-normalize vectors at index time so stored values can differ from ingest.
    See vector search settings for optional ``index.knn.*`` / quantization tuning on 3.6.
    """
    facets_mapping, templates = filter_facets_mapping(f"{METADATA_OBJECT_KEY}.")
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "dynamic_templates": templates,
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
                        "sid": {"type": "integer"},
                        "created": {"type": "long"},
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
                        FILTER_FACETS_KEY: facets_mapping,
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
            },
        },
    }


# Text fields searched lexically on a study, and the keyword fields that give it a stable sort order.
STUDY_TEXT_FIELDS = (
    "title",
    "nation",
    "authoring_entity",
    "keywords",
    "abstract",
    "methodology",
    "var_keywords",
)


def studies_index_body() -> dict[str, Any]:
    """OpenSearch settings + mappings of the **study** index: one document per study, ``_id`` = ``sid``.

    Holds what browsing, filtering, sorting and lexical search need, and nothing NADA can hydrate from its own DB.
    Filters use the same flat mapping as the chunk index, at the document root.
    """
    facets_mapping, templates = filter_facets_mapping()
    properties: dict[str, Any] = {
        "sid": {"type": "integer"},
        "idno": {
            "type": "keyword",
            "normalizer": "nada_sort",
            "fields": {"text": {"type": "text", "analyzer": "nada_text"}},
        },
        **{field: {"type": "text", "analyzer": "nada_text"} for field in STUDY_TEXT_FIELDS},
        "title_sort": {"type": "keyword", "normalizer": "nada_sort"},
        "nation_sort": {"type": "keyword", "normalizer": "nada_sort"},
        "year_start": {"type": "integer"},
        "year_end": {"type": "integer"},
        "created": {"type": "long"},
        "changed": {"type": "long"},
        "total_views": {"type": "integer"},
        "total_downloads": {"type": "integer"},
        "varcount": {"type": "integer"},
        FILTER_FACETS_KEY: facets_mapping,
    }
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "nada_text": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"],
                        }
                    },
                    "normalizer": {"nada_sort": {"type": "custom", "filter": ["lowercase", "asciifolding"]}},
                },
            }
        },
        "mappings": {"dynamic": "strict", "dynamic_templates": templates, "properties": properties},
    }
