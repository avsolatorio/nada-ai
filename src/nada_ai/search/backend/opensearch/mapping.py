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


def _field_type(spec: dict[str, Any]) -> str | None:
    # OpenSearch leaves ``type: object`` out of the mapping it returns
    return spec.get("type", "object" if "properties" in spec else None)


def _param(value: Any) -> Any:
    # OpenSearch returns some booleans as strings (``dynamic: "true"``)
    return str(value).lower() if isinstance(value, bool) else value


def _same_field(live: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Whether ``live`` maps the field as ``expected`` does: same type, and every other parameter the code sets
    (``store``, ``analyzer``, ``normalizer``, ...) set the same. Parameters the code leaves out are not compared."""
    if _field_type(live) != _field_type(expected):
        return False
    return all(
        _param(live.get(key)) == _param(value)
        for key, value in expected.items()
        if key not in ("type", "properties", "fields")
    )


def outdated_fields(live: dict[str, Any], expected: dict[str, Any], prefix: str = "") -> list[str]:
    """The fields of the ``expected`` mapping properties that the ``live`` ones lack or map differently (see
    :func:`_same_field`), multi-fields and object fields included, as dotted paths. An index is created with its
    mapping and never updated in place, so a field added to or changed in the code's mapping after the index was
    created stays missing or different (a strict index rejects every document carrying a missing one; a field that
    is not stored loses what partial updates carry over) until the index is recreated."""
    fields: list[str] = []
    for name, spec in expected.items():
        path = f"{prefix}{name}"
        have = live.get(name)
        if have is None or not _same_field(have, spec):
            fields.append(path)
            continue
        for nested in ("properties", "fields"):
            if nested in spec:
                fields += outdated_fields(have.get(nested) or {}, spec[nested], f"{path}.")
    return fields


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


def index_body(embedding_dimension: int | None) -> dict[str, Any]:
    """OpenSearch index settings + mappings of the **chunk** index (text chunks + embeddings + flat filters).

    Tuned for **OpenSearch 3.6+** (LTS): FAISS HNSW + ``cosinesimil`` (semantic embeddings). With FAISS +
    ``cosinesimil``, OpenSearch may L2-normalize vectors at index time so stored values can differ from ingest.
    See vector search settings for optional ``index.knn.*`` / quantization tuning on 3.6.

    ``embedding_dimension=None`` (``embedding_backend=none``, a deliberately lexical-only deployment) omits the
    ``embedding`` field and ``index.knn`` entirely: there is no vector to store or search, so the field would only
    be dead mapping. ``semantic``/``hybrid`` are already excluded from that deployment's capabilities
    (``info.modes_for``); nothing queries this index by vector when it lacks the field.
    """
    facets_mapping, templates = filter_facets_mapping(f"{METADATA_OBJECT_KEY}.")
    properties: dict[str, Any] = {
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
    }
    if embedding_dimension is not None:
        properties[EMBEDDING_FIELD] = {
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
        }
    return {
        "settings": {
            "index": {
                **({"knn": True} if embedding_dimension is not None else {}),
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "dynamic_templates": templates,
            "properties": properties,
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

#: Every other text value of the study's metadata, one array element per metadata value (NADA builds it, for full
#: updates only). Searched strictly (all words, or an exact phrase), never loosely: a field this large holds most words
#: of most queries somewhere. Array elements sit 100 positions apart (OpenSearch's default ``position_increment_gap``)
#: and an exact phrase allows no slop, so a phrase never spans two metadata values. Kept as a stored field, not in ``_source``: a partial update (published, filters) does not get it
#: from NADA and carries the stored value over into the rewritten document.
STUDY_ALL_TEXT_FIELD = "all_text"


#: The only fields kept in each index's ``_source``: what is read back from a hit. Everything else is indexed for
#: search (and sorting/filtering) but not stored. NADA reads the records themselves from its own database by id
#: (studies by ``sid``, variables by ``uid``, citations by ``citation_id``). Safe only because these indices are never
#: updated in place (``update_by_query`` or partial ``_update`` rebuild a document from its ``_source``, which would drop
#: every field not kept here): their documents are always rewritten whole. The chunk index is updated in place (its
#: filters) and keeps its full ``_source``.
STUDY_SOURCE_FIELDS = ["sid", "idno", "title", "filter_facets"]
VARIABLE_SOURCE_FIELDS = ["uid", "sid", "published"]
CITATION_SOURCE_FIELDS = ["citation_id", "uuid"]


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
        STUDY_ALL_TEXT_FIELD: {"type": "text", "analyzer": "nada_text", "store": True},
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
        "mappings": {
            "dynamic": "strict",
            "_source": {"includes": STUDY_SOURCE_FIELDS},
            "dynamic_templates": templates,
            "properties": properties,
        },
    }


# Citation fields searched lexically (see docs/citations-search-contract.md). Lexical only, like variables: a citation is
# a short bibliographic record, so keyword matching does the job and nothing needs embedding.
CITATION_TEXT_FIELDS = ("title", "subtitle", "authors", "abstract", "keywords", "notes")


def citations_index_body() -> dict[str, Any]:
    """OpenSearch settings + mappings of the **citation** index: one document per citation, ``_id`` = its NADA id.

    Follows NADA's own citation full-text index (title, subtitle, authors, DOI, keywords, abstract, notes) for the
    searched fields. ``doi`` is a keyword (exact, case-folded) rather than text: a DOI is looked up, not tokenized.
    ``ctype``, ``pub_year`` and ``published`` are filters; ``title_sort`` sorts.
    """
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
        "mappings": {
            "dynamic": "strict",
            "_source": {"includes": CITATION_SOURCE_FIELDS},
            "properties": {
                "citation_id": {"type": "integer"},
                "uuid": {"type": "keyword"},
                **{field: {"type": "text", "analyzer": "nada_text"} for field in CITATION_TEXT_FIELDS},
                "title_sort": {"type": "keyword", "normalizer": "nada_sort"},
                "doi": {"type": "keyword", "normalizer": "nada_sort"},
                "ctype": {"type": "keyword"},
                "pub_year": {"type": "integer"},
                "published": {"type": "integer"},
            },
        },
    }


# Variable fields searched lexically (see docs/variables-search-contract.md). Unlike study search, there is no
# semantic/hybrid mode and no golden-query eval yet: weights are a starting point, not a tuned result.
VARIABLE_TEXT_FIELDS = ("name", "label", "question", "categories")


def variables_index_body() -> dict[str, Any]:
    """OpenSearch settings + mappings of the **variable** index: one document per variable, ``_id`` = ``uid``.

    Denormalized with the owning study's fields (no join at query time), following NADA's own, separate variable
    indexer (``OpenSearch_variable_indexer.php``) for the field set. Lexical only: ``name``/``label``/``question``/
    ``categories`` are searched; ``idno``/``title``/``nation`` are stored for display, not matched (NADA's own
    variable search does not match on them either). Sorting uses ``name.sort`` and ``title``, both normalized
    (case- and accent-insensitive): a ``text`` field cannot be sorted on.
    """
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
        "mappings": {
            "dynamic": "strict",
            "_source": {"includes": VARIABLE_SOURCE_FIELDS},
            "properties": {
                "uid": {"type": "integer"},
                "sid": {"type": "integer"},
                "fid": {"type": "keyword"},
                "vid": {"type": "keyword"},
                **{field: {"type": "text", "analyzer": "nada_text"} for field in VARIABLE_TEXT_FIELDS},
                "name": {
                    "type": "text",
                    "analyzer": "nada_text",
                    "fields": {"sort": {"type": "keyword", "normalizer": "nada_sort", "ignore_above": 256}},
                },
                "idno": {"type": "keyword"},
                "title": {"type": "keyword", "normalizer": "nada_sort"},
                "nation": {"type": "keyword"},
                "dataset_type": {"type": "keyword"},
                "published": {"type": "integer"},
                "year_start": {"type": "integer"},
                "year_end": {"type": "integer"},
                "countries": {"type": "integer"},
            },
        },
    }
