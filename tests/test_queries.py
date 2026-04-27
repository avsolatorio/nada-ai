from nada_ai.search.backend.opensearch.mapping import EMBEDDING_FIELD
from nada_ai.search.backend.opensearch.queries import build_filters, build_search_query
from nada_ai.settings import Settings


def test_build_filters_type_and_geo():
    f = build_filters({"type": "indicator", "geographies": ["US", "UK"]})
    assert {"term": {"type": "indicator"}} in f
    assert {"terms": {"geographies": ["US", "UK"]}} in f


def test_build_search_keyword():
    s = Settings()
    body = build_search_query(
        s,
        query_text="gdp",
        mode="keyword",
        query_vector=None,
        filters={"type": "indicator"},
        size=5,
        from_=0,
    )
    assert body["size"] == 5
    assert body["from"] == 0
    assert "multi_match" in body["query"]["bool"]["must"][0]


def test_build_search_vector_requires_vector():
    s = Settings()
    try:
        build_search_query(
            s,
            query_text="x",
            mode="vector",
            query_vector=None,
            filters=None,
        )
    except ValueError as e:
        assert "local embedding backend" in str(e)
        return
    raise AssertionError("expected ValueError")


def test_build_search_hybrid():
    s = Settings()
    body = build_search_query(
        s,
        query_text="test",
        mode="hybrid",
        query_vector=[0.1] * 8,
        filters=None,
        knn_k=10,
    )
    root = body["query"]["bool"]
    assert "should" in root
    assert len(root["should"]) == 2


def test_vector_mode_puts_filter_inside_knn_not_bool_post_filter():
    """Efficient k-NN: filters restrict the ANN search space, not only global top-k."""
    s = Settings()
    vec = [0.1] * 8
    body = build_search_query(
        s,
        query_text="x",
        mode="vector",
        query_vector=vec,
        filters={"type": "indicator"},
        knn_k=20,
    )
    q = body["query"]
    assert "bool" not in q
    knn_field = q["knn"][EMBEDDING_FIELD]
    assert knn_field["k"] == 20
    assert knn_field["vector"] == vec
    assert knn_field["filter"] == {"bool": {"filter": [{"term": {"type": "indicator"}}]}}


def test_hybrid_mode_applies_filter_to_keyword_and_knn_branches():
    s = Settings()
    vec = [0.1] * 8
    body = build_search_query(
        s,
        query_text="test",
        mode="hybrid",
        query_vector=vec,
        filters={"type": "indicator"},
        knn_k=10,
    )
    should = body["query"]["bool"]["should"]
    assert should[0] == {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": "test",
                        "fields": ["page_content"],
                        "boost": float(s.hybrid_keyword_boost),
                        "type": "best_fields",
                    }
                }
            ],
            "filter": [{"term": {"type": "indicator"}}],
        }
    }
    assert should[1]["knn"][EMBEDDING_FIELD]["filter"] == {
        "bool": {"filter": [{"term": {"type": "indicator"}}]}
    }


def test_build_search_collapse_idno():
    s = Settings()
    body = build_search_query(
        s,
        query_text="gdp",
        mode="keyword",
        query_vector=None,
        filters={"type": "indicator"},
        size=5,
        collapse_field="idno",
    )
    assert body["collapse"] == {"field": "idno"}


def test_build_search_collapse_inner_hits():
    s = Settings()
    body = build_search_query(
        s,
        query_text="gdp",
        mode="keyword",
        query_vector=None,
        filters=None,
        size=3,
        collapse_field="idno",
        collapse_inner_hits={"name": "by_qfield", "size": 5},
    )
    assert body["collapse"] == {
        "field": "idno",
        "inner_hits": {"name": "by_qfield", "size": 5},
    }


def test_build_search_exclude_embedding_from_source():
    s = Settings()
    body = build_search_query(
        s,
        query_text="gdp",
        mode="keyword",
        query_vector=None,
        filters=None,
        include_embedding=False,
    )
    assert body["_source"] == {"excludes": [EMBEDDING_FIELD]}


def test_build_search_exclude_embedding_inner_hits():
    s = Settings()
    body = build_search_query(
        s,
        query_text="gdp",
        mode="keyword",
        query_vector=None,
        filters=None,
        collapse_field="idno",
        collapse_inner_hits={"name": "by_qfield", "size": 5},
        include_embedding=False,
    )
    assert body["_source"] == {"excludes": [EMBEDDING_FIELD]}
    assert body["collapse"]["inner_hits"]["_source"] == {"excludes": [EMBEDDING_FIELD]}


def _ml_settings() -> Settings:
    return Settings(
        embedding_backend="opensearch_ml",
        opensearch_ml_model_id="deployed-model-id",
        opensearch_ml_embedding_dimension=768,
    )


def test_vector_mode_opensearch_ml_uses_neural():
    s = _ml_settings()
    body = build_search_query(
        s,
        query_text="labor survey",
        mode="vector",
        query_vector=None,
        filters={"type": "indicator"},
        knn_k=20,
    )
    n = body["query"]["neural"]
    assert n["field"] == EMBEDDING_FIELD
    assert n["query_text"] == "labor survey"
    assert n["model_id"] == "deployed-model-id"
    assert n["k"] == 20
    assert n["filter"] == {"bool": {"filter": [{"term": {"type": "indicator"}}]}}


def test_hybrid_mode_opensearch_ml_neural_plus_keyword():
    s = _ml_settings()
    body = build_search_query(
        s,
        query_text="test",
        mode="hybrid",
        query_vector=None,
        filters=None,
        knn_k=10,
    )
    should = body["query"]["bool"]["should"]
    assert len(should) == 2
    assert "multi_match" in should[0]
    assert should[1]["neural"]["model_id"] == "deployed-model-id"
