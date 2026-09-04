import pytest
from pydantic import ValidationError

from nada_ai.settings import Settings


def test_opensearch_ml_requires_model_and_dimension():
    with pytest.raises(ValidationError):
        Settings(embedding_backend="opensearch_ml", opensearch_ml_model_id="x")
    with pytest.raises(ValidationError):
        Settings(embedding_backend="opensearch_ml", opensearch_ml_embedding_dimension=384)


def test_opensearch_ml_ok():
    s = Settings(
        search_backend="opensearch",
        embedding_backend="opensearch_ml",
        opensearch_ml_model_id="mid",
        opensearch_ml_embedding_dimension=384,
    )
    assert s.opensearch_ml_model_id == "mid"
    assert s.opensearch_ml_embedding_dimension == 384


def test_qdrant_forbids_opensearch_ml_embedding():
    with pytest.raises(ValidationError):
        Settings(search_backend="qdrant", embedding_backend="opensearch_ml", opensearch_ml_model_id="m", opensearch_ml_embedding_dimension=384)
