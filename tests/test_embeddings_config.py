"""Embedding behavior without loading large models."""

from unittest.mock import MagicMock, patch

import numpy as np

from nada_ai.search.backend.opensearch.embeddings import EmbeddingService, _load_model
from nada_ai.settings import Settings


def test_encode_query_uses_prompt_when_set():
    _load_model.cache_clear()
    settings = Settings()
    settings.query_prompt_name = "web_search_query"
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
    mock_model.get_sentence_embedding_dimension.return_value = 3

    with patch("nada_ai.search.backend.opensearch.embeddings._load_model", return_value=mock_model):
        svc = EmbeddingService(settings)
        svc.encode_query("hello")

    mock_model.encode.assert_called()
    assert mock_model.encode.call_args.kwargs.get("prompt_name") == "web_search_query"


def test_encode_query_symmetric_when_prompt_none():
    _load_model.cache_clear()
    settings = Settings()
    settings.query_prompt_name = None
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.0, 1.0, 0.0]])
    mock_model.get_sentence_embedding_dimension.return_value = 3

    with patch("nada_ai.search.backend.opensearch.embeddings._load_model", return_value=mock_model):
        svc = EmbeddingService(settings)
        svc.encode_query("hello")

    assert "prompt_name" not in mock_model.encode.call_args.kwargs
