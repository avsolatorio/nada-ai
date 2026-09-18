"""Embedding behavior without loading large models."""

import os
from unittest.mock import MagicMock, patch

import numpy as np

from nada_ai.search.backend.opensearch.embeddings import EmbeddingService, _cap_native_thread_pools, _load_model
from nada_ai.settings import Settings


def test_describe_query_encoding_literal_overrides_name():
    s = Settings()
    s.query_prompt = "Instruct: x\nQuery: "
    s.query_prompt_name = "web_search_query"
    assert s.describe_query_encoding() == {
        "active": "literal_prompt",
        "prompt": "Instruct: x\nQuery: ",
        "prompt_name_configured": "web_search_query",
    }


def test_describe_query_encoding_prompt_name():
    s = Settings()
    s.query_prompt = None
    s.query_prompt_name = "web_search_query"
    assert s.describe_query_encoding() == {"active": "prompt_name", "prompt_name": "web_search_query"}


def test_describe_query_encoding_symmetric():
    s = Settings()
    s.query_prompt = None
    s.query_prompt_name = None
    assert s.describe_query_encoding() == {"active": "symmetric"}


def test_encode_query_uses_prompt_when_set():
    _load_model.cache_clear()
    settings = Settings()
    settings.query_prompt = None
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
    settings.query_prompt = None
    settings.query_prompt_name = None
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.0, 1.0, 0.0]])
    mock_model.get_sentence_embedding_dimension.return_value = 3

    with patch("nada_ai.search.backend.opensearch.embeddings._load_model", return_value=mock_model):
        svc = EmbeddingService(settings)
        svc.encode_query("hello")

    assert "prompt_name" not in mock_model.encode.call_args.kwargs


def test_encode_query_literal_prompt_overrides_prompt_name():
    _load_model.cache_clear()
    settings = Settings()
    settings.query_prompt = "Instruct: test\nQuery: "
    settings.query_prompt_name = "web_search_query"
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.0, 0.0, 1.0]])
    mock_model.get_sentence_embedding_dimension.return_value = 3

    with patch("nada_ai.search.backend.opensearch.embeddings._load_model", return_value=mock_model):
        svc = EmbeddingService(settings)
        svc.encode_query("hello")

    assert mock_model.encode.call_args.kwargs.get("prompt") == "Instruct: test\nQuery: "
    assert "prompt_name" not in mock_model.encode.call_args.kwargs


def test_encode_query_request_override_prompt_name():
    _load_model.cache_clear()
    settings = Settings()
    settings.query_prompt = "Instruct: server\nQuery: "
    settings.query_prompt_name = None
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
    mock_model.get_sentence_embedding_dimension.return_value = 3

    with patch("nada_ai.search.backend.opensearch.embeddings._load_model", return_value=mock_model):
        svc = EmbeddingService(settings)
        svc.encode_query("hello", query_prompt_name="web_search_query")

    assert mock_model.encode.call_args.kwargs.get("prompt_name") == "web_search_query"
    assert "prompt" not in mock_model.encode.call_args.kwargs


def test_cap_native_thread_pools_sets_all_expected_vars(monkeypatch):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        monkeypatch.delenv(var, raising=False)

    _cap_native_thread_pools(4)

    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_cap_native_thread_pools_respects_operators_own_setting(monkeypatch):
    """An operator's own explicit env var (set independently of
    NADA_EMBEDDING_NUM_THREADS in their deployment) must win — this only
    fills in a default, never overrides."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    _cap_native_thread_pools(4)

    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_encode_query_request_override_literal_prompt():
    _load_model.cache_clear()
    settings = Settings()
    settings.query_prompt = None
    settings.query_prompt_name = "web_search_query"
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.0, 1.0, 0.0]])
    mock_model.get_sentence_embedding_dimension.return_value = 3

    with patch("nada_ai.search.backend.opensearch.embeddings._load_model", return_value=mock_model):
        svc = EmbeddingService(settings)
        svc.encode_query("hello", query_prompt="Custom:\n")

    assert mock_model.encode.call_args.kwargs.get("prompt") == "Custom:\n"
    assert "prompt_name" not in mock_model.encode.call_args.kwargs
