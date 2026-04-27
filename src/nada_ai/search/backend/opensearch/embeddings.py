from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from nada_ai.settings import Settings


@lru_cache(maxsize=8)
def _load_model(model_id: str, model_kwargs_tuple: tuple[tuple[str, str], ...], device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for embedding_backend=local. "
            "Install: uv sync --extra local (or pip install 'nada-ai[local]')."
        ) from e

    kwargs = dict(model_kwargs_tuple)
    if kwargs:
        return SentenceTransformer(model_id, device=device, model_kwargs=kwargs)
    return SentenceTransformer(model_id, device=device)


class EmbeddingService:
    """SentenceTransformer wrapper with optional asymmetric query encoding."""

    def __init__(self, settings: Settings):
        self._settings = settings
        t = tuple(sorted((self._settings.embedding_model_kwargs or {}).items()))
        self._model = _load_model(
            self._settings.embedding_model_id,
            t,
            self._settings.embedding_device,
        )

    @property
    def model(self) -> Any:
        return self._model

    def embedding_dimension(self) -> int:
        # sentence-transformers 5+ prefers get_embedding_dimension; older models use get_sentence_embedding_dimension.
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if callable(get_dim):
            try:
                return int(get_dim())
            except (TypeError, ValueError):
                pass
        return int(self._model.get_sentence_embedding_dimension())

    def encode_corpus(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        if not texts:
            return np.array([])
        return self._model.encode(
            texts,
            batch_size=self._settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

    def encode_query(self, text: str, show_progress_bar: bool = False) -> np.ndarray:
        common = {
            "batch_size": 1,
            "normalize_embeddings": True,
            "show_progress_bar": show_progress_bar,
        }
        if self._settings.query_prompt:
            return self._model.encode([text], prompt=self._settings.query_prompt, **common)[0]
        if self._settings.query_prompt_name:
            return self._model.encode([text], prompt_name=self._settings.query_prompt_name, **common)[0]
        return self._model.encode([text], **common)[0]
