from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np

from nada_ai.settings import Settings


def _cap_native_thread_pools(num_threads: int) -> None:
    """Cap every thread pool involved in local embedding inference, not just PyTorch's.

    ``torch.set_num_threads`` only controls PyTorch's own ATen/intra-op thread
    pool. The OpenMP/MKL/OpenBLAS backend underneath it and the HuggingFace
    ``tokenizers`` library (Rust, via Rayon) each keep their own independent
    thread pool that defaults to every visible CPU core regardless of that
    call — confirmed live: with only ``torch.set_num_threads`` capped, a
    4-thread budget still measured 700-1000%+ CPU during a real reindex,
    because these two were still using every core underneath it.

    Must run before the first import of ``sentence_transformers``/``torch`` in
    this process: these are native libraries read at load/first-use time, not
    something a later Python-level call can retroactively constrain.
    ``setdefault`` so an operator's own explicit env var (set in their
    deployment, decoupled from ``NADA_EMBEDDING_NUM_THREADS``) still wins.
    """
    n = str(num_threads)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=8)
def _load_model(
    model_id: str, model_kwargs_tuple: tuple[tuple[str, str], ...], device: str | None, num_threads: int | None
) -> Any:
    if num_threads is not None:
        _cap_native_thread_pools(num_threads)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for embedding_backend=local. "
            "Install: uv sync --extra local (or pip install 'nada-ai[local]')."
        ) from e

    if num_threads is not None:
        # Also set torch's own thread count explicitly: even though the env
        # vars above are the ones that actually stopped the 700%+ CPU usage,
        # torch.set_num_threads still matters for its ATen/interop pool and
        # for a torch that was already imported elsewhere in this process
        # before this function ran (env vars set here would be too late for
        # a thread pool that already initialized).
        import torch

        torch.set_num_threads(num_threads)

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
            self._settings.embedding_num_threads,
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

    def encode_query(
        self,
        text: str,
        show_progress_bar: bool = False,
        *,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
    ) -> np.ndarray:
        """Encode a search query; optional overrides apply instead of server settings when set."""
        common = {
            "batch_size": 1,
            "normalize_embeddings": True,
            "show_progress_bar": show_progress_bar,
        }
        if query_prompt is not None:
            lit, name = query_prompt, None
        elif query_prompt_name is not None:
            lit, name = None, query_prompt_name
        else:
            lit, name = self._settings.query_prompt, self._settings.query_prompt_name
        if lit:
            return self._model.encode([text], prompt=lit, **common)[0]
        if name:
            return self._model.encode([text], prompt_name=name, **common)[0]
        return self._model.encode([text], **common)[0]
