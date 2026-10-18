"""The generation of this deployment's study index, as nada-ai sends it to NADA.

Sent with every search-index request (``X-Index-Generation``) so NADA can refuse a nada-ai whose index is not the one
it tracks. Read from the index's mapping (``mappings._meta.generation``, stamped by
:func:`nada_ai.search.backend.opensearch.mapping.new_index_generation` when the index is created) and kept for a
minute; whatever creates or recreates the study index forgets it, so the next request reads the new one.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from nada_ai.search.backend.opensearch.client import QUICK_REQUEST_TIMEOUT_SECONDS, build_client
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

_generation_cache: dict[str, Any] = {"value": None, "at": float("-inf")}
_generation_lock = threading.Lock()
_GENERATION_TTL = 60.0


def index_generation(settings: Settings) -> str | None:
    """The study index's generation (``mappings._meta.generation``), or None when there is no study index."""
    if settings.search_backend != "opensearch":
        return None
    with _generation_lock:
        if time.monotonic() - _generation_cache["at"] < _GENERATION_TTL:
            return _generation_cache["value"]
        client = build_client(settings)
        try:
            # held under the lock: a hanging OpenSearch must not hold every request to NADA for long
            mapping = client.indices.get_mapping(
                index=settings.studies_index, ignore_unavailable=True, request_timeout=QUICK_REQUEST_TIMEOUT_SECONDS
            )
            meta = (next(iter(mapping.values()))["mappings"].get("_meta") or {}) if mapping else {}
            value = meta.get("generation")
        except Exception as e:  # noqa: BLE001 - the header is optional for NADA; OpenSearch being down must not stop NADA calls
            logger.warning("could not read the study index generation: %s", e)
            value = None
        finally:
            client.close()
        _generation_cache.update(value=value, at=time.monotonic())
        return value


def forget_index_generation() -> None:
    """Read the generation again on the next request (the index was just created or recreated)."""
    with _generation_lock:
        _generation_cache.update(value=None, at=float("-inf"))
