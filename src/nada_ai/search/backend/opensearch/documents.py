"""Backward-compatible re-export; canonical builder lives in :mod:`nada_ai.search.documents`."""

from __future__ import annotations

from nada_ai.search.documents import langdoc_to_source

__all__ = ["langdoc_to_source"]
