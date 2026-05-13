"""Shared search heuristics (backend-agnostic)."""

from __future__ import annotations

import re

_IDNO_LIKE = re.compile(r"^[A-Za-z0-9._-]{3,128}$")


def looks_like_catalog_idno(query: str) -> bool:
    """Heuristic: compact idno-shaped token (e.g. WDI codes) with no spaces."""
    q = query.strip()
    if not q or " " in q:
        return False
    return bool(_IDNO_LIKE.match(q))
