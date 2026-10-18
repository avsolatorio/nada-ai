"""Shared search heuristics (backend-agnostic)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_IDNO_LIKE = re.compile(r"^[A-Za-z0-9._-]{3,128}$")


def looks_like_catalog_idno(query: str) -> bool:
    """Heuristic: compact idno-shaped token (e.g. WDI codes) with no spaces."""
    q = query.strip()
    if not q or " " in q:
        return False
    return bool(_IDNO_LIKE.match(q))


#: Typographic quotes people paste along with text count as straight ones.
_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"'})
_QUOTED = re.compile(r'"([^"]*)"')


@dataclass(frozen=True)
class QueryParts:
    """A query split into the ``"…"`` phrases that must appear exactly and the words around them."""

    phrases: tuple[str, ...]
    rest: str


def split_quoted(query: str) -> QueryParts:
    """The quoted phrases of ``query`` and the rest of its text.

    Only balanced pairs count: with an odd number of quotes the last one is ordinary text (and dropped, like any
    punctuation). Empty quotes are ignored.
    """
    text = query.translate(_QUOTE_CHARS)
    phrases = tuple(p for p in (m.strip() for m in _QUOTED.findall(text)) if p)
    rest = " ".join(_QUOTED.sub(" ", text).replace('"', " ").split())
    return QueryParts(phrases=phrases, rest=rest)


def has_quoted_phrase(query: str | None) -> bool:
    """Whether ``query`` asks for at least one exact phrase."""
    return bool(query) and bool(split_quoted(query).phrases)
