"""Paging cursors: where the next (or previous) page of a search starts, past the reach of offset paging.

OpenSearch refuses ``from + size`` beyond ``index.max_result_window``, and deep ``from`` gets slower with depth. A cursor
instead carries the sort values of the row a page starts after (``search_after``), so every page costs the same at any
depth. It is stateless: nothing is stored server side, and a cursor never expires (a bookmarked one keeps working,
though rows may have shifted if the index changed since).

Wire form: base64url (no padding) of a compact JSON array ``[version, direction, sort_values, position, total,
total_is_lower_bound]``. It is not signed: the caller (NADA) passes it through unchanged, and an edited cursor can
only move where a page starts or what its labels say, so :func:`decode` validates it strictly and nothing more.

* ``direction`` — ``"a"`` (the page after the row, for Next) or ``"b"`` (the page before it, for Prev).
* ``position`` — for ``a``, the 0-based offset of the first row of the page it leads to; for ``b``, the offset of
  the row it was taken from (the first row of the page the user was on). Display only: it labels "rows X-Y".
* ``total``/``total_is_lower_bound`` — the match count of the first page of the search, carried so that cursor pages
  do not count again.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

CURSOR_VERSION = 1

#: Longest text sort key written to an index (titles, countries): documents store ``sort_key(text)``, so every
#: string sort value a cursor carries is bounded, whatever the length of the title.
MAX_SORT_KEY_CHARS = 256

#: Longest string sort value accepted. Above ``MAX_SORT_KEY_CHARS`` because the sort normalizer's ASCII folding can
#: lengthen a key (``ß`` -> ``ss``).
MAX_STRING_SORT_VALUE = 2 * MAX_SORT_KEY_CHARS

#: Longest cursor accepted, checked before decoding anything. Fits two maximal multibyte text keys (the variable
#: title sort) plus the rest of the payload, base64-encoded.
MAX_CURSOR_LENGTH = 4096


def sort_key(text: str) -> str:
    """The value to index in a text sort field for ``text``: its first ``MAX_SORT_KEY_CHARS`` characters."""
    return text[:MAX_SORT_KEY_CHARS]


class Direction(StrEnum):
    after = "a"
    before = "b"


class SortValueKind(StrEnum):
    """What one position of a sort's values must hold."""

    score = "score"  # a finite number (``_score``)
    id = "id"  # a positive integer (the unique tiebreaker)
    text = "text"  # a string, or null where the document has no value
    number = "number"  # a finite number (a year, a count, a timestamp), or null where the document has no value


class CursorError(ValueError):
    """A cursor that is not one this service issued for the requested search."""


@dataclass(frozen=True)
class Cursor:
    direction: Direction
    sort_values: tuple[Any, ...]
    position: int
    total: int
    total_is_lower_bound: bool


def encode(cursor: Cursor) -> str:
    payload = [
        CURSOR_VERSION,
        cursor.direction.value,
        list(cursor.sort_values),
        cursor.position,
        cursor.total,
        cursor.total_is_lower_bound,
    ]
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _reject_constant(name: str) -> Any:
    raise CursorError(f"cursor holds {name}")


def _check_value(value: Any, kind: SortValueKind) -> Any:
    if kind is SortValueKind.score:
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise CursorError("cursor sort value is not a score")
        return float(value)
    if kind is SortValueKind.id:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CursorError("cursor sort value is not an id")
        return value
    if value is None:
        return None
    if kind is SortValueKind.number:
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise CursorError("cursor sort value is not a number")
        return value
    if not isinstance(value, str) or len(value) > MAX_STRING_SORT_VALUE:
        raise CursorError("cursor sort value is not a text value")
    return value


def _non_negative_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CursorError(f"cursor {what} is not a non-negative integer")
    return value


def decode(token: str, kinds: tuple[SortValueKind, ...]) -> Cursor:
    """The cursor in ``token``, checked against the sort it must continue (``kinds``, one per sort value).

    Raises :class:`CursorError` for anything else: too long, not base64url JSON, NaN or Infinity, an unknown version
    or direction, or sort values that do not fit the sort (a cursor of another sort, or an edited one).
    """
    if not token or len(token) > MAX_CURSOR_LENGTH:
        raise CursorError("cursor is empty or too long")
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
        if isinstance(e, CursorError):
            raise
        raise CursorError("cursor is not readable") from e

    if not isinstance(payload, list) or len(payload) != 6:
        raise CursorError("cursor has the wrong shape")
    version, direction, sort_values, position, total, lower_bound = payload
    if version != CURSOR_VERSION or isinstance(version, bool):
        raise CursorError("cursor version is not supported")
    if direction not in (Direction.after.value, Direction.before.value):
        raise CursorError("cursor direction is not a or b")
    if not isinstance(sort_values, list) or len(sort_values) != len(kinds):
        raise CursorError("cursor does not fit this sort")
    if not isinstance(lower_bound, bool):
        raise CursorError("cursor total flag is not a boolean")

    return Cursor(
        direction=Direction(direction),
        sort_values=tuple(_check_value(v, k) for v, k in zip(sort_values, kinds, strict=True)),
        position=_non_negative_int(position, "position"),
        total=_non_negative_int(total, "total"),
        total_is_lower_bound=lower_bound,
    )
