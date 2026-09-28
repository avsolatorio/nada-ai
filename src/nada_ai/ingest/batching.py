from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

Item = TypeVar("Item")


def batched(items: Iterable[Item], size: int) -> Iterator[tuple[Item, ...]]:
    """Yield tuples of at most ``size`` items, including the final partial batch."""
    if size < 1:
        raise ValueError("batch size must be at least one")
    iterator = iter(items)
    while batch := tuple(islice(iterator, size)):
        yield batch
