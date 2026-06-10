"""CLI: ``python -m nada_ai.filters.cli`` (Fire).

Examples::

    uv run python -m nada_ai.filters.cli sync --idno=DOC-123 --filters-file=filters.json
    uv run python -m nada_ai.filters.cli sync-batch --file=records.json
    uv run python -m nada_ai.filters.cli get --idno=DOC-123
    uv run python -m nada_ai.filters.cli ensure-indexes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nada_ai.filters.service import (
    ensure_filter_indexes_op_service,
    get_filters_op,
    sync_filter_for_idno_op,
    sync_filters_op,
)
from nada_ai.filters.sync import parse_filters_input
from nada_ai.settings import Settings


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sync(idno: str, filters: str | None = None, filters_file: str | None = None) -> None:
    """Sync external filters to all points for ``idno``."""
    if not filters and not filters_file:
        raise ValueError("Provide --filters or --filters-file")
    raw = _load_json(filters_file) if filters_file else json.loads(filters or "{}")
    parsed = parse_filters_input(raw)
    settings = Settings()
    res = sync_filter_for_idno_op(settings, idno, parsed)
    print(res)


def sync_batch(file: str) -> None:
    """Batch sync from JSON file: ``[{idno, filters}, ...]``."""
    records = _load_json(file)
    if not isinstance(records, list):
        raise ValueError("Batch file must be a JSON array of {idno, filters} records")
    settings = Settings()
    res = sync_filters_op(settings, records)
    print(json.dumps(res, indent=2))


def get(idno: str) -> None:
    """Read stored ``filter_fields`` for ``idno``."""
    settings = Settings()
    res = get_filters_op(settings, idno)
    print(json.dumps(res, indent=2))


def ensure_indexes() -> None:
    """Ensure Qdrant payload indexes / OpenSearch nested mapping for filter_fields."""
    settings = Settings()
    res = ensure_filter_indexes_op_service(settings)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "sync": sync,
            "sync-batch": sync_batch,
            "sync_batch": sync_batch,
            "get": get,
            "ensure-indexes": ensure_indexes,
            "ensure_indexes": ensure_indexes,
        }
    )
