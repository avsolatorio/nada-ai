"""CLI: ``python -m nada_ai.filters.cli`` (Fire).

Examples::

    uv run python -m nada_ai.filters.cli sync --sid=516 --filters-file=filters.json
    uv run python -m nada_ai.filters.cli sync-batch --file=records.json
    uv run python -m nada_ai.filters.cli get --sid=516
    uv run python -m nada_ai.filters.cli ensure-indexes
    uv run python -m nada_ai.filters.cli backfill-facets
    uv run python -m nada_ai.filters.cli sync-from-extract --sid=516
    uv run python -m nada_ai.filters.cli sync-from-extract --all --page-size=100
    uv run python -m nada_ai.filters.cli fetch-from-extract --all --out=records.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nada_ai.filters.metadata_extract import fetch_all_study_records, fetch_study_records
from nada_ai.filters.service import (
    backfill_filter_facets_op,
    ensure_filter_indexes_op_service,
    get_filters_op,
    sync_filter_for_sid_op,
    sync_filters_op,
)
from nada_ai.filters.sync import parse_filters_input
from nada_ai.settings import Settings


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sync(sid: int, filters: str | None = None, filters_file: str | None = None) -> None:
    """Sync external filters to every indexed document of study ``sid``."""
    if not filters and not filters_file:
        raise ValueError("Provide --filters or --filters-file")
    raw = _load_json(filters_file) if filters_file else json.loads(filters or "{}")
    parsed = parse_filters_input(raw)
    settings = Settings()
    res = sync_filter_for_sid_op(settings, int(sid), parsed)
    print(res)


def sync_batch(file: str, show_progress_bar: bool = True) -> None:
    """Batch sync from JSON file: ``[{sid, filters}, ...]``."""
    records = _load_json(file)
    if not isinstance(records, list):
        raise ValueError("Batch file must be a JSON array of {sid, filters} records")
    settings = Settings()
    res = sync_filters_op(settings, records, show_progress_bar=show_progress_bar)
    print(json.dumps(res, indent=2))


def get(sid: int) -> None:
    """Read the stored filters of study ``sid``."""
    settings = Settings()
    res = get_filters_op(settings, int(sid))
    print(json.dumps(res, indent=2))


def ensure_indexes() -> None:
    """Ensure Qdrant ``filter_facets`` payload indexes / the OpenSearch flat ``filter_facets`` mapping."""
    settings = Settings()
    res = ensure_filter_indexes_op_service(settings)
    print(json.dumps(res, indent=2))


def backfill_facets(show_progress_bar: bool = True) -> None:
    """Derive ``metadata.filter_facets`` from existing ``filter_fields`` on all Qdrant points."""
    settings = Settings()
    res = backfill_filter_facets_op(settings, show_progress_bar=show_progress_bar)
    print(json.dumps(res, indent=2))


def fetch_from_extract(
    sid: int | None = None,
    all: bool = False,
    out: str | None = None,
    include_admin_metadata: bool = True,
    include_metadata: bool = False,
    page_size: int = 100,
    limit: int | None = None,
    show_progress_bar: bool = True,
) -> None:
    """Fetch ``{sid, filters}`` records from the NADA metadata-extract API (no index sync)."""
    if bool(sid) == bool(all):
        raise ValueError("Provide exactly one of --sid or --all")
    settings = Settings()
    if sid:
        records = fetch_study_records(settings, int(sid))
    else:
        records = fetch_all_study_records(
            settings,
            include_admin_metadata=include_admin_metadata,
            include_metadata=include_metadata,
            page_size=page_size,
            max_records=limit,
            show_progress_bar=show_progress_bar,
        )
    payload = {"count": len(records), "records": records}
    text = json.dumps(payload, indent=2)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {len(records)} record(s) to {out}")
    else:
        print(text)


def sync_from_extract(
    sid: int | None = None,
    all: bool = False,
    dry_run: bool = False,
    include_admin_metadata: bool = True,
    include_metadata: bool = False,
    page_size: int = 100,
    limit: int | None = None,
    show_progress_bar: bool = True,
) -> None:
    """Fetch filters from the NADA metadata-extract API and sync them to the index by study id."""
    if bool(sid) == bool(all):
        raise ValueError("Provide exactly one of --sid or --all")
    settings = Settings()
    if sid:
        records = fetch_study_records(settings, int(sid))
    else:
        records = fetch_all_study_records(
            settings,
            include_admin_metadata=include_admin_metadata,
            include_metadata=include_metadata,
            page_size=page_size,
            max_records=limit,
            show_progress_bar=show_progress_bar,
        )

    if dry_run:
        print(json.dumps({"dry_run": True, "count": len(records), "records": records}, indent=2))
        return

    res = sync_filters_op(settings, records, show_progress_bar=show_progress_bar)
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
            "backfill-facets": backfill_facets,
            "backfill_facets": backfill_facets,
            "fetch-from-extract": fetch_from_extract,
            "fetch_from_extract": fetch_from_extract,
            "sync-from-extract": sync_from_extract,
            "sync_from_extract": sync_from_extract,
        }
    )
