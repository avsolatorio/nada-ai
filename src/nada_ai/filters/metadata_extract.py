"""Fetch study filters from a NADA search-metadata-extract API.

This talks to the metadata-extract endpoint of whichever NADA instance is
configured (``NADA_METADATA_EXTRACT_BASE_URL``, falling back to
``AI4DATA_METADATA_CATALOG_URL`` + ``_EXTRACT_PATH``) — it is not tied to any
single deployment (e.g. IHSN's Data Compass instance is just one such NADA
instance). Credentials always come from ``AI4DATA_METADATA_CATALOG_*`` — see
``nada_ai.nada.admin_auth`` for why there's no separate credential setting
here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import ai4data.discovery.catalog.extract as catalog_extract

from nada_ai.nada.admin_auth import resolve_admin_cookies, resolve_admin_headers, scrub_admin_credentials
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

_USER_AGENT = "nada-ai-filters-cli/1.0"


class MetadataExtractError(RuntimeError):
    """Raised when the metadata-extract API returns an error payload."""


class MetadataExtractNotConfigured(MetadataExtractError):
    """Raised when no metadata-extract base URL is configured for this instance."""


def _base_url(settings: Settings) -> str:
    if settings.metadata_extract_base_url:
        return settings.metadata_extract_base_url.rstrip("/")
    if url := catalog_extract.extract_base_url():
        return url
    raise MetadataExtractNotConfigured(
        "No metadata-extract base URL configured. Set NADA_METADATA_EXTRACT_BASE_URL "
        "(or AI4DATA_METADATA_CATALOG_EXTRACT_PATH) to the metadata-extract API for "
        "your NADA instance."
    )


def _request_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "base_url": _base_url(settings),
        "headers": resolve_admin_headers(user_agent=_USER_AGENT),
        "cookies": resolve_admin_cookies(),
    }


def _wrap_extract_error(exc: Exception) -> MetadataExtractError:
    return MetadataExtractError(scrub_admin_credentials(str(exc)))


def study_idno(study: dict[str, Any]) -> str | None:
    return catalog_extract.study_idno(study)


def study_to_sync_record(study: dict[str, Any]) -> dict[str, Any] | None:
    """Extract ``{idno, filters}`` from one study payload."""
    idno = study_idno(study)
    filters = study.get("filters")
    if not idno or not isinstance(filters, dict):
        return None
    return {"idno": idno, "filters": filters}


def parse_extract_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse list or single-study metadata-extract JSON into sync records."""
    records: list[dict[str, Any]] = []

    studies = data.get("studies")
    if isinstance(studies, list):
        for study in studies:
            if isinstance(study, dict) and (rec := study_to_sync_record(study)):
                records.append(rec)
        return records

    study = data.get("study")
    if isinstance(study, dict) and (rec := study_to_sync_record(study)):
        return [rec]

    if "filters" in data and (rec := study_to_sync_record(data)):
        return [rec]

    return records


def fetch_study_records(
    settings: Settings,
    idno: str,
    *,
    include_admin_metadata: bool = True,
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Fetch filters for a single study idno."""
    try:
        data = catalog_extract.fetch_extract_study(
            idno,
            include_admin_metadata=include_admin_metadata,
            include_metadata=include_metadata,
            **_request_kwargs(settings),
        )
    except Exception as e:
        raise _wrap_extract_error(e) from e

    records = parse_extract_response(data)
    if not records:
        raise MetadataExtractError(f"No filters found in response for idno {idno!r}")
    return records


def iter_study_records(
    settings: Settings,
    *,
    include_admin_metadata: bool = True,
    include_metadata: bool = False,
    page_size: int = 100,
    max_records: int | None = None,
    show_progress_bar: bool = False,
) -> Iterator[dict[str, Any]]:
    """Paginate all studies and yield ``{idno, filters}`` records."""
    request_kwargs = _request_kwargs(settings)
    offset = 0
    seen = 0
    pbar: Any = None
    try:
        while True:
            try:
                data = catalog_extract.fetch_extract_page(
                    {"offset": offset, "limit": page_size},
                    include_admin_metadata=include_admin_metadata,
                    include_metadata=include_metadata,
                    **request_kwargs,
                )
            except Exception as e:
                raise _wrap_extract_error(e) from e

            batch = parse_extract_response(data)
            if not batch:
                break

            if show_progress_bar and pbar is None:
                from tqdm.auto import tqdm

                total: int | None = None
                if max_records is not None:
                    total = max_records
                elif isinstance(data.get("total"), int):
                    total = int(data["total"])
                pbar = tqdm(total=total, unit="study", desc="Fetch NADA filters")

            for rec in batch:
                yield rec
                seen += 1
                if pbar is not None:
                    pbar.update(1)
                if max_records is not None and seen >= max_records:
                    return

            if not data.get("has_more"):
                break
            offset += page_size
    finally:
        if pbar is not None:
            pbar.close()


def fetch_all_study_records(
    settings: Settings,
    *,
    include_admin_metadata: bool = True,
    include_metadata: bool = False,
    page_size: int = 100,
    max_records: int | None = None,
    show_progress_bar: bool = False,
) -> list[dict[str, Any]]:
    return list(
        iter_study_records(
            settings,
            include_admin_metadata=include_admin_metadata,
            include_metadata=include_metadata,
            page_size=page_size,
            max_records=max_records,
            show_progress_bar=show_progress_bar,
        )
    )
