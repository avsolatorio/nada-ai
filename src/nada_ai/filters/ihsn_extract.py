"""Fetch study filters from IHSN search-metadata-extract API."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import ai4data.discovery.catalog.extract as catalog_extract
from ai4data.discovery.config import metadata_catalog

from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://training.ihsn.org/index.php/api/admin/search-metadata-extract"


class IhsnExtractError(RuntimeError):
    """Raised when the metadata-extract API returns an error payload."""


def _build_headers(settings: Settings) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "nada-ai-filters-cli/1.0"}
    api_key = settings.ihsn_api_key or metadata_catalog.x_api_key
    if api_key:
        headers["X-API-KEY"] = api_key
    bearer = settings.ihsn_auth_bearer or metadata_catalog.auth_bearer
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    cookie = settings.ihsn_auth_cookie or metadata_catalog.cookies
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _build_cookies(settings: Settings) -> dict[str, str]:
    raw = settings.ihsn_auth_cookie or metadata_catalog.cookies
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if name:
            out[name] = value.strip()
    return out


def _base_url(settings: Settings) -> str:
    if settings.ihsn_metadata_extract_base_url:
        return settings.ihsn_metadata_extract_base_url.rstrip("/")
    if url := catalog_extract.extract_base_url():
        return url
    return DEFAULT_BASE_URL


def _request_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "base_url": _base_url(settings),
        "headers": _build_headers(settings),
        "cookies": _build_cookies(settings),
    }


def _scrub_credentials(msg: str, settings: Settings) -> str:
    """Remove known credential values from an exception message string."""
    for secret in filter(None, [
        settings.ihsn_api_key,
        settings.ihsn_auth_bearer,
        settings.ihsn_auth_cookie,
    ]):
        msg = msg.replace(secret, "[REDACTED]")
    return msg


def _wrap_extract_error(exc: Exception, settings: Settings | None = None) -> IhsnExtractError:
    msg = _scrub_credentials(str(exc), settings) if settings is not None else str(exc)
    return IhsnExtractError(msg)


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
        raise _wrap_extract_error(e, settings) from e

    records = parse_extract_response(data)
    if not records:
        raise IhsnExtractError(f"No filters found in response for idno {idno!r}")
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
                raise _wrap_extract_error(e, settings) from e

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
                pbar = tqdm(total=total, unit="study", desc="Fetch IHSN filters")

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
