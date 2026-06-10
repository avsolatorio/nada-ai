"""Fetch study filters from IHSN search-metadata-extract API."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator

from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://training.ihsn.org/index.php/api/admin/search-metadata-extract"


class IhsnExtractError(RuntimeError):
    """Raised when the metadata-extract API returns an error payload."""


def _build_headers(settings: Settings) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "nada-ai-filters-cli/1.0"}
    if settings.ihsn_api_key:
        headers["X-API-KEY"] = settings.ihsn_api_key
    if settings.ihsn_auth_bearer:
        headers["Authorization"] = f"Bearer {settings.ihsn_auth_bearer}"
    if settings.ihsn_auth_cookie:
        headers["Cookie"] = settings.ihsn_auth_cookie
    return headers


def _base_url(settings: Settings) -> str:
    return (settings.ihsn_metadata_extract_base_url or DEFAULT_BASE_URL).rstrip("/")


def _fetch_json(url: str, settings: Settings, *, timeout: float = 120.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=_build_headers(settings), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise IhsnExtractError(f"HTTP {e.code} for {url}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise IhsnExtractError(f"Request failed for {url}: {e}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise IhsnExtractError(f"Non-JSON response from {url}: {raw[:500]}") from e

    if not isinstance(data, dict):
        raise IhsnExtractError(f"Expected JSON object from {url}, got {type(data).__name__}")

    status = str(data.get("status") or "").lower()
    if status not in {"", "success", "ok"}:
        message = data.get("message") or data.get("error") or status
        raise IhsnExtractError(f"API error ({status}): {message}")

    return data


def _study_idno(study: dict[str, Any]) -> str | None:
    core = study.get("core_fields")
    if isinstance(core, dict):
        idno = core.get("idno")
        if idno:
            return str(idno).strip()
    idno = study.get("idno")
    return str(idno).strip() if idno else None


def study_to_sync_record(study: dict[str, Any]) -> dict[str, Any] | None:
    """Extract ``{idno, filters}`` from one study payload."""
    idno = _study_idno(study)
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


def _studies_list_url(
    settings: Settings,
    *,
    offset: int,
    limit: int,
    include_admin_metadata: bool,
    include_metadata: bool,
) -> str:
    params = {
        "offset": offset,
        "limit": limit,
        "include_admin_metadata": "1" if include_admin_metadata else "0",
        "include_metadata": "1" if include_metadata else "0",
    }
    query = urllib.parse.urlencode(params)
    return f"{_base_url(settings)}/studies?{query}"


def _study_url(
    settings: Settings,
    idno: str,
    *,
    include_admin_metadata: bool,
    include_metadata: bool,
) -> str:
    params = {
        "include_admin_metadata": "1" if include_admin_metadata else "0",
        "include_metadata": "1" if include_metadata else "0",
    }
    query = urllib.parse.urlencode(params)
    encoded_idno = urllib.parse.quote(idno.strip(), safe="")
    return f"{_base_url(settings)}/studies/{encoded_idno}?{query}"


def fetch_study_records(
    settings: Settings,
    idno: str,
    *,
    include_admin_metadata: bool = True,
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Fetch filters for a single study idno."""
    url = _study_url(
        settings,
        idno,
        include_admin_metadata=include_admin_metadata,
        include_metadata=include_metadata,
    )
    data = _fetch_json(url, settings)
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
) -> Iterator[dict[str, Any]]:
    """Paginate all studies and yield ``{idno, filters}`` records."""
    offset = 0
    seen = 0
    while True:
        url = _studies_list_url(
            settings,
            offset=offset,
            limit=page_size,
            include_admin_metadata=include_admin_metadata,
            include_metadata=include_metadata,
        )
        data = _fetch_json(url, settings)
        batch = parse_extract_response(data)
        if not batch:
            break
        for rec in batch:
            yield rec
            seen += 1
            if max_records is not None and seen >= max_records:
                return
        if not data.get("has_more"):
            break
        offset += page_size


def fetch_all_study_records(
    settings: Settings,
    *,
    include_admin_metadata: bool = True,
    include_metadata: bool = False,
    page_size: int = 100,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    return list(
        iter_study_records(
            settings,
            include_admin_metadata=include_admin_metadata,
            include_metadata=include_metadata,
            page_size=page_size,
            max_records=max_records,
        )
    )
