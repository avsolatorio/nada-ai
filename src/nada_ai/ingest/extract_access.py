"""How the standalone indexers (variables, citations) reach NADA's search-metadata-extract API."""

from __future__ import annotations

from typing import Any

import ai4data.discovery.catalog.extract as catalog_extract

from nada_ai.nada.admin_auth import resolve_admin_cookies, resolve_admin_headers
from nada_ai.settings import Settings

_USER_AGENT = "nada-ai-extract-indexer/1.0"


class ExtractError(RuntimeError):
    """Raised when NADA's search-metadata-extract API cannot be reached or returns an error payload."""


def base_url(settings: Settings) -> str:
    if settings.metadata_extract_base_url:
        return settings.metadata_extract_base_url.rstrip("/")
    if url := catalog_extract.extract_base_url():
        return url
    raise ExtractError(
        "No metadata-extract base URL configured. Set NADA_METADATA_EXTRACT_BASE_URL "
        "(or AI4DATA_METADATA_CATALOG_EXTRACT_PATH) to the metadata-extract API for your NADA instance."
    )


def request_kwargs(settings: Settings) -> dict[str, Any]:
    """``base_url``/``headers``/``cookies`` for the ai4data extract client calls."""
    return {
        "base_url": base_url(settings),
        "headers": resolve_admin_headers(user_agent=_USER_AGENT),
        "cookies": resolve_admin_cookies(),
    }
