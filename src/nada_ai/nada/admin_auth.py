"""Resolve the one NADA admin-API credential set, shared by every admin-only
surface this package talks to (search-metadata-extract, search-index).

Both surfaces require the same admin-capable NADA account per NADA's own API
docs ("will only work for 'ADMIN' user accounts only"), and so does
extract-mode content ingest — so there is exactly one canonical outbound
credential set for the whole package:
``ai4data.discovery.config.metadata_catalog`` (``AI4DATA_METADATA_CATALOG_X_API_KEY``/
``_AUTH_BEARER``/``_COOKIES``). There is deliberately no separate,
feature-specific override layer for these — a prior design had one
(``NADA_METADATA_EXTRACT_API_KEY`` etc.), and it was removed because nothing
in NADA's account model gives a reason for filters-sync or search-index to
ever use a *different* admin account than the main catalog client, so the
override only added a second place to configure the same one secret.

The base **URLs** for these two surfaces (``metadata_extract_base_url``,
``search_index_base_url`` in ``nada_ai.settings``) remain independently
overridable — unlike credentials, different path/host structure per surface
is a real (if rare) possibility, e.g. a nonstandard reverse-proxy route.

Trust-boundary reminder: these are OUTBOUND credentials (nada-ai
authenticating itself to NADA), never ``NADA_ADMIN_API_KEY`` (INBOUND —
callers authenticating to nada-ai). Never reuse one for the other.
"""

from __future__ import annotations

from ai4data.discovery.config import metadata_catalog


def resolve_admin_headers(*, user_agent: str) -> dict[str, str]:
    """Build request headers for a NADA admin-API call (X-API-KEY / Bearer / Cookie)."""
    headers = {"Accept": "application/json", "User-Agent": user_agent}
    if metadata_catalog.x_api_key:
        headers["X-API-KEY"] = metadata_catalog.x_api_key
    if metadata_catalog.auth_bearer:
        headers["Authorization"] = f"Bearer {metadata_catalog.auth_bearer}"
    if metadata_catalog.cookies:
        headers["Cookie"] = metadata_catalog.cookies
    return headers


def resolve_admin_cookies() -> dict[str, str]:
    """Parse the configured cookie string into a dict for httpx's ``cookies=`` kwarg."""
    raw = metadata_catalog.cookies
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        if name := name.strip():
            out[name] = value.strip()
    return out


def scrub_admin_credentials(msg: str) -> str:
    """Remove known admin-credential values from an exception message string."""
    secrets = [metadata_catalog.x_api_key, metadata_catalog.auth_bearer, metadata_catalog.cookies]
    for secret in filter(None, secrets):
        msg = msg.replace(secret, "[REDACTED]")
    return msg
