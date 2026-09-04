"""Tests for the consolidated NADA admin-API credential resolution
(nada_ai.nada.admin_auth). There is exactly one outbound credential source —
AI4DATA_METADATA_CATALOG_* (ai4data.discovery.config.metadata_catalog) — used
by both filters/metadata_extract.py and ingest/search_index_sync.py. A prior
design had a second, per-feature override layer (NADA_METADATA_EXTRACT_API_KEY
etc.); it was removed as redundant since nothing gives those features a
reason to ever use different credentials than the main catalog client."""

from __future__ import annotations

from unittest.mock import patch

from nada_ai.nada.admin_auth import resolve_admin_cookies, resolve_admin_headers, scrub_admin_credentials


def test_resolve_admin_headers_reads_x_api_key():
    with patch("nada_ai.nada.admin_auth.metadata_catalog") as mock_catalog:
        mock_catalog.x_api_key = "the-key"
        mock_catalog.auth_bearer = None
        mock_catalog.cookies = None
        headers = resolve_admin_headers(user_agent="ua/1.0")
    assert headers["X-API-KEY"] == "the-key"
    assert "Authorization" not in headers
    assert "Cookie" not in headers
    assert headers["User-Agent"] == "ua/1.0"


def test_resolve_admin_headers_bearer_and_cookie():
    with patch("nada_ai.nada.admin_auth.metadata_catalog") as mock_catalog:
        mock_catalog.x_api_key = None
        mock_catalog.auth_bearer = "tok"
        mock_catalog.cookies = "a=1; b=2"
        headers = resolve_admin_headers(user_agent="ua/1.0")
    assert headers["Authorization"] == "Bearer tok"
    assert headers["Cookie"] == "a=1; b=2"


def test_resolve_admin_headers_empty_when_nothing_configured():
    with patch("nada_ai.nada.admin_auth.metadata_catalog") as mock_catalog:
        mock_catalog.x_api_key = None
        mock_catalog.auth_bearer = None
        mock_catalog.cookies = None
        headers = resolve_admin_headers(user_agent="ua/1.0")
    assert set(headers) == {"Accept", "User-Agent"}


def test_resolve_admin_cookies_parses_into_dict():
    with patch("nada_ai.nada.admin_auth.metadata_catalog") as mock_catalog:
        mock_catalog.cookies = " a=1 ; b=2"
        cookies = resolve_admin_cookies()
    assert cookies == {"a": "1", "b": "2"}


def test_resolve_admin_cookies_empty_when_none_configured():
    with patch("nada_ai.nada.admin_auth.metadata_catalog") as mock_catalog:
        mock_catalog.cookies = None
        assert resolve_admin_cookies() == {}


def test_scrub_admin_credentials_redacts_known_secrets():
    with patch("nada_ai.nada.admin_auth.metadata_catalog") as mock_catalog:
        mock_catalog.x_api_key = "super-secret-key"
        mock_catalog.auth_bearer = "super-secret-token"
        mock_catalog.cookies = None
        msg = "request failed: key=super-secret-key token=super-secret-token"
        scrubbed = scrub_admin_credentials(msg)
    assert "super-secret-key" not in scrubbed
    assert "super-secret-token" not in scrubbed
    assert scrubbed.count("[REDACTED]") == 2


def test_scrub_admin_credentials_noop_when_nothing_configured():
    with patch("nada_ai.nada.admin_auth.metadata_catalog") as mock_catalog:
        mock_catalog.x_api_key = None
        mock_catalog.auth_bearer = None
        mock_catalog.cookies = None
        msg = "plain error, no secrets here"
        assert scrub_admin_credentials(msg) == msg
