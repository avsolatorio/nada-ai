from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from opensearchpy import AsyncOpenSearch, OpenSearch

from nada_ai.settings import Settings

#: ``request_timeout`` of a read that must not hold anything up when OpenSearch hangs (health probes, the index mapping
#: check, the study index generation read before every request to NADA). The clients' own timeout is for bulk work.
QUICK_REQUEST_TIMEOUT_SECONDS = 5


def _parse_url(settings: Settings) -> tuple[str, int, str, bool]:
    parsed = urlparse(settings.opensearch_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 9200)
    scheme = parsed.scheme or "http"
    use_ssl = scheme == "https"
    return host, port, scheme, use_ssl


def _hosts_kw(settings: Settings) -> list[dict[str, Any]]:
    host, port, scheme, _ = _parse_url(settings)
    return [{"host": host, "port": port, "scheme": scheme}]


def _http_auth(settings: Settings) -> tuple[str, str] | None:
    if settings.opensearch_user and settings.opensearch_password:
        return (settings.opensearch_user, settings.opensearch_password)
    return None


def build_client(settings: Settings) -> OpenSearch:
    """Synchronous client (ingest CLI, bulk)."""
    _, _, _, use_ssl = _parse_url(settings)

    return OpenSearch(
        hosts=_hosts_kw(settings),
        http_auth=_http_auth(settings),
        use_ssl=use_ssl,
        verify_certs=settings.opensearch_verify_certs if use_ssl else False,
        ca_certs=settings.opensearch_ca_certs,
        timeout=120,
    )


def build_async_client(settings: Settings) -> AsyncOpenSearch:
    """Async client for FastAPI (non-blocking OpenSearch I/O)."""
    _, _, _, use_ssl = _parse_url(settings)

    return AsyncOpenSearch(
        hosts=_hosts_kw(settings),
        http_auth=_http_auth(settings),
        use_ssl=use_ssl,
        verify_certs=settings.opensearch_verify_certs if use_ssl else False,
        ca_certs=settings.opensearch_ca_certs,
        timeout=120,
    )
