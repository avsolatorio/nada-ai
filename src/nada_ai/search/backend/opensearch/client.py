from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from opensearchpy import AsyncOpenSearch, OpenSearch, RequestsHttpConnection

from nada_ai.settings import Settings


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


def _aws_credentials(settings: Settings) -> Any:
    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            "AWS SigV4 auth requires boto3. Install: uv sync --extra aws (or pip install boto3)"
        ) from e
    session = boto3.Session(profile_name=settings.aws_profile)
    creds = session.get_credentials()
    if creds is None:
        raise ValueError("No AWS credentials found; configure AWS_ACCESS_KEY_ID, ~/.aws/credentials, or IAM role")
    region = settings.aws_region or session.region_name
    if not region:
        raise ValueError("Set NADA_AWS_REGION or default region in AWS config for SigV4 signing")
    return creds, region


def _http_auth_sync(settings: Settings) -> tuple[Any, dict[str, Any]]:
    """Returns (auth, extra_client_kwargs)."""
    if settings.opensearch_auth_mode == "aws_sigv4":
        from opensearchpy import AWSV4SignerAuth

        creds, region = _aws_credentials(settings)
        auth = AWSV4SignerAuth(creds, region, settings.aws_service)
        return auth, {"connection_class": RequestsHttpConnection}

    auth = None
    if settings.opensearch_user and settings.opensearch_password:
        auth = (settings.opensearch_user, settings.opensearch_password)
    return auth, {}


def _http_auth_async(settings: Settings) -> tuple[Any, dict[str, Any]]:
    if settings.opensearch_auth_mode == "aws_sigv4":
        from opensearchpy import AWSV4SignerAsyncAuth

        creds, region = _aws_credentials(settings)
        auth = AWSV4SignerAsyncAuth(creds, region, settings.aws_service)
        return auth, {}

    auth = None
    if settings.opensearch_user and settings.opensearch_password:
        auth = (settings.opensearch_user, settings.opensearch_password)
    return auth, {}


def build_client(settings: Settings) -> OpenSearch:
    """Synchronous client (ingest CLI, bulk)."""
    _, _, _, use_ssl = _parse_url(settings)
    auth, extra = _http_auth_sync(settings)

    return OpenSearch(
        hosts=_hosts_kw(settings),
        http_auth=auth,
        use_ssl=use_ssl,
        verify_certs=settings.opensearch_verify_certs if use_ssl else False,
        ca_certs=settings.opensearch_ca_certs,
        timeout=120,
        **extra,
    )


def build_async_client(settings: Settings) -> AsyncOpenSearch:
    """Async client for FastAPI (non-blocking OpenSearch I/O)."""
    _, _, _, use_ssl = _parse_url(settings)
    auth, extra = _http_auth_async(settings)

    return AsyncOpenSearch(
        hosts=_hosts_kw(settings),
        http_auth=auth,
        use_ssl=use_ssl,
        verify_certs=settings.opensearch_verify_certs if use_ssl else False,
        ca_certs=settings.opensearch_ca_certs,
        timeout=120,
        **extra,
    )
