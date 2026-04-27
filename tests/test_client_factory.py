"""Smoke-test client construction (no cluster I/O)."""

import asyncio

from nada_ai.search.backend.opensearch.client import build_async_client, build_client
from nada_ai.settings import Settings


def test_build_sync_client_basic():
    c = build_client(Settings())
    try:
        assert c.transport is not None
    finally:
        c.transport.close()


def test_build_async_client_basic():
    async def _run():
        c = build_async_client(Settings())
        try:
            assert c.transport is not None
        finally:
            await c.close()

    asyncio.run(_run())
