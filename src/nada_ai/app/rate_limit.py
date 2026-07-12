"""Lightweight in-memory rate limiting for public (unauthenticated) endpoints.

Fixed-window counter per client key (the presented admin key if any, else
client IP), held in a single process's memory — intentionally simple, no new
dependency. It does not coordinate across multiple worker processes; if the
app is scaled to multiple processes/replicas, swap the per-process dict for a
shared store (Redis) behind the same :meth:`RateLimiter.check` interface.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request

from nada_ai.app.state import AppState, get_state


@dataclass
class _Bucket:
    window: int = 0
    count: int = 0


class RateLimiter:
    def __init__(self, limit_per_minute: int, *, max_tracked_keys: int = 20_000) -> None:
        self.limit = limit_per_minute
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()
        self._max_tracked_keys = max_tracked_keys

    async def check(self, key: str) -> bool:
        """Return ``True`` if this call is within the limit (and counts it)."""
        if self.limit <= 0:
            return True
        now_minute = int(time.time() // 60)
        async with self._lock:
            if len(self._buckets) > self._max_tracked_keys:
                stale = [k for k, b in self._buckets.items() if b.window < now_minute]
                for k in stale:
                    self._buckets.pop(k, None)
            bucket = self._buckets.get(key)
            if bucket is None or bucket.window != now_minute:
                bucket = _Bucket(window=now_minute, count=0)
                self._buckets[key] = bucket
            bucket.count += 1
            return bucket.count <= self.limit


def rate_limited(attr: str):
    """FastAPI dependency factory. ``attr`` names a ``RateLimiter`` on ``AppState``."""

    async def dependency(request: Request, s: AppState = Depends(get_state)) -> None:
        limiter: RateLimiter | None = getattr(s, attr, None)
        if limiter is None:
            return
        client_key = request.headers.get("X-NADA-Admin-Key") or (
            request.client.host if request.client else "unknown"
        )
        if not await limiter.check(client_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded, slow down")

    return dependency
