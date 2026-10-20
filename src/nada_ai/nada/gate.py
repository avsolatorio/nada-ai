"""Every request nada-ai sends to NADA goes through one gate.

NADA is a PHP application whose workers also serve its public site, and it does not limit its API callers itself. So
nada-ai never sends it two requests at a time: one request is in flight at most, a minimum gap separates the end of
one from the start of the next, and a request that fails because NADA (or a proxy in front of it) is busy or down is
retried after a growing wait. What the retries could not fix is recorded, for the scheduler to decide whether NADA or
one item is the problem (see ``app.reconcile_scheduler``).

The gate covers the search-index client (``ingest.search_index_sync``) and, through ai4data's
``discovery.transport`` hook, every metadata request ai4data makes — both are given an ``httpx.Client`` whose transport
is :class:`GatedTransport`. Only requests to NADA's host pass through the gate; nada-ai's MCP tools use their own client
and stay outside it, so a chat user never waits behind indexing.

The gate is per process: "one request at a time" holds for the whole deployment only with one app worker (the Docker
image runs one). Another worker's requests, or a CLI run's, go through a gate of their own.

Failure kinds:

- temporary (timeouts, connection errors, 408, 429, 5xx): retried; after the last retry the response (or the error) is
  handed back and the failure is recorded as ``unavailable`` (or ``rate_limited`` for 429). A ``Retry-After`` longer
  than :data:`RETRY_AFTER_MAX_SECONDS` ends the retries at once: the gate does not hold every other request that long.
  It is kept instead: until that time no request goes out, each raises :class:`NadaAskedToWait`.
- firewall (403/503 that is not NADA's JSON, with Cloudflare's markers): not retried — repeating requests can turn a
  short block into a long one. Raises :class:`NadaFirewallBlocked`.
- anything else (2xx, NADA's own JSON 4xx answers): returned as it is, for the caller to read.
"""

from __future__ import annotations

import email.utils
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

TEMPORARY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, *range(520, 531)})
#: After a 429, the gap between requests is doubled for this long.
_RATE_LIMIT_SLOWDOWN_SECONDS = 3600
#: The longest ``Retry-After`` the gate waits out itself (about its own longest backoff).
RETRY_AFTER_MAX_SECONDS = 60


class NadaTrouble(RuntimeError):
    """NADA (or what is in front of it) could not serve a request — not a problem with what was asked."""

    kind = "unavailable"


class NadaUnavailable(NadaTrouble):
    kind = "unavailable"


class NadaAskedToWait(NadaTrouble):
    """Not sent: NADA's ``Retry-After`` asked to wait until ``retry_at`` (``time.time()``)."""

    kind = "asked_to_wait"

    def __init__(self, message: str, *, retry_at: float) -> None:
        super().__init__(message)
        self.retry_at = retry_at


class NadaFirewallBlocked(NadaTrouble):
    kind = "firewall"

    def __init__(self, message: str, *, cf_ray: str | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.cf_ray = cf_ray
        self.status = status


@dataclass(frozen=True)
class Trouble:
    """The last failure the gate could not retry away."""

    at: float  # time.time()
    kind: str  # unavailable | rate_limited | firewall
    status: int | None
    message: str
    cf_ray: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": int(self.at),
            "kind": self.kind,
            "http_status": self.status,
            "message": self.message,
            "cf_ray": self.cf_ray,
        }


def is_firewall_response(response: httpx.Response) -> bool:
    """A 403/503 from a firewall or CDN in front of NADA, not from NADA: NADA always answers JSON."""
    if response.status_code not in (403, 503):
        return False
    content_type = response.headers.get("content-type", "").lower()
    if "json" in content_type:
        return False
    if "cf-mitigated" in response.headers or "cf-ray" in response.headers:
        return True
    body = response.content[:4000].lower()
    return b"cloudflare" in body or b"attention required" in body or b"challenge" in body


def retry_after_seconds(response: httpx.Response) -> float | None:
    """``Retry-After`` (seconds or an HTTP date) in seconds from now."""
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        parsed = email.utils.parsedate_to_datetime(value) if value else None
        if parsed is None:
            return None
        seconds = parsed.timestamp() - time.time()
    return max(0.0, seconds)


class NadaGate:
    """One request to NADA at a time, a minimum gap between requests, retries with backoff.

    ``retry_after_max_seconds``: the longest ``Retry-After`` the gate waits out itself. It holds every other request
    while it waits, so a longer one is not retried: the response is handed back and the scheduler pauses instead.
    ``sleep`` and ``clock`` are injectable for tests.
    """

    def __init__(
        self,
        *,
        min_interval_seconds: float,
        max_retries: int,
        retry_base_seconds: float,
        retry_after_max_seconds: float,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_after_max_seconds = retry_after_max_seconds
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last_end = float("-inf")
        self._slow_until = float("-inf")
        self._sent: deque[float] = deque()  # time.time() of each request sent, for the last hour
        self.last_trouble: Trouble | None = None
        #: The earliest time (``time.time()``) NADA said to come back, from the ``Retry-After`` of a response the gate
        #: gave up on: no request goes out before it (:class:`NadaAskedToWait`), and the scheduler pauses until then.
        self.not_before = float("-inf")

    # -- what the scheduler reads ---------------------------------------------------------------------------------

    def trouble_since(self, since: float) -> Trouble | None:
        """The last failure recorded at or after ``since`` (``time.time()``), if any."""
        trouble = self.last_trouble
        return trouble if trouble is not None and trouble.at >= since else None

    def requests_last_hour(self) -> int:
        cutoff = time.time() - 3600
        while self._sent and self._sent[0] < cutoff:
            self._sent.popleft()
        return len(self._sent)

    def interval_seconds(self) -> float:
        slow = self._clock() < self._slow_until
        return self.min_interval_seconds * (2 if slow else 1)

    # -- sending --------------------------------------------------------------------------------------------------

    def _record(self, kind: str, status: int | None, message: str, cf_ray: str | None = None) -> None:
        self.last_trouble = Trouble(at=time.time(), kind=kind, status=status, message=message[:500], cf_ray=cf_ray)
        logger.warning("NADA request failed (%s%s): %s", kind, f", HTTP {status}" if status else "", message[:300])

    def _pace(self) -> None:
        wait = self._last_end + self.interval_seconds() - self._clock()
        if wait > 0:
            self._sleep(wait)

    def _backoff(self, attempt: int, response: httpx.Response | None) -> float | None:
        """The wait before the next attempt; None when NADA asks for a longer one than the gate waits out."""
        if response is not None:
            after = retry_after_seconds(response)
            if after is not None:
                return after if after <= self.retry_after_max_seconds else None
        base = self.retry_base_seconds * (3**attempt)
        return base + random.uniform(0, base / 4)

    def send(self, handle, request: httpx.Request) -> httpx.Response:
        """Send ``request`` with ``handle`` (a transport's ``handle_request``) under the gate's rules."""
        with self._lock:
            if time.time() < self.not_before:
                wait = int(self.not_before - time.time())
                raise NadaAskedToWait(
                    f"{request.method} {request.url.path} not sent: NADA asked to wait {wait}s more (Retry-After)",
                    retry_at=self.not_before,
                )
            response: httpx.Response | None = None
            for attempt in range(self.max_retries + 1):
                self._pace()
                error: Exception | None = None
                response = None
                try:
                    response = handle(request)
                    response.read()
                except httpx.TransportError as e:  # timeouts included
                    error = e
                finally:
                    self._last_end = self._clock()
                    self._sent.append(time.time())

                if response is not None:
                    if is_firewall_response(response):
                        ray = response.headers.get("cf-ray")
                        message = f"{request.method} {request.url.path} blocked (HTTP {response.status_code})"
                        self._record("firewall", response.status_code, message, ray)
                        raise NadaFirewallBlocked(message, cf_ray=ray, status=response.status_code)
                    if response.status_code not in TEMPORARY_STATUSES:
                        return response
                    if response.status_code == 429:
                        self._slow_until = self._clock() + _RATE_LIMIT_SLOWDOWN_SECONDS

                if attempt == self.max_retries:
                    break
                wait = self._backoff(attempt, response)
                if wait is None:
                    break
                self._sleep(wait)

            if response is not None:
                kind = "rate_limited" if response.status_code == 429 else "unavailable"
                message = f"{request.method} {request.url.path}: HTTP {response.status_code}"
                after = retry_after_seconds(response)
                if after is not None:
                    message += f" (Retry-After {int(after)}s)"
                    self.not_before = max(self.not_before, time.time() + after)
                self._record(kind, response.status_code, message)
                return response
            assert error is not None
            self._record("unavailable", None, f"{request.method} {request.url.path}: {error}")
            raise NadaUnavailable(f"{request.method} {request.url.path}: {error}") from error


class GatedTransport(httpx.BaseTransport):
    """Sends requests to NADA's host through the gate; any other host directly."""

    def __init__(self, gate: NadaGate, nada_host: str, inner: httpx.BaseTransport | None = None) -> None:
        self.gate = gate
        self.nada_host = nada_host.lower()
        self.inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host.lower() == self.nada_host:
            return self.gate.send(self.inner.handle_request, request)
        return self.inner.handle_request(request)

    def close(self) -> None:
        """Nothing: the process's one transport outlives every client built on it. httpx closes a client's transport
        when the client is closed, which would drop the pool's connections (keep-alive, and a request another thread
        has in flight)."""


_gate: NadaGate | None = None
_transport: GatedTransport | None = None
_install_lock = threading.Lock()


def nada_host(base_url: str) -> str:
    host = urlsplit(base_url).hostname
    if not host:
        raise ValueError(f"No host in NADA URL {base_url!r}")
    return host


def install(settings: Any, base_url: str) -> GatedTransport:
    """Create the process's gate for NADA at ``base_url`` (once) and send ai4data's requests through it."""
    global _gate, _transport
    with _install_lock:
        if _transport is not None:
            return _transport
        from ai4data.discovery import transport as ai4data_transport

        _gate = NadaGate(
            min_interval_seconds=settings.request_min_interval_ms / 1000.0,
            max_retries=settings.request_max_retries,
            retry_base_seconds=settings.request_retry_base_seconds,
            retry_after_max_seconds=RETRY_AFTER_MAX_SECONDS,
        )
        _transport = GatedTransport(_gate, nada_host(base_url))
        # the timeout of an ai4data request that sets none (its metadata extract and document downloads set their own)
        ai4data_transport.set_http_client(httpx.Client(transport=_transport, timeout=settings.request_timeout_seconds))
        return _transport


def get_gate() -> NadaGate | None:
    """The process's gate, once :func:`install` ran."""
    return _gate


def uninstall() -> None:
    """Close the process's connections to NADA and forget the gate: at app shutdown (closing a client does not close
    the shared transport, see :meth:`GatedTransport.close`), and between tests, which install their own."""
    global _gate, _transport
    from ai4data.discovery import transport as ai4data_transport

    with _install_lock:
        if _transport is not None:
            _transport.inner.close()
        _gate = None
        _transport = None
        ai4data_transport.set_http_client(None)
