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

Failure kinds:

- temporary (timeouts, connection errors, 408, 429, 5xx): retried; after the last retry the response (or the error) is
  handed back and the failure is recorded as ``unavailable`` (or ``rate_limited`` for 429).
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


class NadaTrouble(RuntimeError):
    """NADA (or what is in front of it) could not serve a request — not a problem with what was asked."""

    kind = "unavailable"


class NadaUnavailable(NadaTrouble):
    kind = "unavailable"


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


def retry_after_seconds(response: httpx.Response, cap: float) -> float | None:
    """``Retry-After`` (seconds or an HTTP date), at most ``cap``."""
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
    return max(0.0, min(seconds, cap))


class NadaGate:
    """One request to NADA at a time, a minimum gap between requests, retries with backoff.

    ``sleep`` and ``clock`` are injectable for tests.
    """

    def __init__(
        self,
        *,
        min_interval_seconds: float,
        max_retries: int,
        retry_base_seconds: float,
        retry_after_cap_seconds: float,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_after_cap_seconds = retry_after_cap_seconds
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last_end = float("-inf")
        self._slow_until = float("-inf")
        self._sent: deque[float] = deque()  # time.time() of each request sent, for the last hour
        self.last_trouble: Trouble | None = None

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

    def _backoff(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            after = retry_after_seconds(response, self.retry_after_cap_seconds)
            if after is not None:
                return after
        base = self.retry_base_seconds * (3**attempt)
        return base + random.uniform(0, base / 4)

    def send(self, handle, request: httpx.Request) -> httpx.Response:
        """Send ``request`` with ``handle`` (a transport's ``handle_request``) under the gate's rules."""
        with self._lock:
            response: httpx.Response | None = None
            for attempt in range(self.max_retries + 1):
                self._pace()
                error: Exception | None = None
                response = None
                try:
                    response = handle(request)
                    response.read()
                except (httpx.TimeoutException, httpx.TransportError) as e:
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
                self._sleep(self._backoff(attempt, response))

            if response is not None:
                kind = "rate_limited" if response.status_code == 429 else "unavailable"
                self._record(
                    kind, response.status_code, f"{request.method} {request.url.path}: HTTP {response.status_code}"
                )
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
        self.inner.close()


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
            min_interval_seconds=settings.nada_request_min_interval_ms / 1000.0,
            max_retries=settings.nada_request_max_retries,
            retry_base_seconds=settings.nada_request_retry_base_seconds,
            retry_after_cap_seconds=settings.nada_firewall_pause_seconds,
        )
        _transport = GatedTransport(_gate, nada_host(base_url))
        ai4data_transport.set_http_client(httpx.Client(transport=_transport, timeout=120.0))
        return _transport


def get_gate() -> NadaGate | None:
    """The process's gate, once :func:`install` ran."""
    return _gate


def reset_for_tests() -> None:
    """Forget the installed gate (tests install their own)."""
    global _gate, _transport
    from ai4data.discovery import transport as ai4data_transport

    with _install_lock:
        _gate = None
        _transport = None
        ai4data_transport.set_http_client(None)
