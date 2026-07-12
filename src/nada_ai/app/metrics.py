"""In-process HTTP metrics, exposed in Prometheus text-exposition format.

No external dependency: this hand-rolls the minimal subset of the format
(``# HELP``, ``# TYPE``, and ``name{labels} value`` lines) a Prometheus
server (or any promtool-compatible scraper) needs to ingest a request
counter and a latency histogram. If richer metrics (gauges, summaries,
multi-process aggregation) are needed later, swap this module for
``prometheus_client`` behind the same ``record`` / ``render_prometheus``
interface — nothing else in the app needs to change.

Per-process, in-memory — does not aggregate across multiple worker
processes/replicas. Fine for the current single-process deployment; a
multi-worker deployment would need a shared backend (e.g. the
``prometheus_client`` multiprocess mode, or push to a collector).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field

_LATENCY_BUCKETS_SECONDS: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


@dataclass
class _RouteStats:
    count_by_status: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    #: Per-bucket (non-cumulative) observation counts: index i is bucket
    #: ``(_LATENCY_BUCKETS_SECONDS[i-1], _LATENCY_BUCKETS_SECONDS[i]]``, and the
    #: trailing slot is the +Inf overflow bucket. ``render_prometheus`` turns
    #: this into the cumulative "le" counts Prometheus's histogram format expects.
    bucket_counts: list[int] = field(default_factory=lambda: [0] * (len(_LATENCY_BUCKETS_SECONDS) + 1))
    sum_seconds: float = 0.0
    total_count: int = 0


class MetricsRegistry:
    def __init__(self) -> None:
        self._routes: dict[str, _RouteStats] = defaultdict(_RouteStats)
        self._lock = asyncio.Lock()

    async def record(self, route: str, status_code: int, duration_seconds: float) -> None:
        async with self._lock:
            stats = self._routes[route]
            stats.count_by_status[status_code] += 1
            stats.total_count += 1
            stats.sum_seconds += duration_seconds
            for i, bound in enumerate(_LATENCY_BUCKETS_SECONDS):
                if duration_seconds <= bound:
                    stats.bucket_counts[i] += 1
                    break
            else:
                stats.bucket_counts[-1] += 1  # exceeded every finite bound -> +Inf overflow bucket

    def render_prometheus(self) -> str:
        lines: list[str] = [
            "# HELP nada_http_requests_total Total HTTP requests by route and status code.",
            "# TYPE nada_http_requests_total counter",
        ]
        for route, stats in sorted(self._routes.items()):
            for status_code, count in sorted(stats.count_by_status.items()):
                lines.append(f'nada_http_requests_total{{route="{route}",status="{status_code}"}} {count}')

        lines += [
            "# HELP nada_http_request_duration_seconds HTTP request latency.",
            "# TYPE nada_http_request_duration_seconds histogram",
        ]
        for route, stats in sorted(self._routes.items()):
            cumulative = 0
            for i, bound in enumerate(_LATENCY_BUCKETS_SECONDS):
                cumulative += stats.bucket_counts[i]
                lines.append(
                    f'nada_http_request_duration_seconds_bucket{{route="{route}",le="{bound}"}} {cumulative}'
                )
            cumulative += stats.bucket_counts[-1]
            lines.append(f'nada_http_request_duration_seconds_bucket{{route="{route}",le="+Inf"}} {cumulative}')
            lines.append(f'nada_http_request_duration_seconds_sum{{route="{route}"}} {stats.sum_seconds}')
            lines.append(f'nada_http_request_duration_seconds_count{{route="{route}"}} {stats.total_count}')
        return "\n".join(lines) + "\n"
