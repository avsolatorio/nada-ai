"""Tests for the hand-rolled Prometheus-format in-process metrics registry."""

from __future__ import annotations

import asyncio

from nada_ai.app.metrics import MetricsRegistry


def test_histogram_buckets_are_correctly_cumulative():
    """Regression test: bucket counts must not be double-accumulated.

    Each observation should land in exactly one non-cumulative bucket at
    write time; ``render_prometheus`` performs the cumulative summation
    required by the Prometheus "le" histogram convention. A prior bug
    incremented every matching bucket at write time *and* re-summed them at
    render time, making bucket counts grow without bound across buckets.
    """

    async def run() -> str:
        reg = MetricsRegistry()
        await reg.record("/x", 200, 0.005)
        await reg.record("/x", 200, 0.005)
        await reg.record("/x", 200, 0.005)
        await reg.record("/x", 200, 3.0)
        await reg.record("/x", 500, 20.0)
        return reg.render_prometheus()

    text = asyncio.run(run())

    assert 'nada_http_request_duration_seconds_bucket{route="/x",le="0.01"} 3' in text
    assert 'nada_http_request_duration_seconds_bucket{route="/x",le="1"} 3' in text
    assert 'nada_http_request_duration_seconds_bucket{route="/x",le="5"} 4' in text
    assert 'nada_http_request_duration_seconds_bucket{route="/x",le="10"} 4' in text
    assert 'nada_http_request_duration_seconds_bucket{route="/x",le="+Inf"} 5' in text
    assert 'nada_http_request_duration_seconds_count{route="/x"} 5' in text
    assert 'nada_http_requests_total{route="/x",status="200"} 4' in text
    assert 'nada_http_requests_total{route="/x",status="500"} 1' in text


def test_bucket_counts_never_exceed_total_observations():
    """Every cumulative bucket must be monotonically non-decreasing and capped at total_count."""

    async def run() -> str:
        reg = MetricsRegistry()
        for d in [0.001, 0.02, 0.2, 2.0, 0.001, 0.001, 0.001]:
            await reg.record("/y", 200, d)
        return reg.render_prometheus()

    text = asyncio.run(run())
    bucket_values = []
    for line in text.splitlines():
        if line.startswith("nada_http_request_duration_seconds_bucket") and 'route="/y"' in line:
            bucket_values.append(int(line.rsplit(" ", 1)[1]))

    assert bucket_values == sorted(bucket_values)
    assert all(v <= 7 for v in bucket_values)
    assert bucket_values[-1] == 7


def test_separate_routes_have_independent_stats():
    async def run() -> str:
        reg = MetricsRegistry()
        await reg.record("/a", 200, 0.001)
        await reg.record("/b", 200, 0.001)
        await reg.record("/b", 200, 0.001)
        return reg.render_prometheus()

    text = asyncio.run(run())
    assert 'nada_http_requests_total{route="/a",status="200"} 1' in text
    assert 'nada_http_requests_total{route="/b",status="200"} 2' in text
