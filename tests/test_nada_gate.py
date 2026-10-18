"""The request gate in front of NADA (nada_ai.nada.gate)."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from nada_ai.nada.gate import (
    GatedTransport,
    NadaAskedToWait,
    NadaFirewallBlocked,
    NadaGate,
    NadaUnavailable,
    is_firewall_response,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


def _gate(clock: _Clock, **kw) -> NadaGate:
    return NadaGate(
        min_interval_seconds=kw.get("interval", 0.5),
        max_retries=kw.get("retries", 3),
        retry_base_seconds=kw.get("base", 5.0),
        retry_after_max_seconds=kw.get("retry_after_max", 60),
        sleep=clock.sleep,
        clock=clock,
    )


def _send(gate: NadaGate, responses: list) -> httpx.Response:
    """Send one request through the gate; ``responses`` are what the server answers, in order."""
    answers = iter(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return gate.send(handle, httpx.Request("GET", "https://nada.example/api/admin/search-index/queue"))


def test_requests_are_spaced_by_the_minimum_interval():
    clock = _Clock()
    gate = _gate(clock, interval=0.5)
    _send(gate, [httpx.Response(200)])
    _send(gate, [httpx.Response(200)])
    assert clock.sleeps == [0.5]


def test_a_temporary_failure_is_retried_with_a_growing_wait():
    clock = _Clock()
    gate = _gate(clock, interval=0, base=5.0)
    response = _send(gate, [httpx.Response(502), httpx.Response(503, json={}), httpx.Response(200)])
    assert response.status_code == 200
    # 5 s, then 15 s (plus up to a quarter of jitter each)
    assert 5 <= clock.sleeps[0] <= 6.25 and 15 <= clock.sleeps[1] <= 18.75
    assert gate.last_trouble is None


def test_retry_after_is_honoured():
    clock = _Clock()
    gate = _gate(clock, interval=0)
    _send(gate, [httpx.Response(429, headers={"Retry-After": "42"}), httpx.Response(200)])
    assert clock.sleeps == [42.0]


def test_a_longer_retry_after_is_not_waited_out():
    clock = _Clock()
    gate = _gate(clock, interval=0, retry_after_max=60)
    response = _send(gate, [httpx.Response(503, json={}, headers={"Retry-After": "3600"})])
    assert response.status_code == 503
    assert clock.sleeps == []
    assert gate.last_trouble.kind == "unavailable"
    assert "Retry-After 3600s" in gate.last_trouble.message
    assert 3590 < gate.not_before - time.time() <= 3600  # the scheduler waits until then


def test_after_the_last_retry_the_response_is_returned_and_recorded():
    clock = _Clock()
    gate = _gate(clock, interval=0, retries=2)
    response = _send(gate, [httpx.Response(500)] * 3)
    assert response.status_code == 500
    assert gate.last_trouble.kind == "unavailable"
    assert gate.trouble_since(0) is not None


def test_nothing_goes_out_before_the_time_nada_asked_for():
    clock = _Clock()
    gate = _gate(clock, interval=0, retry_after_max=60)
    _send(gate, [httpx.Response(429, json={}, headers={"Retry-After": "900"})])
    with pytest.raises(NadaAskedToWait) as refused:
        _send(gate, [])  # the server is never called: there is no answer to give
    assert 890 < refused.value.retry_at - time.time() <= 900

    gate.not_before = time.time() - 1  # the time has come
    assert _send(gate, [httpx.Response(200)]).status_code == 200


def test_a_connection_error_that_keeps_failing_raises_nada_unavailable():
    clock = _Clock()
    gate = _gate(clock, interval=0, retries=1)
    with pytest.raises(NadaUnavailable):
        _send(gate, [httpx.ConnectError("refused"), httpx.ConnectError("refused")])


def test_a_firewall_block_is_not_retried():
    clock = _Clock()
    gate = _gate(clock, interval=0)
    blocked = httpx.Response(
        403, headers={"cf-ray": "abc123", "content-type": "text/html"}, text="<html>Attention Required</html>"
    )
    with pytest.raises(NadaFirewallBlocked) as e:
        _send(gate, [blocked])
    assert e.value.cf_ray == "abc123"
    assert clock.sleeps == []
    assert gate.last_trouble.kind == "firewall"


def test_nadas_own_json_403_is_not_a_firewall():
    response = httpx.Response(403, json={"status": "failed", "message": "ACCESS_DENIED"})
    assert is_firewall_response(response) is False
    assert _send(_gate(_Clock(), interval=0), [response]).status_code == 403


def test_a_php_html_500_is_temporary_not_a_firewall():
    assert is_firewall_response(httpx.Response(500, text="<html>Fatal error</html>")) is False


def test_a_429_slows_the_gate_down():
    clock = _Clock()
    gate = _gate(clock, interval=0.5)
    _send(gate, [httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200)])
    assert gate.interval_seconds() == 1.0


def test_only_one_request_is_in_flight_at_a_time():
    gate = NadaGate(min_interval_seconds=0, max_retries=0, retry_base_seconds=0, retry_after_max_seconds=1)
    in_flight = 0
    most = 0
    lock = threading.Lock()

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, most
        with lock:
            in_flight += 1
            most = max(most, in_flight)
        threading.Event().wait(0.01)
        with lock:
            in_flight -= 1
        return httpx.Response(200)

    threads = [
        threading.Thread(target=gate.send, args=(handle, httpx.Request("GET", "https://nada.example/x")))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert most == 1


def test_the_transport_gates_nadas_host_only():
    sent: list[str] = []

    class Inner(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            sent.append(request.url.host)
            return httpx.Response(200)

    gate = NadaGate(min_interval_seconds=0, max_retries=0, retry_base_seconds=0, retry_after_max_seconds=1)
    transport = GatedTransport(gate, "nada.example", inner=Inner())
    with httpx.Client(transport=transport) as client:
        client.get("https://nada.example/api/x")
        client.get("https://elsewhere.example/doc.pdf")
    assert sent == ["nada.example", "elsewhere.example"]
    assert gate.requests_last_hour() == 1


def test_closing_a_client_leaves_the_shared_transport_open():
    closed: list[bool] = []

    class Inner(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        def close(self) -> None:
            closed.append(True)

    gate = NadaGate(min_interval_seconds=0, max_retries=0, retry_base_seconds=0, retry_after_max_seconds=1)
    transport = GatedTransport(gate, "nada.example", inner=Inner())
    for _ in range(2):
        with httpx.Client(transport=transport) as client:
            client.get("https://nada.example/api/x")
    assert closed == []


def test_uninstall_closes_the_shared_connections_and_forgets_the_gate():
    from ai4data.discovery import transport as ai4data_transport

    from nada_ai.nada import gate as gate_module
    from nada_ai.settings import Settings

    closed: list[bool] = []

    class Inner(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        def close(self) -> None:
            closed.append(True)

    transport = gate_module.install(Settings(), "https://nada.example/api")
    transport.inner = Inner()
    gate_module.uninstall()
    assert closed == [True]
    assert gate_module.get_gate() is None and ai4data_transport._client is None


def test_the_app_closes_the_connections_to_nada_on_shutdown(monkeypatch):
    from starlette.testclient import TestClient

    from nada_ai.app.main import app
    from nada_ai.nada import gate as gate_module

    calls: list[bool] = []
    monkeypatch.setattr(gate_module, "uninstall", lambda: calls.append(True))
    with TestClient(app):
        assert calls == []
    assert calls == [True]
