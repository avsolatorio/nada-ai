"""Unit tests for the in-memory JobRegistry single-flight semantics.

Tests use ``asyncio.run`` to avoid a pytest-asyncio dependency.
"""

from __future__ import annotations

import asyncio
import threading
import time

from nada_ai.app.jobs import JobRegistry, JobStatus
from nada_ai.ingest.progress import CancelToken


async def _wait_status(registry: JobRegistry, job_id: str, status: JobStatus, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        snap = registry.get(job_id)
        if snap is not None and snap.status == status:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {status} in {timeout}s")


def test_submit_runs_factory_and_records_result():
    async def main():
        registry = JobRegistry()

        async def factory():
            return {"indexed": 3}

        job = await registry.submit("k", "k:1", factory, params={"x": 1})
        assert job.was_already_running is False
        await _wait_status(registry, job.id, JobStatus.succeeded)
        snap = registry.get(job.id)
        assert snap is not None
        assert snap.result == {"indexed": 3}
        assert snap.error is None
        assert snap.finished_at is not None

    asyncio.run(main())


def test_submit_same_key_returns_running_job():
    async def main():
        registry = JobRegistry()
        gate = asyncio.Event()

        async def factory():
            await gate.wait()
            return {"ok": True}

        first = await registry.submit("k", "same", factory, params={})
        assert first.was_already_running is False
        await _wait_status(registry, first.id, JobStatus.running)

        second = await registry.submit("k", "same", factory, params={})
        assert second.was_already_running is True
        assert second.id == first.id

        gate.set()
        await _wait_status(registry, first.id, JobStatus.succeeded)

        third = await registry.submit("k", "same", factory, params={})
        assert third.was_already_running is False
        assert third.id != first.id

    asyncio.run(main())


def test_different_keys_run_concurrently():
    async def main():
        registry = JobRegistry()
        started_a = asyncio.Event()
        started_b = asyncio.Event()
        release = asyncio.Event()

        async def fa():
            started_a.set()
            await release.wait()
            return {"k": "a"}

        async def fb():
            started_b.set()
            await release.wait()
            return {"k": "b"}

        a = await registry.submit("k", "key:a", fa, params={})
        b = await registry.submit("k", "key:b", fb, params={})

        await asyncio.wait_for(started_a.wait(), timeout=1.0)
        await asyncio.wait_for(started_b.wait(), timeout=1.0)
        assert a.id != b.id

        release.set()
        await _wait_status(registry, a.id, JobStatus.succeeded)
        await _wait_status(registry, b.id, JobStatus.succeeded)

    asyncio.run(main())


def test_failed_job_releases_key_slot():
    async def main():
        registry = JobRegistry()

        async def boom():
            raise RuntimeError("nope")

        j = await registry.submit("k", "key", boom, params={})
        await _wait_status(registry, j.id, JobStatus.failed)
        snap = registry.get(j.id)
        assert snap is not None and snap.error is not None and "nope" in snap.error

        async def ok():
            return {"ok": True}

        k = await registry.submit("k", "key", ok, params={})
        assert k.was_already_running is False
        assert k.id != j.id

    asyncio.run(main())


def test_cancel_running_job():
    async def main():
        registry = JobRegistry()
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(5)
            return {"never": True}

        j = await registry.submit("k", "key", slow, params={})
        await asyncio.wait_for(started.wait(), timeout=1.0)
        snap = await registry.cancel(j.id)
        assert snap is not None
        await _wait_status(registry, j.id, JobStatus.cancelled)

    asyncio.run(main())


def test_list_filter_by_status():
    async def main():
        registry = JobRegistry()

        async def ok():
            return {"ok": True}

        async def boom():
            raise ValueError("x")

        a = await registry.submit("k", "a", ok, params={})
        b = await registry.submit("k", "b", boom, params={})

        await _wait_status(registry, a.id, JobStatus.succeeded)
        await _wait_status(registry, b.id, JobStatus.failed)

        succeeded = registry.list(status=JobStatus.succeeded)
        failed = registry.list(status=JobStatus.failed)
        assert {j.id for j in succeeded} == {a.id}
        assert {j.id for j in failed} == {b.id}

    asyncio.run(main())


def test_get_unknown_id_returns_none():
    async def main():
        registry = JobRegistry()
        assert registry.get("nope") is None
        assert (await registry.cancel("nope")) is None

    asyncio.run(main())


def test_submit_with_explicit_job_id_uses_it():
    async def main():
        registry = JobRegistry()

        async def factory():
            return {"ok": True}

        job = await registry.submit("k", "key", factory, params={}, job_id="my-fixed-id")
        assert job.id == "my-fixed-id"
        await _wait_status(registry, "my-fixed-id", JobStatus.succeeded)

    asyncio.run(main())


def test_set_progress_updates_job_snapshot():
    async def main():
        registry = JobRegistry()
        gate = asyncio.Event()

        async def factory():
            await gate.wait()
            return {"ok": True}

        job = await registry.submit("k", "key", factory, params={}, job_id="job-1")
        registry.set_progress("job-1", {"processed": 5, "total": 10})
        snap = registry.get("job-1")
        assert snap is not None
        assert snap.progress == {"processed": 5, "total": 10}

        gate.set()
        await _wait_status(registry, job.id, JobStatus.succeeded)

    asyncio.run(main())


def test_set_progress_on_unknown_job_is_a_noop():
    registry = JobRegistry()
    registry.set_progress("nope", {"processed": 1})  # must not raise


def test_cancel_signals_cancel_token_for_thread_backed_work():
    """The realistic shape: work runs synchronously in a worker thread via
    ``asyncio.to_thread`` (as every ingest job does), not as a plain coroutine.

    ``task.cancel()`` alone cancels the *wrapping* coroutine almost
    immediately — the job's status flips to ``cancelled`` right away — but
    does nothing to the OS thread still executing ``slow_sync`` in the
    background; that thread only stops promptly because it's checking
    ``cancel_token`` itself. This is exactly why ``JobRegistry.cancel``
    signals both.
    """

    async def main():
        registry = JobRegistry()
        started = threading.Event()
        finished = threading.Event()
        token = CancelToken()

        def slow_sync():
            started.set()
            for _ in range(200):
                if token.is_set():
                    finished.set()
                    return {"stopped_early": True}
                time.sleep(0.01)
            finished.set()
            return {"never": True}

        async def factory():
            return await asyncio.to_thread(slow_sync)

        j = await registry.submit("k", "key", factory, params={}, cancel_token=token)

        loop = asyncio.get_event_loop()
        deadline = loop.time() + 1.0
        while not started.is_set() and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert started.is_set()

        await registry.cancel(j.id)
        # The wrapping asyncio.Task is cancelled promptly...
        await _wait_status(registry, j.id, JobStatus.cancelled)
        # ...but the still-running background thread only stops because the
        # cancel_token was set, not because of task.cancel() itself.
        assert token.is_set()

        deadline = loop.time() + 1.0
        while not finished.is_set() and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert finished.is_set()

    asyncio.run(main())


def test_to_dict_serializes_job():
    async def main():
        registry = JobRegistry()

        async def factory():
            return {"indexed": 7}

        job = await registry.submit("create_index", "create_index", factory, params={"recreate": False})
        await _wait_status(registry, job.id, JobStatus.succeeded)
        snap = registry.get(job.id)
        assert snap is not None
        d = snap.to_dict()
        assert d["id"] == job.id
        assert d["kind"] == "create_index"
        assert d["status"] == "succeeded"
        assert d["params"] == {"recreate": False}
        assert d["result"] == {"indexed": 7}
        assert d["created_at"] is not None
        assert d["finished_at"] is not None

    asyncio.run(main())
