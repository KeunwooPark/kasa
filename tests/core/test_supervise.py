"""The loops nobody awaits, and what happens when one of them raises."""

from __future__ import annotations

import asyncio

import pytest

from kasa.core.supervise import keep_running
from tests.conftest import until


async def stop(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_body_that_raises_does_not_end_the_loop() -> None:
    """The whole point. Without this the first transient stops the clock for
    the life of the process, and nothing says so."""
    calls = 0

    async def body() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("the store was busy")

    task = asyncio.create_task(keep_running(body, every=0.01, name="test loop"))
    await until(lambda: calls >= 5)
    await stop(task)


async def test_a_failure_is_reported_with_the_name_of_the_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A loop that fails silently is the bug. It has to be findable in a log by
    what it does, not by the method that happened to be running."""

    async def body() -> None:
        raise RuntimeError("the store was busy")

    task = asyncio.create_task(keep_running(body, every=0.01, name="scheduler clock"))
    await until(lambda: any("scheduler clock" in record.message for record in caplog.records))
    await stop(task)

    failure = next(r for r in caplog.records if "scheduler clock" in r.message)
    assert failure.levelname == "ERROR"
    assert failure.exc_info is not None  # log.exception, so the traceback is there


async def test_only_the_first_consecutive_failure_has_a_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    calls = 0

    async def body() -> None:
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise RuntimeError("the store was busy")

    task = asyncio.create_task(keep_running(body, every=0.01, name="scheduler clock"))
    await until(lambda: calls >= 4)
    await stop(task)

    failures = [r for r in caplog.records if "will try again" in r.message]
    assert len(failures) == 3
    assert failures[0].exc_info is not None
    assert all(record.exc_info is None for record in failures[1:])
    assert any(
        record.message == "scheduler clock recovered after 3 failed ticks"
        for record in caplog.records
    )


async def test_a_new_run_of_failures_gets_a_new_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    async def body() -> None:
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            raise RuntimeError("the store was busy")

    task = asyncio.create_task(
        keep_running(body, every=0.01, name="scheduler clock", start_now=True)
    )
    await until(lambda: calls >= 4)
    await stop(task)

    failures = [r for r in caplog.records if "will try again" in r.message]
    assert len(failures) == 2
    assert all(record.exc_info is not None for record in failures)


async def test_the_loop_waits_out_the_first_interval_by_default() -> None:
    """A keepalive with nothing in flight and a sweeper with nothing to sweep
    both have no work at startup."""
    calls = 0

    async def body() -> None:
        nonlocal calls
        calls += 1

    task = asyncio.create_task(keep_running(body, every=5.0, name="test loop"))
    await asyncio.sleep(0.05)

    assert calls == 0
    await stop(task)


async def test_start_now_runs_before_the_first_interval() -> None:
    """The clock queues the next occurrence of every recurring job, and must do
    it at startup rather than a tick later."""
    calls = 0

    async def body() -> None:
        nonlocal calls
        calls += 1

    task = asyncio.create_task(keep_running(body, every=5.0, name="test loop", start_now=True))
    await until(lambda: calls == 1)
    await stop(task)


async def test_cancellation_still_stops_it() -> None:
    """`CancelledError` is a `BaseException`, so the `except Exception` that
    keeps the loop alive does not swallow shutdown."""
    entered = asyncio.Event()

    async def body() -> None:
        entered.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(keep_running(body, every=0.01, name="test loop", start_now=True))
    await entered.wait()
    await stop(task)

    assert task.cancelled()
