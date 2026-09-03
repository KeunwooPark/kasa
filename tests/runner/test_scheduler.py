"""Jobs as rows: scheduled, leased, retried, and replayed after a crash."""

from __future__ import annotations

import asyncio
import signal
import sys
import textwrap
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from kasa.core.backoff import Backoff
from kasa.errors import StoreError
from kasa.runner.cron import HOURLY, NIGHTLY, Cron
from kasa.runner.scheduler import (
    Job,
    JobSpec,
    Scheduler,
    UnknownJob,
    scheduled_id,
)
from kasa.store import Store
from tests.conftest import until

NOW = datetime(2026, 9, 3, 10, 30, tzinfo=UTC)


def records(into: list[Job]) -> Callable[[Job], Any]:
    async def handler(job: Job) -> None:
        into.append(job)

    return handler


def explodes(message: str = "the remote was busy") -> Callable[[Job], Any]:
    async def handler(job: Job) -> None:
        raise RuntimeError(message)

    return handler


async def states(store: Store) -> dict[str, str]:
    return {row["id"]: row["state"] for row in await store.raw("SELECT id, state FROM jobs")}


# -- the clock keeps running -------------------------------------------------


async def test_a_slow_job_does_not_pay_for_the_leases_it_gave_itself(store: Store) -> None:
    """`attempts` counts leases rather than failures deliberately: work that
    kills the process answering it leaves no failure behind to count, and
    counting only failures is how such work loops forever.

    A lease the drainer hands to *itself* while it is still running the work is
    not a killed process. #114 stopped it running that work twice; it kept
    taking the lease, and the lease is what `attempts` counts — so the retry
    budget went to the one job guaranteed not to need it that way: the slow one.
    """
    gate = asyncio.Event()
    runs = 0

    async def slow(job: Job) -> None:
        nonlocal runs
        runs += 1
        await gate.wait()

    scheduler = Scheduler(
        store,
        [JobSpec(kind="reindex", handler=slow)],
        concurrency=2,
        poll_interval=0.01,
        lease_ttl=0.0,  # every lease is expired the moment it is taken
        backoff=Backoff(max_attempts=3, base=0.0, cap=0.0),
    )
    await scheduler.queue.enqueue("reindex")

    task = asyncio.create_task(scheduler.run())
    try:
        await until(lambda: runs >= 1)
        await asyncio.sleep(0.2)  # twenty polls, each of which re-leased the row
        attempts = [row["attempts"] for row in await store.raw("SELECT attempts FROM jobs")]
    finally:
        gate.set()
        scheduler.stop()
        await asyncio.wait_for(task, timeout=10.0)

    assert runs == 1, f"the work ran {runs} time(s)"
    assert attempts == [1], f"one run, and the row was charged {attempts[0]} attempt(s)"


async def test_a_slow_job_still_gets_every_retry_it_was_promised(store: Store) -> None:
    """The same accounting, stated as the behaviour it buys.

    A job that outlives two lease TTLs and then fails used to reach its dead
    letter having really run twice against a budget of three — the leases it
    was handed while running spent the other one. Slowness is not a failure,
    and the job most likely to outlive its TTL is the one most likely to want
    the retry.
    """
    runs = 0

    async def slow_then_broken(job: Job) -> None:
        nonlocal runs
        runs += 1
        await asyncio.sleep(0.05)  # several polls' worth, each of them a re-lease
        raise RuntimeError("the remote was busy")

    scheduler = Scheduler(
        store,
        [JobSpec(kind="reindex", handler=slow_then_broken)],
        concurrency=2,
        poll_interval=0.01,
        lease_ttl=0.0,  # every lease is expired the moment it is taken
        backoff=Backoff(max_attempts=3, base=0.0, cap=0.0),
    )
    await scheduler.queue.enqueue("reindex")

    task = asyncio.create_task(scheduler.run())
    try:
        for _ in range(1000):
            rows = await store.raw("SELECT state, attempts FROM jobs")
            if rows[0]["state"] == "failed":
                break
            await asyncio.sleep(0.01)
    finally:
        scheduler.stop()
        await asyncio.wait_for(task, timeout=10.0)

    rows = await store.raw("SELECT state, attempts FROM jobs")
    assert rows[0]["state"] == "failed", "it should have run out of retries by now"
    assert runs == 3, f"a budget of three, spent on {runs} real run(s)"
    assert rows[0]["attempts"] == 3, f"three runs recorded as {rows[0]['attempts']} attempt(s)"


async def test_a_spec_that_cannot_be_scheduled_does_not_starve_the_others(store: Store) -> None:
    """February 30th parses and never arrives. `_specs` is iterated in
    registration order, so one bad expression used to take every spec behind it
    down with it — on this tick and on every tick after."""
    ran: list[Job] = []
    sched = Scheduler(
        store,
        [
            JobSpec(kind="broken", handler=records(ran), cron=Cron.parse("0 0 30 2 *")),
            JobSpec(kind="healthy", handler=records(ran), cron=Cron.parse(HOURLY)),
        ],
    )

    queued = await sched.schedule_due(now=NOW)

    assert [job_id.split("@")[0] for job_id in queued] == ["healthy"]


async def test_a_broken_spec_reports_one_traceback_until_it_recovers(
    store: Store, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level("INFO")
    scheduler = Scheduler(
        store,
        [JobSpec(kind="fragile", handler=records([]), cron=Cron.parse(HOURLY))],
    )
    cron = scheduler._specs["fragile"].cron
    assert cron is not None
    calls = 0
    original = cron.next_after

    def intermittently_broken(self: Cron, moment: datetime) -> datetime:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError("clock failed")
        return original(moment)

    monkeypatch.setattr(Cron, "next_after", intermittently_broken)

    await scheduler.schedule_due(now=NOW)
    await scheduler.schedule_due(now=NOW)
    await scheduler.schedule_due(now=NOW)

    failures = [r for r in caplog.records if r.message.startswith("could not schedule fragile")]
    assert len(failures) == 2
    assert failures[0].exc_info is not None
    assert failures[1].exc_info is None
    assert any("recovered after 2 failed ticks" in r.message for r in caplog.records)


async def test_a_tick_that_raises_does_not_stop_the_clock(store: Store) -> None:
    """`schedule_due` reaches the store, so a locked database is enough. The
    clock used to end there, silently, and never queue another occurrence."""
    ticks = 0
    sched = Scheduler(store, [], tick_interval=0.01)

    async def failing_tick(*, now: datetime | None = None) -> list[str]:
        nonlocal ticks
        ticks += 1
        raise StoreError("database is locked")

    sched.schedule_due = failing_tick  # type: ignore[method-assign]
    task = asyncio.create_task(sched.run())
    await until(lambda: ticks >= 5)
    sched.stop()
    await asyncio.wait_for(task, timeout=5.0)


# -- running -----------------------------------------------------------------


async def test_a_one_shot_runs_and_the_row_says_so(store: Store) -> None:
    ran: list[Job] = []
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records(ran))])

    row = await scheduler.run_now("reindex")

    assert [job.kind for job in ran] == ["reindex"]
    assert (row["state"], row["attempts"]) == ("done", 1)
    assert row["finished_at"] is not None


async def test_a_payload_reaches_the_handler(store: Store) -> None:
    ran: list[Job] = []
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records(ran))])

    row = await scheduler.run_now("reindex", {"full": True})

    assert [job.payload for job in ran] == [{"full": True}]
    assert row["state"] == "done"


async def test_a_one_shot_runs_even_with_older_rows_of_its_kind_ahead_of_it(
    store: Store,
) -> None:
    """`lease` is ordered by `run_after` and bounded by concurrency, so a row
    queued just now is last in line. Reporting on a row nobody ran is how
    `kasa job run` came to print `reindex pending: None` and exit 1."""
    for n in range(5):
        await store.enqueue_job(
            job_id=f"older-{n}",
            kind="reindex",
            payload=None,
            run_after="2020-01-01T00:00:00.000+00:00",
        )
    ran: list[Job] = []
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records(ran))], concurrency=2)

    row = await scheduler.run_now("reindex")

    assert row["state"] == "done", "the job the user asked for is the one reported on"
    assert row["id"] not in {f"older-{n}" for n in range(5)}


async def test_a_one_shot_runs_one_job(store: Store) -> None:
    """ "Run one job now, in this process" is what the command says it does.

    #117 reached this row by draining until it had been attempted, which runs
    everything of the same kind that is already due on the way past. Each of
    those calls a frontier model, and not one of them is what was asked for.
    """
    for n in range(20):
        await store.enqueue_job(
            job_id=f"older-{n}",
            kind="reindex",
            payload=None,
            run_after="2020-01-01T00:00:00.000+00:00",
        )
    ran: list[Job] = []
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records(ran))], concurrency=2)

    row = await scheduler.run_now("reindex")

    assert row["state"] == "done"
    assert [job.id for job in ran] == [row["id"]], f"one job asked for, {len(ran)} run"


async def test_the_backlog_it_no_longer_runs_is_left_exactly_as_it_was(store: Store) -> None:
    """Not running them means not touching them: still pending, still unspent,
    and still there for the daemon whose job they are."""
    for n in range(5):
        await store.enqueue_job(
            job_id=f"older-{n}",
            kind="reindex",
            payload=None,
            run_after="2020-01-01T00:00:00.000+00:00",
        )
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records([]))], concurrency=2)

    await scheduler.run_now("reindex")

    older = await store.raw("SELECT state, attempts FROM jobs WHERE id LIKE 'older-%'")
    assert [row["state"] for row in older] == ["pending"] * 5
    assert [row["attempts"] for row in older] == [0] * 5, "a row not run has not been tried"


async def test_running_a_job_by_hand_leaves_nothing_queued_behind(store: Store) -> None:
    """Every call inserts a row. One that is not drained is one that stays,
    so a command retried four times used to grow the table by four.

    Stated over the rows this command inserted, which is what #117 was about.
    The five seeded ahead of them stay pending on purpose now (#127): they
    belong to the daemon, and running them was never what was asked for.
    """
    for n in range(5):
        await store.enqueue_job(
            job_id=f"older-{n}",
            kind="reindex",
            payload=None,
            run_after="2020-01-01T00:00:00.000+00:00",
        )
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records([]))], concurrency=2)

    inserted = [(await scheduler.run_now("reindex"))["id"] for _ in range(4)]

    ran = await states(store)
    assert len(ran) == 9, "five already due, one per call"
    assert [ran[job_id] for job_id in inserted] == ["done"] * 4, "not one left queued"


async def test_a_job_it_cannot_reach_is_reported_rather_than_waited_for(store: Store) -> None:
    """A row held by another process is not this one's to run. Reporting what
    is there beats looping until the lease expires."""
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records([]))])

    async def nothing_runnable(
        *, limit: int = 1, exclude: Sequence[Any] = (), only: Sequence[Any] = ()
    ) -> list[Job]:
        return []

    scheduler.queue.lease = nothing_runnable  # type: ignore[method-assign]

    row = await asyncio.wait_for(scheduler.run_now("reindex"), timeout=5.0)

    assert (row["state"], row["attempts"], row["last_error"]) == ("pending", 0, None)


async def test_a_job_nobody_can_run_is_refused_at_the_edge(store: Store) -> None:
    """Better here, where a person typed the name, than as a dead letter."""
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records([]))])

    with pytest.raises(UnknownJob, match="registered: reindex"):
        await scheduler.trigger("promote")


async def test_a_worker_leases_only_the_kinds_it_knows(store: Store) -> None:
    """This is what an out-of-process worker is: another drainer over the same
    table, registered for a different subset."""
    await store.enqueue_job(
        job_id="j1", kind="promote", payload=None, run_after="2020-01-01T00:00:00.000+00:00"
    )
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records([]))])

    assert await scheduler.queue.lease(limit=5) == []
    assert await states(store) == {"j1": "pending"}


# -- failure -----------------------------------------------------------------


async def test_a_failing_job_retries_and_then_dead_letters(store: Store) -> None:
    scheduler = Scheduler(
        store,
        [JobSpec(kind="reindex", handler=explodes())],
        backoff=Backoff(max_attempts=2, base=0.0, cap=0.0),
    )

    first = await scheduler.run_now("reindex")
    assert first["state"] == "pending", "a first failure is a retry, not a verdict"

    await scheduler.queue.lease(limit=1)
    await scheduler.queue.fail(Job(id=first["id"], kind="reindex", payload={}, attempts=2), "again")

    assert (await states(store))[first["id"]] == "failed"


async def test_a_retry_waits_out_its_backoff(store: Store) -> None:
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=explodes())])

    await scheduler.run_now("reindex")

    assert await scheduler.queue.lease(limit=1) == [], "it is pending, but not yet due"


async def test_a_dead_letter_can_be_put_back(store: Store) -> None:
    scheduler = Scheduler(
        store,
        [JobSpec(kind="reindex", handler=explodes())],
        backoff=Backoff(max_attempts=1, base=0.0, cap=0.0),
    )
    await scheduler.run_now("reindex")

    assert await store.revive_failed_jobs() == 1
    assert [job.attempts for job in await scheduler.queue.lease(limit=1)] == [1]


# -- the clock ---------------------------------------------------------------


async def test_the_next_occurrence_is_queued_for_when_it_fires(store: Store) -> None:
    scheduler = Scheduler(store, [JobSpec("reflect", records([]), cron=Cron.parse(NIGHTLY))])

    assert await scheduler.schedule_due(now=NOW) == ["reflect@2026-09-04T03:00+00:00"]
    assert await scheduler.queue.lease(limit=1) == [], "not due for another sixteen hours"


async def test_queueing_the_same_occurrence_twice_queues_it_once(store: Store) -> None:
    """The clock ticks far more often than a job fires, and two schedulers may
    tick at the same moment. The id is derived from the fire time for that."""
    scheduler = Scheduler(store, [JobSpec("reflect", records([]), cron=Cron.parse(NIGHTLY))])

    await scheduler.schedule_due(now=NOW)
    await scheduler.schedule_due(now=NOW + timedelta(minutes=1))

    assert len(await store.raw("SELECT id FROM jobs")) == 1


async def test_a_job_with_no_cron_is_never_queued_by_the_clock(store: Store) -> None:
    scheduler = Scheduler(store, [JobSpec(kind="reindex", handler=records([]))])

    assert await scheduler.schedule_due(now=NOW) == []


async def test_an_occurrence_queued_before_a_restart_still_runs_late(store: Store) -> None:
    """The clock looks forward only — it does not backfill the hours a daemon
    was down. What it had already queued is a row, so that one survives."""
    ran: list[Job] = []
    scheduler = Scheduler(
        store, [JobSpec("reflect", records(ran), cron=Cron.parse(HOURLY))], poll_interval=0.01
    )
    await scheduler.schedule_due(now=NOW - timedelta(days=1))

    running = asyncio.create_task(scheduler.run())
    try:
        await until(lambda: len(ran) == 1)
    finally:
        scheduler.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert ran[0].id == scheduled_id("reflect", datetime(2026, 9, 2, 11, 0, tzinfo=UTC))


async def test_the_running_scheduler_fills_the_table_on_its_own(store: Store) -> None:
    scheduler = Scheduler(
        store,
        [JobSpec("reflect", records([]), cron=Cron.parse(NIGHTLY))],
        poll_interval=0.01,
        tick_interval=0.01,
    )

    running = asyncio.create_task(scheduler.run())
    try:
        await asyncio.sleep(0.1)  # several ticks
    finally:
        scheduler.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert [row["state"] for row in await store.raw("SELECT state FROM jobs")] == ["pending"]


# -- the acceptance criterion ------------------------------------------------


CRASH_MID_JOB = """
import asyncio, os, signal, sys

from kasa.runner.scheduler import JobQueue
from kasa.store import Store


async def main() -> None:
    db, marker = sys.argv[1], sys.argv[2]
    store = await Store.open(db)
    queue = JobQueue(store, lease_ttl=3600)
    queue.accept("reindex")
    leased = await queue.lease(limit=1)
    assert len(leased) == 1, leased
    # Proof the kill below lands mid-job rather than before it.
    open(marker, "w").write(leased[0].id)
    os.kill(os.getpid(), signal.SIGKILL)


asyncio.run(main())
"""


async def test_killing_the_daemon_mid_job_leaves_exactly_one_completed_run(
    tmp_path: Path,
) -> None:
    """The acceptance criterion of #26, run for real."""
    db = tmp_path / "kasa.db"
    async with await Store.open(db) as setup:
        await Scheduler(setup, [JobSpec(kind="reindex", handler=records([]))]).trigger("reindex")

    script = tmp_path / "crash.py"
    script.write_text(textwrap.dedent(CRASH_MID_JOB))
    marker = tmp_path / "leased"
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(script), str(db), str(marker)
    )
    assert await process.wait() == -signal.SIGKILL
    job_id = marker.read_text()

    ran: list[Job] = []
    async with await Store.open(db) as store:
        scheduler = Scheduler(
            store, [JobSpec(kind="reindex", handler=records(ran))], poll_interval=0.01
        )
        running = asyncio.create_task(scheduler.run())
        try:
            await until(lambda: len(ran) == 1)
            await asyncio.sleep(0.1)  # a second run would land in this window
        finally:
            scheduler.stop()
            await asyncio.wait_for(running, timeout=10.0)

        assert [job.id for job in ran] == [job_id]
        assert await states(store) == {job_id: "done"}
        assert (await store.raw("SELECT attempts FROM jobs"))[0]["attempts"] == 2
