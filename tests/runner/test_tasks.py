"""Standing tasks: a schedule somebody created, fired by the clock, answered
as an ordinary turn.

The claim under test is mostly a composition claim. The clock's idempotency,
the queue's retries and the inbox's dedupe all already worked; what is new is
that they compose into *one* answer per fire time, and that a person cannot
create a schedule that posts somewhere they were not already talking.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from kasa.config import Config, TaskSettings
from kasa.core.events import InboundEvent
from kasa.runner.cron import Cron
from kasa.runner.jobs import default_specs
from kasa.runner.scheduler import Job, JobSpec, Scheduler
from kasa.runner.tasks import (
    ACTIVE,
    DONE,
    PAUSED,
    TASK_KIND,
    Task,
    TaskError,
    Tasks,
    occurrence_id,
    render_fires,
    task_handler,
)
from kasa.store import Store

NOW = datetime(2026, 9, 5, 10, 30, tzinfo=UTC)

#: Every weekday at nine. The schedule the whole feature exists for.
WEEKDAY_NINE = "0 9 * * 1-5"


async def a_task(store: Store, **overrides: Any) -> Task:
    fields: dict[str, Any] = {
        "owner": "U01",
        "surface": "slack",
        "session_id": "slack:T01:C0123:1756890000.123",
        "channel": "C0123",
        "reply_to": "1756890000.123",
        "scope": "workspace",
        "prompt": "search for what happened in AI overnight and give me the five that matter",
        "cron": WEEKDAY_NINE,
        "timezone": "Asia/Seoul",
        "now": NOW,
    }
    return await Tasks(store).create(**(fields | overrides))


async def jobs(store: Store) -> list[dict[str, Any]]:
    return await store.raw("SELECT * FROM jobs ORDER BY run_after, id")


async def events(store: Store) -> list[InboundEvent]:
    rows = await store.raw("SELECT payload FROM inbox ORDER BY id")
    return [InboundEvent.from_json(row["payload"]) for row in rows]


def a_job(task: Task, fire_at: datetime, attempts: int = 1) -> Job:
    return Job(
        id=occurrence_id(task.id, fire_at),
        kind=TASK_KIND,
        payload={"task_id": task.id},
        attempts=attempts,
    )


# -- what a task is allowed to be --------------------------------------------


async def test_a_schedule_that_fires_too_often_is_refused_with_the_floor_named(
    store: Store,
) -> None:
    """Every fire is a full turn — retrieval, a frontier model, whatever tools
    it reaches for — so `* * * * *` is a bill rather than a schedule. The
    message has to name the floor, because the model that wrote the expression
    is what has to correct it."""
    with pytest.raises(TaskError, match="fires every 1 minute"):
        await a_task(store, cron="* * * * *")

    with pytest.raises(TaskError, match=r"floor is 15"):
        await a_task(store, cron="*/5 * * * *")


async def test_the_floor_is_measured_on_the_gap_not_on_how_it_is_written(store: Store) -> None:
    """`*/15` and `0,15,30,45` are the same schedule. Judging the text rather
    than the schedule would let one through and refuse the other."""
    assert await a_task(store, cron="*/15 * * * *")
    assert await a_task(store, cron="0,15,30,45 * * * *")

    with pytest.raises(TaskError):
        await a_task(store, cron="0,5,15,30 * * * *")


async def test_a_zone_this_machine_does_not_know_is_refused_at_creation(store: Store) -> None:
    with pytest.raises(TaskError, match="not a time zone"):
        await a_task(store, timezone="KST")


async def test_a_task_with_nothing_to_do_is_not_a_task(store: Store) -> None:
    with pytest.raises(TaskError, match="prompt is empty"):
        await a_task(store, prompt="   ")


async def test_a_person_may_only_hold_so_many(store: Store) -> None:
    """The cap is per owner and counts what will still fire. Without it a loop
    of "make me a schedule" fills the table, and every row in it is a recurring
    model call."""
    tasks = Tasks(store, TaskSettings(max_per_owner=2))
    for _ in range(2):
        await tasks.create(
            owner="U01", surface="slack", session_id="s", prompt="go", cron=WEEKDAY_NINE, now=NOW
        )

    with pytest.raises(TaskError, match="which is the limit"):
        await tasks.create(
            owner="U01", surface="slack", session_id="s", prompt="go", cron=WEEKDAY_NINE, now=NOW
        )

    # Somebody else's allowance is their own.
    assert await tasks.create(
        owner="U02", surface="slack", session_id="s", prompt="go", cron=WEEKDAY_NINE, now=NOW
    )


async def test_a_fired_one_shot_stops_counting_against_its_owner(store: Store) -> None:
    """`done` is history. Counting it forever would mean somebody who used the
    feature correctly runs out of it."""
    tasks = Tasks(store, TaskSettings(max_per_owner=1))
    task = await tasks.create(
        owner="U01",
        surface="slack",
        session_id="s",
        prompt="go",
        cron=WEEKDAY_NINE,
        fire_once=True,
        now=NOW,
    )
    await tasks.finish(task.id)

    assert await tasks.create(
        owner="U01", surface="slack", session_id="s", prompt="go", cron=WEEKDAY_NINE, now=NOW
    )


# -- the confirmation a person can actually check ----------------------------


async def test_the_next_fires_come_back_in_the_task_s_own_zone(store: Store) -> None:
    """Nobody can check `0 9 * * 1-5`. Everybody can check "Mon 07 Sep 09:00
    Asia/Seoul", which is what the tool in #180 reads back before treating a
    task as set up.

    `NOW` is a Saturday evening in Seoul, so the weekend is the part of this
    worth reading: the next fire is Monday, and the hour is nine on Seoul's
    clock rather than nine on the server's.
    """
    task = await a_task(store)

    fires = task.next_fires(3, now=NOW)

    assert render_fires(fires, task.timezone) == [
        "Mon 07 Sep 09:00 Asia/Seoul",
        "Tue 08 Sep 09:00 Asia/Seoul",
        "Wed 09 Sep 09:00 Asia/Seoul",
    ]
    # The instants themselves are UTC, which is what the scheduler deals in.
    assert all(fire.tzinfo is UTC for fire in fires)
    assert fires[0] == datetime(2026, 9, 7, 0, 0, tzinfo=UTC)


# -- the clock ---------------------------------------------------------------


async def test_the_clock_queues_the_next_fire_of_every_active_task(store: Store) -> None:
    task = await a_task(store)
    scheduler = Scheduler(store, default_specs(bare(), store), clocks=[Tasks(store)])

    queued = await scheduler.schedule_due(now=NOW)

    fire_at = task.next_fires(1, now=NOW)[0]
    assert occurrence_id(task.id, fire_at) in queued
    row = next(job for job in await jobs(store) if job["kind"] == TASK_KIND)
    assert json.loads(row["payload"]) == {"task_id": task.id}


async def test_two_schedulers_on_the_same_tick_produce_one_occurrence(store: Store) -> None:
    """The whole idempotency story, inherited rather than rebuilt: the job id
    is the fire time, so the second insert is a no-op."""
    await a_task(store)
    specs = default_specs(bare(), store)
    one = Scheduler(store, specs, clocks=[Tasks(store)])
    two = Scheduler(store, specs, clocks=[Tasks(store)])

    first = await one.schedule_due(now=NOW)
    second = await two.schedule_due(now=NOW)

    assert first and not second
    assert len([job for job in await jobs(store) if job["kind"] == TASK_KIND]) == 1


async def test_a_daemon_that_was_down_runs_the_one_it_queued_and_not_the_ten_it_missed(
    store: Store,
) -> None:
    """A stampede on restart would be ten frontier-model turns in one minute,
    in a thread nobody is reading. The occurrence already queued still runs,
    late; the ones never queued never happened."""
    await a_task(store)
    scheduler = Scheduler(store, default_specs(bare(), store), clocks=[Tasks(store)])
    await scheduler.schedule_due(now=NOW)

    # A fortnight of nine o'clocks passes with nothing running.
    await scheduler.schedule_due(now=NOW + timedelta(days=14))

    rows = [job for job in await jobs(store) if job["kind"] == TASK_KIND]
    assert len(rows) == 2
    assert all(row["state"] == "pending" for row in rows)


async def test_a_task_whose_schedule_stopped_reading_does_not_stop_the_others(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    """A zone is resolved against whatever tz database the machine has, so a
    row written on one host can be unreadable on another. These expressions
    were written by a model reading what somebody typed; one of them being
    wrong must cost only itself."""
    broken = await a_task(store)
    await store.write("UPDATE tasks SET timezone = 'Mars/Olympus' WHERE id = ?", (broken.id,))
    healthy = await a_task(store, cron="0 8 * * *", timezone=None)
    scheduler = Scheduler(store, default_specs(bare(), store), clocks=[Tasks(store)])

    queued = await scheduler.schedule_due(now=NOW)

    assert any(job_id.startswith(f"task:{healthy.id}@") for job_id in queued)
    assert not any(job_id.startswith(f"task:{broken.id}@") for job_id in queued)
    assert "no longer reads" in caplog.text


async def test_an_unreadable_tasks_table_does_not_take_the_compiled_in_jobs_down(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    """`promote` and `reindex` are what hold the system together. A tick where
    somebody's schedule cannot be read is not a tick where those stop."""

    class Broken:
        async def occurrences(self, moment: datetime) -> Sequence[Any]:
            raise RuntimeError("the tasks table is on fire")

    ran: list[Job] = []
    scheduler = Scheduler(
        store,
        [JobSpec(kind="reindex", handler=records(ran), cron=Cron.parse("* * * * *"))],
        clocks=[Broken()],
    )

    assert await scheduler.schedule_due(now=NOW)
    assert "could not read a schedule source" in caplog.text


async def test_a_paused_task_is_not_queued_at_all(store: Store) -> None:
    task = await a_task(store)
    await Tasks(store).pause(task.id)
    scheduler = Scheduler(store, default_specs(bare(), store), clocks=[Tasks(store)])

    assert not [job_id for job_id in await scheduler.schedule_due(now=NOW) if "task:" in job_id]


# -- the run itself ----------------------------------------------------------


async def test_a_fire_becomes_an_ordinary_message_in_the_thread_it_belongs_to(
    store: Store,
) -> None:
    """The handler does not run the agent. It puts an event in the inbox, and
    from there it is a turn like any other — which is why the Slack adapter
    needed nothing: it answers from `channel` and `reply_to` alone."""
    task = await a_task(store)
    fire_at = task.next_fires(1, now=NOW)[0]

    await task_handler(store)(a_job(task, fire_at))

    (event,) = await events(store)
    assert event.text == task.prompt
    assert event.session_id == task.session_id
    assert event.channel == "C0123"
    assert event.reply_to == "1756890000.123"
    assert event.scope == "workspace"
    assert event.author == "U01"
    assert event.source == "slack"
    # Nobody said anything just now, and the turn has to know it.
    assert event.origin == "scheduled"


async def test_a_retried_run_produces_one_answer_rather_than_two(store: Store) -> None:
    """Both queues are at-least-once. They compose into at-most-one-answer
    because the event id *is* the job id: the inbox's UNIQUE constraint is what
    makes the second delivery a no-op."""
    task = await a_task(store)
    fire_at = task.next_fires(1, now=NOW)[0]
    handler = task_handler(store)

    await handler(a_job(task, fire_at, attempts=1))
    await handler(a_job(task, fire_at, attempts=2))

    assert len(await events(store)) == 1


async def test_a_deleted_task_does_not_post_the_run_already_queued_for_it(store: Store) -> None:
    """Deleting a task stops it. An occurrence the clock queued a moment before
    is work that was called off, not work that was lost."""
    task = await a_task(store)
    fire_at = task.next_fires(1, now=NOW)[0]
    await Tasks(store).cancel(task.id)

    await task_handler(store)(a_job(task, fire_at))

    assert not await events(store)


async def test_a_paused_task_does_not_post_the_run_already_queued_for_it(store: Store) -> None:
    task = await a_task(store)
    fire_at = task.next_fires(1, now=NOW)[0]
    await Tasks(store).pause(task.id)

    await task_handler(store)(a_job(task, fire_at))

    assert not await events(store)


async def test_a_one_shot_fires_once_and_ends_done(store: Store) -> None:
    task = await a_task(store, fire_once=True)
    fire_at = task.next_fires(1, now=NOW)[0]
    handler = task_handler(store)

    await handler(a_job(task, fire_at))
    # The clock has no idea it is finished until the row says so, so the second
    # occurrence is the one the state has to stop.
    await handler(a_job(task, task.next_fires(2, now=NOW)[1]))

    assert len(await events(store)) == 1
    after = await Tasks(store).get(task.id)
    assert after is not None and after.state == DONE


async def test_a_run_that_worked_clears_what_went_wrong_before_it(store: Store) -> None:
    task = await a_task(store)
    await store.record_task_failure(task.id, error="RuntimeError: nope")
    fire_at = task.next_fires(1, now=NOW)[0]

    await task_handler(store)(a_job(task, fire_at))

    after = await Tasks(store).get(task.id)
    assert after is not None
    assert after.consecutive_failures == 0
    assert after.last_error is None
    assert after.last_job_id == occurrence_id(task.id, fire_at)


# -- failing forever is worse than stopping ----------------------------------


async def test_a_task_that_keeps_failing_is_paused_and_its_owner_told_once(store: Store) -> None:
    """Told on the run that crosses the threshold, and never again — the same
    call pauses the task, so there is no second crossing to announce."""
    task = await a_task(store)
    told: list[tuple[str, str]] = []

    async def notify(paused: Task, text: str) -> None:
        told.append((paused.id, text))

    handler = task_handler(
        store,
        TaskSettings(disable_after_failures=2),
        inbox=Exploding(),  # type: ignore[arg-type]
        notify=notify,
    )
    fires = task.next_fires(3, now=NOW)

    for fire_at in fires[:2]:
        with pytest.raises(RuntimeError):
            await handler(a_job(task, fire_at))

    after = await Tasks(store).get(task.id)
    assert after is not None and after.state == PAUSED
    assert after.consecutive_failures == 2
    assert [task_id for task_id, _ in told] == [task.id]
    assert "paused a scheduled task" in told[0][1]

    # And once it is paused, the next occurrence is skipped rather than failed
    # again — so nobody is told twice.
    await handler(a_job(task, fires[2]))
    assert len(told) == 1


async def test_the_queue_s_own_retries_are_one_failed_run_not_three(store: Store) -> None:
    """The job retries on the queue's backoff. Counting each of those tries as
    a failed *run* would pause a task after two bad fire times while telling
    its owner it had failed six times."""
    task = await a_task(store)
    handler = task_handler(store, TaskSettings(disable_after_failures=2), inbox=Exploding())  # type: ignore[arg-type]
    fire_at = task.next_fires(1, now=NOW)[0]

    for attempt in (1, 2, 3):
        with pytest.raises(RuntimeError):
            await handler(a_job(task, fire_at, attempts=attempt))

    after = await Tasks(store).get(task.id)
    assert after is not None
    assert after.consecutive_failures == 1
    assert after.state == ACTIVE


async def test_resuming_forgets_the_failures_that_stopped_it(store: Store) -> None:
    """Whoever resumed it is saying the reason it stopped has been dealt with.
    Coming back one failure from paused would be a trap."""
    task = await a_task(store)
    await store.record_task_failure(task.id, error="RuntimeError: nope")
    await Tasks(store).pause(task.id, reason="RuntimeError: nope")

    await Tasks(store).resume(task.id)

    after = await Tasks(store).get(task.id)
    assert after is not None
    assert after.state == ACTIVE
    assert after.consecutive_failures == 0
    assert after.last_error is None


async def test_a_notifier_that_itself_fails_does_not_keep_the_task_running(store: Store) -> None:
    """A task that could not be paused because nobody could be told would keep
    failing, which is the outcome the pause exists to prevent."""
    task = await a_task(store)

    async def notify(paused: Task, text: str) -> None:
        raise RuntimeError("Slack is down")

    handler = task_handler(
        store,
        TaskSettings(disable_after_failures=1),
        inbox=Exploding(),  # type: ignore[arg-type]
        notify=notify,
    )

    with pytest.raises(RuntimeError, match="the inbox is full"):
        await handler(a_job(task, task.next_fires(1, now=NOW)[0]))

    after = await Tasks(store).get(task.id)
    assert after is not None and after.state == PAUSED


# -- the whole path ----------------------------------------------------------


async def test_the_clock_and_the_drainer_together_put_a_turn_in_the_thread(store: Store) -> None:
    """The acceptance criterion, with both halves running: the clock queues the
    occurrence, the drainer leases it, and what comes out the far side is a
    message in the session the task was created in — waiting for a dispatcher
    exactly as one somebody typed would be.

    `now` is a year in the past so the fire time the clock computes is already
    behind us and the row is runnable the moment it is written. The alternative
    is a test that waits until nine in the morning.
    """
    task = await a_task(store)
    scheduler = Scheduler(
        store, default_specs(bare(), store), poll_interval=0.01, clocks=[Tasks(store)]
    )

    await scheduler.schedule_due(now=NOW - timedelta(days=365))
    running = asyncio.create_task(scheduler.run())
    try:
        await until_events(store)
    finally:
        scheduler.stop()
        await running

    (event,) = await events(store)
    assert event.session_id == task.session_id
    assert event.origin == "scheduled"
    assert (await jobs(store))[0]["state"] == "done"


async def until_events(store: Store, *, within: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while loop.time() < deadline:
        if await events(store):
            return
        await asyncio.sleep(0.005)
    raise AssertionError("nothing reached the inbox")


# -- what a task may not do --------------------------------------------------


async def test_a_task_created_in_a_dm_stays_in_the_dm(store: Store) -> None:
    """The destination is copied from the conversation that asked for the task
    and is not settable afterwards (§11.1). `create` has no argument that could
    widen it, and the scope travels onto the event."""
    task = await a_task(store, scope="dm:U01", channel="D0999", reply_to=None)

    await task_handler(store)(a_job(task, task.next_fires(1, now=NOW)[0]))

    (event,) = await events(store)
    assert event.scope == "dm:U01"
    assert event.channel == "D0999"


async def test_listing_is_narrowed_in_the_query_rather_than_afterwards(store: Store) -> None:
    """A tool that read every row and then discarded the ones it should not
    show has already had them (§7.1)."""
    mine = await a_task(store, owner="U01", session_id="C0123")
    await a_task(store, owner="U02", session_id="C0456")

    visible = await Tasks(store).all(owner="U01", session_id="C0123")

    assert [task.id for task in visible] == [mine.id]


# -- helpers -----------------------------------------------------------------


class Exploding:
    """An inbox that cannot take anything, for the failure path."""

    async def enqueue(self, event: InboundEvent) -> None:
        raise RuntimeError("the inbox is full")


def records(into: list[Job]) -> Callable[[Job], Awaitable[None]]:
    async def handler(job: Job) -> None:
        into.append(job)

    return handler


def bare() -> Config:
    """A config with no model, no repo and no Slack.

    `task_run` still registers on it — that is the point of registering it on
    nothing at all — and nothing else that would want a provider does.
    """
    return Config()
