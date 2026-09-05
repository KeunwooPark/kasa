"""Standing tasks: a schedule somebody created, delivered as an ordinary turn.

Every recurring thing Kasa did before this was compiled in. `default_specs`
(`kasa/runner/jobs.py`) is a list written by whoever built the binary, and the
clock iterates exactly that list — so there was no way for a person to say
"every weekday at nine, tell me what happened in AI overnight" and have it
happen. A standing task is that: a row someone created, fired by the daemon,
answered in the thread they created it in.

The interesting claim in this module is how little it adds.

* The clock already queues occurrences under an id derived from their fire
  time (`scheduled_id`), which is what makes it idempotent, safe to run twice,
  and free of a stampede after downtime. A task's occurrences use the same
  trick, so none of that is re-reasoned here.
* The handler does **not** run the agent. It builds an `InboundEvent` and puts
  it in the inbox. From there it is an ordinary turn — dispatcher, session
  actor, context packing, retrieval, tools, and the surface's own reply path —
  and no adapter needs to learn anything: Slack answers from `event.channel`
  and `event.reply_to` alone.
* `UNIQUE (source, external_id)` on the inbox is what stops a *retried job*
  producing a second answer. The job's own id is the event's `external_id`, so
  the two queues' at-least-once semantics compose into at-most-one-answer
  without a new mechanism.

The destination is never a parameter. `session_id`, `channel`, `reply_to` and
`scope` are copied from the conversation that created the task and are fixed
there (§11.1): a task inherits the visibility of the thread it was asked for
in, so nothing arriving in a DM can arrange to be said in a public channel.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self
from zoneinfo import ZoneInfo

from ulid import ULID

from kasa.config import TaskSettings
from kasa.core.events import InboundEvent
from kasa.core.inbox import Inbox
from kasa.errors import KasaError
from kasa.runner.cron import Cron, CronError
from kasa.runner.scheduler import Job, JobHandler, Occurrence
from kasa.store import Store

log = logging.getLogger(__name__)

#: The one job kind standing tasks run under. Registered with no cron of its
#: own: the occurrences come from the tasks table, and this exists so the queue
#: accepts the kind and the drainer has a handler for it.
TASK_KIND = "task_run"

ACTIVE, PAUSED, DONE = "active", "paused", "done"

#: How a paused task's owner is told. None where there is nothing to tell them
#: with — a build with no Slack, or a task created from a terminal.
TaskNotifier = Callable[["Task", str], Awaitable[None]]


class TaskError(KasaError):
    """A schedule Kasa will not create, or one it can no longer read."""


def occurrence_id(task_id: str, fire_at: datetime) -> str:
    """The job id of one firing.

    Derived from the fire time for the same reason `scheduled_id` is: two
    schedulers ticking on the same minute write the same row, and only the
    first one counts. Prefixed with `task:` so a glance at the jobs table says
    which task a row belongs to.
    """
    return f"task:{task_id}@{fire_at.isoformat(timespec='minutes')}"


@dataclass(frozen=True, slots=True)
class Task:
    """One standing schedule, as it is stored."""

    id: str
    owner: str
    surface: str
    session_id: str
    channel: str | None
    reply_to: str | None
    scope: str
    prompt: str
    cron: str
    timezone: str | None
    state: str
    fire_once: bool
    created_at: str
    last_run_at: str | None = None
    last_job_id: str | None = None
    last_error: str | None = None
    consecutive_failures: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        return cls(
            id=str(row["id"]),
            owner=str(row["owner"]),
            surface=str(row["surface"]),
            session_id=str(row["session_id"]),
            channel=row["channel"],
            reply_to=row["reply_to"],
            scope=str(row["scope"]),
            prompt=str(row["prompt"]),
            cron=str(row["cron"]),
            timezone=row["timezone"],
            state=str(row["state"]),
            fire_once=bool(row["fire_once"]),
            created_at=str(row["created_at"]),
            last_run_at=row["last_run_at"],
            last_job_id=row["last_job_id"],
            last_error=row["last_error"],
            consecutive_failures=int(row["consecutive_failures"]),
        )

    def schedule(self) -> Cron:
        """The parsed expression. Raises if the row no longer reads.

        It can stop reading: the zone is resolved against whatever tz database
        the machine has, and a task created on one host and run on another can
        name a zone the second one has never heard of.
        """
        return Cron.parse(self.cron, tz=self.timezone)

    def next_fires(self, count: int = 3, *, now: datetime | None = None) -> list[datetime]:
        """The next `count` instants this fires on, earliest first.

        What confirmation is built out of. Nobody can check `0 9 * * 1-5`, and
        everybody can check "Mon 8 Sep 09:00, Tue 9 Sep 09:00".
        """
        schedule = self.schedule()
        moment = now or datetime.now(UTC)
        fires = []
        for _ in range(count):
            moment = schedule.next_after(moment)
            fires.append(moment)
        return fires

    @property
    def label(self) -> str:
        """How to name this schedule in a listing or a log line.

        Formatted rather than parsed. `Cron.label` would say the same thing,
        but it can only say it about an expression that still reads — and the
        one listing that has to name a task whose zone this machine has never
        heard of is the listing where that has gone wrong.
        """
        return self.cron if self.timezone is None else f"{self.cron} ({self.timezone})"


class Tasks:
    """The tasks table, and every judgement about what may go in it.

    The store holds rows and counts them; what a schedule is *allowed* to be
    lives here, in one place, so the CLI and the `schedule_*` tools cannot
    disagree about it.
    """

    def __init__(self, store: Store, settings: TaskSettings | None = None) -> None:
        self._store = store
        self._settings = settings or TaskSettings()

    @property
    def settings(self) -> TaskSettings:
        return self._settings

    async def create(
        self,
        *,
        owner: str,
        surface: str,
        session_id: str,
        prompt: str,
        cron: str,
        timezone: str | None = None,
        channel: str | None = None,
        reply_to: str | None = None,
        scope: str = "workspace",
        fire_once: bool = False,
        now: datetime | None = None,
    ) -> Task:
        """Create a schedule, having checked it is one Kasa will honour."""
        if not prompt.strip():
            raise TaskError("a task needs something to do; the prompt is empty")
        schedule = self._validate(cron, timezone, now=now)
        if (held := await self._store.count_owner_tasks(owner)) >= self._settings.max_per_owner:
            raise TaskError(
                f"{owner} already has {held} schedule(s), which is the limit "
                f"({self._settings.max_per_owner}). Cancel one first."
            )
        task_id = str(ULID())
        await self._store.create_task(
            task_id=task_id,
            owner=owner,
            surface=surface,
            session_id=session_id,
            channel=channel,
            reply_to=reply_to,
            scope=scope,
            prompt=prompt.strip(),
            cron=schedule.expression,
            timezone=timezone,
            fire_once=fire_once,
        )
        created = await self.get(task_id)
        if created is None:  # pragma: no cover - the row was just written
            raise TaskError(f"task {task_id} vanished between writing and reading it")
        return created

    def _validate(self, cron: str, timezone: str | None, *, now: datetime | None) -> Cron:
        """Parse the expression, and refuse one that fires too often.

        The floor is measured on the gap the expression actually produces
        rather than on how it is written, so `*/15` and `0,15,30,45` are the
        same schedule and are judged the same way.
        """
        try:
            schedule = Cron.parse(cron, tz=timezone)
        except CronError as exc:
            raise TaskError(str(exc)) from exc
        moment = now or datetime.now(UTC)
        try:
            first = schedule.next_after(moment)
            gap = (schedule.next_after(first) - first).total_seconds() / 60
        except CronError as exc:
            raise TaskError(str(exc)) from exc
        if gap < self._settings.min_interval_minutes:
            raise TaskError(
                f"{schedule.label} fires every {gap:.0f} minute(s), and the floor is "
                f"{self._settings.min_interval_minutes}. Every fire is a full turn."
            )
        return schedule

    async def get(self, task_id: str) -> Task | None:
        row = await self._store.get_task(task_id)
        return Task.from_row(row) if row else None

    async def all(
        self, *, state: str | None = None, owner: str | None = None, session_id: str | None = None
    ) -> list[Task]:
        """Every task matching the narrowing, oldest first.

        Not named `list`: a method by that name shadows the builtin inside the
        class body, and every `list[Task]` annotation below it stops meaning a
        list.
        """
        return [
            Task.from_row(row)
            for row in await self._store.list_tasks(state=state, owner=owner, session_id=session_id)
        ]

    async def cancel(self, task_id: str) -> bool:
        """Delete a task. An occurrence already queued for it will not post."""
        return await self._store.delete_task(task_id)

    async def pause(self, task_id: str, *, reason: str | None = None) -> bool:
        return await self._store.set_task_state(task_id, state=PAUSED, error=reason)

    async def resume(self, task_id: str) -> bool:
        return await self._store.set_task_state(task_id, state=ACTIVE)

    async def finish(self, task_id: str) -> bool:
        return await self._store.set_task_state(task_id, state=DONE)

    async def record_failure(self, task_id: str, *, error: str) -> int:
        return await self._store.record_task_failure(task_id, error=error)

    async def occurrences(self, moment: datetime) -> list[Occurrence]:
        """The next firing of every active task. The clock's half of this.

        Per task, because one whose expression or zone no longer reads must not
        stop the tasks behind it — the same isolation the spec loop already
        has, for the same reason, and here it matters more: these expressions
        were written by a model reading what somebody typed.
        """
        found = []
        for task in await self.all(state=ACTIVE):
            try:
                fire_at = task.schedule().next_after(moment)
            except CronError:
                log.exception(
                    "task %s has a schedule that no longer reads (%s)", task.id, task.cron
                )
                continue
            found.append(
                Occurrence(
                    job_id=occurrence_id(task.id, fire_at),
                    kind=TASK_KIND,
                    fire_at=fire_at,
                    payload={"task_id": task.id},
                    label=f"task {task.id} ({task.label})",
                )
            )
        return found


def task_handler(
    store: Store,
    settings: TaskSettings | None = None,
    *,
    inbox: Inbox | None = None,
    notify: TaskNotifier | None = None,
) -> JobHandler:
    """Run one occurrence: put the task's prompt in the inbox as a message.

    Deliberately not "run the agent". Everything that makes a turn a turn —
    one actor per conversation, retrieval, the surface's reply path, the
    failure handling that redelivers a message whose model call fell over —
    already exists on the inbox side of the queue, and a second path to it
    would be a second set of all of that to keep correct.

    What this counts as a failure is therefore narrow, and worth being plain
    about: the enqueue, not the turn. A model that times out answering a
    standing task is an inbox failure with the inbox's own retries, and it is
    not what pauses a task. What pauses a task is the run never reaching the
    inbox at all — a deleted session, a row that stopped parsing — which is the
    failure that would otherwise repeat silently forever.
    """
    tasks = Tasks(store, settings)
    queue = inbox or Inbox(store)

    async def run(job: Job) -> None:
        task_id = str(job.payload.get("task_id") or "")
        task = await tasks.get(task_id)
        if task is None:
            # Cancelled between the clock queueing this and the drainer
            # reaching it. Deleting a task stops it, so this run is not work
            # that was lost — it is work that was called off.
            log.info("task %s is gone; dropping the run queued for it", task_id or "?")
            return
        if task.state != ACTIVE:
            log.info("task %s is %s; skipping this run", task.id, task.state)
            return
        try:
            await _deliver(queue, task, job)
        except Exception as exc:
            # Once per occurrence, not once per attempt. The job retries on the
            # queue's own backoff, and counting each of those three tries as a
            # failed *run* would pause a task after two bad fire times while
            # claiming it had six.
            if job.attempts <= 1:
                await _blame(tasks, task, exc, notify=notify)
            raise
        await store.record_task_run(task.id, job_id=job.id)
        if task.fire_once:
            await tasks.finish(task.id)
        log.info("task %s queued a turn in %s", task.id, task.session_id)

    return run


async def _deliver(queue: Inbox, task: Task, job: Job) -> None:
    """Hand the task's prompt to the inbox as though somebody had said it.

    `external_id` is the job's id, which is the fire time. A job retried after
    a partial failure re-enqueues the same event id, and the inbox's UNIQUE
    constraint turns that into one answer rather than two.
    """
    await queue.enqueue(
        InboundEvent(
            source=task.surface,
            external_id=job.id,
            session_id=task.session_id,
            text=task.prompt,
            scope=task.scope,
            author=task.owner,
            channel=task.channel,
            reply_to=task.reply_to,
            origin="scheduled",
        )
    )


async def _blame(tasks: Tasks, task: Task, exc: Exception, *, notify: TaskNotifier | None) -> None:
    """Count a failed run, and stop the task once it has failed enough.

    Told once, on the run that crosses the threshold, and never again: the
    task is paused by the same call, so there is no second crossing to
    announce. A notifier that itself fails is logged and swallowed — a task
    that could not be paused because nobody could be told would keep failing,
    which is the outcome this exists to prevent.
    """
    reason = f"{type(exc).__name__}: {exc}"
    failures = await tasks.record_failure(task.id, error=reason)
    if failures < tasks.settings.disable_after_failures:
        return
    await tasks.pause(task.id, reason=reason)
    log.error("task %s paused after %d consecutive failures: %s", task.id, failures, reason)
    if notify is None:
        return
    try:
        await notify(
            task,
            f"I have paused a scheduled task after {failures} failed run(s) — "
            f"{task.prompt!r} ({task.label}). The last error was: {reason}",
        )
    except Exception:
        log.exception("could not tell %s that task %s was paused", task.owner, task.id)


def render_fires(fires: Sequence[datetime], tz_name: str | None) -> list[str]:
    """Fire times as somebody reads them: in the task's own zone, named.

    UTC instants are what the scheduler deals in and are useless as
    confirmation — "2026-09-08T00:00:00+00:00" is not something a person in
    Seoul can check against "every weekday at nine".
    """
    zone = ZoneInfo(tz_name) if tz_name else UTC
    suffix = tz_name or "UTC"
    return [f"{fire.astimezone(zone).strftime('%a %d %b %H:%M')} {suffix}" for fire in fires]
